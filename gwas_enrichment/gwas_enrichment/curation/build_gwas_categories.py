#!/usr/bin/env python3
"""
build_gwas_categories.py
========================
Assigns GWAS Catalog traits to the 65 categories used in the enrichment analysis. Every
assignment, filter and removal is an explicit, logged rule (curation_rules.py); every
decision is written to an audit table, so any trait's categorization can be traced.

PIPELINE
--------
  GWAS Catalog TSV (DISEASE/TRAIT column)
    |
    | Stage 0  unique trait names
    v
  Stage 1  category assignment    category_cues.yaml        a trait can join several categories
    v                                                        (pleiotropy is kept)
  Stage 2  filter                                            interaction terms, drug response,
    v                                                        adverse drug reactions, misplaced survival
  Stage 3  collision review                                  removes keyword matches that don't hold
    v                                                        up on inspection, e.g. 'hip fracture' out
    v                                                        of Body Composition, craniofacial 'height'
    v                                                        out of Height and Growth
  Stage 4  curation                                          supplement_memberships.tsv (adds traits
    v                                                        the criteria miss), manual_overrides.tsv
    v                                                        (the Sensory - Hearing / Taste and Smell
    v                                                        split), optional --exclusions
  gwas_categories_generated.yaml  +  audit table

Categories outside these 65 (e.g. proteomics, metabolomics, gene expression, genomic
features, brain imaging) are simply not defined in category_cues.yaml, so no trait can be
assigned to them; no separate step is needed for this.

Pass --frozen to compare the output against another category file, for internal QA.

USAGE
-----
  python build_gwas_categories.py \\
      --catalog gwas_catalog_GRCh37.tsv \\
      --cues category_cues.yaml \\
      --supplement supplement_memberships.tsv \\
      --overrides manual_overrides.tsv \\
      --out gwas_categories_generated.yaml \\
      --audit build_audit.tsv

Add --no-supplement to see what the category criteria alone produce.

Matching is done on a normalised form of the trait name (lower case, curly quotes and
non-breaking spaces replaced), so typographic differences between files do not matter.
ASCII-only source on purpose (cluster compile restriction).
"""
import argparse
import collections
import csv
import os
import re
import sys

import yaml

import curation_rules as R


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def norm(text):
    """Normalised key used for all matching: typographic quotes/spaces unified, doubled
    apostrophes (a known typo in a few category entries) collapsed, lower case."""
    text = (text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"')
            .replace("\u201d", '"').replace("\u00a0", " ").replace("''", "'"))
    return " ".join(text.split()).lower()


def read_tsv(path):
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_catalog_traits(path, col):
    """Unique trait names of the catalog (every DISEASE/TRAIT value that appears at least
    once, regardless of the row's p-value, ancestry or risk-allele reporting -- the
    vocabulary is built from which traits the catalog names, not from how strong or how
    European any single association row is). Returns {normalised key: raw spelling}."""
    csv.field_size_limit(sys.maxsize)
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        fields = rdr.fieldnames or []
        if col not in fields:
            sys.exit("column %r not found in %s; columns are: %s" % (col, path, fields))

        n_rows = 0
        traits = collections.OrderedDict()
        for row in rdr:
            n_rows += 1
            t = (row.get(col) or "").strip()
            if t and norm(t) not in traits:
                traits[norm(t)] = t
    print("Stage 0: %d rows read, %d unique trait names" % (n_rows, len(traits)), file=sys.stderr)
    return traits


def cue_pattern(cue):
    """A cue is a lower-case word or phrase. It matches whole words that follow each other,
    separated only by spaces or punctuation ("waist hip" matches "waist-hip", "waist (hip)")."""
    words = cue.lower().split()
    return r"(?<![a-z0-9])" + r"[^a-z0-9]+".join(re.escape(w) for w in words) + r"(?![a-z0-9])"


def load_cues(path):
    """Read category_cues.yaml -> ordered {category: (include_regex, exclude_regex or None)}."""
    raw = yaml.safe_load(open(path, encoding="utf-8"))["categories"]
    out = collections.OrderedDict()
    for cat, d in raw.items():
        inc = [cue_pattern(c["cue"] if isinstance(c, dict) else c) for c in (d.get("cues") or [])]
        exc = [cue_pattern(c["cue"] if isinstance(c, dict) else c) for c in (d.get("exclude") or [])]  # collision guards
        out[cat] = (re.compile("|".join(inc)) if inc else None,
                    re.compile("|".join(exc)) if exc else None)
    return out


class Result:
    """Category membership with provenance and an audit trail."""

    def __init__(self, categories):
        self.members = {c: collections.OrderedDict() for c in categories}  # cat -> {key: (raw, source)}
        self.audit = []

    def add(self, cat, key, raw, source):
        self.members[cat].setdefault(key, (raw, source))

    def drop(self, cat, key, stage, reason, action="remove"):
        if key in self.members[cat]:
            raw = self.members[cat].pop(key)[0]
            self.audit.append((raw, cat, stage, action, reason))
            return True
        return False

    def n(self):
        return sum(len(v) for v in self.members.values())


# -----------------------------------------------------------------------------
# stages
# -----------------------------------------------------------------------------
def stage1_keyword_assignment(traits, cues, res):
    """Stage 1: a trait joins every category one of whose cues occurs in its name
    (and none of whose optional exclude-cues does)."""
    for key, raw in traits.items():
        for cat, (inc, exc) in cues.items():
            if inc is None or not inc.search(key):
                continue
            if exc is not None and exc.search(key):
                continue
            res.add(cat, key, raw, "cue")


def stage2_filter(res):
    """Stage 2: 'moderate, keep pleiotropy' filter (interaction terms, drug response, ...)."""
    for cat in list(res.members):
        for key, (raw, _) in list(res.members[cat].items()):
            why = R.filter_stage2(raw, cat)
            if why:
                res.drop(cat, key, "2-filter", why)


def stage4_collision_rules(res):
    """Stage 4: keyword-collision rules from clean_categories.py."""
    for cat, rules in R.CATEGORY_RULES.items():
        for key, (raw, _) in list(res.members.get(cat, {}).items()):
            for rule in rules:
                remove, why = rule(raw)
                if remove:
                    res.drop(cat, key, "4-collision", why)
                    break


def stage5_collision_rules(res):
    """Stage 5: keyword-collision rules from clean_gwas_categories.py (stage A)."""
    for cat in list(res.members):
        for key, (raw, _) in list(res.members[cat].items()):
            hit = R.stage5(raw, cat)
            if not hit:
                continue
            action, why = hit
            if action == "remove":
                res.drop(cat, key, "5-collision", why)
            else:  # 'flag': logged for review, kept
                res.audit.append((raw, cat, "5-collision", "flag", why))


def stage6_supplement(res, traits, path, log):
    """Stage 6a: members of the frozen vocabulary that no keyword cue reaches."""
    added = missing = 0
    for row in read_tsv(path):
        key = norm(row["trait"])
        cat = row["category"]
        if cat not in res.members:
            continue
        if key in traits:
            if key not in res.members[cat]:
                res.add(cat, key, traits[key], "supplement")
                added += 1
        else:
            missing += 1
    log("  supplement: %d added, %d not in this catalog" % (added, missing))


def stage6_overrides(res, traits, path, log):
    """Stage 6b: hand-made curation decisions (op = remove | add)."""
    rem = add = 0
    for row in read_tsv(path):
        op, key, cat = row["op"].strip().lower(), norm(row["trait"]), row["category"]
        if cat not in res.members:
            continue
        if op == "remove":
            rem += res.drop(cat, key, "6-manual", row["reason"])
        elif op == "add" and key in traits:
            if key not in res.members[cat]:
                res.add(cat, key, traits[key], "override")
                add += 1
                res.audit.append((traits[key], cat, "6-manual", "add", row["reason"]))
    log("  manual overrides: %d removed, %d added" % (rem, add))


def stage6_exclusions(res, path, log):
    """Stage 6c (optional): further removals decided after reviewing the comparison report."""
    rem = 0
    for row in read_tsv(path):
        rem += res.drop(row["category"], norm(row["trait"]), "6-exclusion", row.get("reason") or "excluded after review")
    log("  exclusions: %d removed" % rem)


# -----------------------------------------------------------------------------
# output and comparison
# -----------------------------------------------------------------------------
def write_yaml(res, path, comment_lines):
    with open(path, "w", encoding="utf-8") as f:
        for line in comment_lines:
            f.write("# " + line + "\n")
        f.write("\n")
        out = collections.OrderedDict((c, sorted(raw for raw, _ in m.values())) for c, m in res.members.items())
        yaml.safe_dump(dict(out), f, sort_keys=False, allow_unicode=True, default_flow_style=False, width=200)


def write_audit(res, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("trait\tcategory\tstage\taction\treason\n")
        for row in res.audit:
            f.write("\t".join(row) + "\n")


def compare(res, traits, frozen_path, report_dir, log):
    """Compare generated memberships with the frozen vocabulary (pairs of trait x category)."""
    frozen = yaml.safe_load(open(frozen_path, encoding="utf-8"))
    gen = {(k, c) for c, m in res.members.items() for k in m}
    fro = {(norm(t), c) for c, ts in frozen.items() for t in ts}
    fro_in_catalog = {p for p in fro if p[0] in traits}
    tp = gen & fro
    log("")
    log("COMPARISON WITH %s" % os.path.basename(frozen_path))
    log("  frozen memberships: %d (%d of them with a trait name present in this catalog)" % (len(fro), len(fro_in_catalog)))
    log("  generated memberships: %d" % len(gen))
    log("  agreeing: %d | precision %.1f%% | recall %.1f%% (recall among frozen memberships present in the catalog: %.1f%%)"
        % (len(tp), 100.0 * len(tp) / max(1, len(gen)), 100.0 * len(tp) / max(1, len(fro)),
           100.0 * len(tp) / max(1, len(fro_in_catalog))))
    os.makedirs(report_dir, exist_ok=True)
    src = {(k, c): s for c, m in res.members.items() for k, (_, s) in m.items()}
    with open(os.path.join(report_dir, "compare_extra.tsv"), "w", encoding="utf-8") as f:
        f.write("trait\tcategory\tassigned_by\n")
        for k, c in sorted(gen - fro, key=lambda p: (p[1], p[0])):
            f.write("%s\t%s\t%s\n" % (res.members[c][k][0], c, src[(k, c)]))
    frozen_raw = {(norm(t), c): t for c, ts in frozen.items() for t in ts}
    with open(os.path.join(report_dir, "compare_missing.tsv"), "w", encoding="utf-8") as f:
        f.write("trait\tcategory\tin_catalog\n")
        for k, c in sorted(fro - gen, key=lambda p: (p[1], p[0])):
            f.write("%s\t%s\t%s\n" % (frozen_raw[(k, c)], c, "yes" if k in traits else "no"))
    by_cat = collections.OrderedDict()
    for c in res.members:
        g = {p for p in gen if p[1] == c}
        r_ = {p for p in fro if p[1] == c}
        by_cat[c] = (len(r_), len(g), len(g & r_))
    with open(os.path.join(report_dir, "compare_by_category.tsv"), "w", encoding="utf-8") as f:
        f.write("category\tfrozen\tgenerated\tagreeing\tprecision\trecall\n")
        for c, (nr, ng, nt) in by_cat.items():
            f.write("%s\t%d\t%d\t%d\t%.3f\t%.3f\n" % (c, nr, ng, nt, nt / max(1, ng), nt / max(1, nr)))
    worst = sorted(by_cat.items(), key=lambda kv: kv[1][1] - kv[1][2], reverse=True)[:5]
    log("  categories with most extra memberships: " + "; ".join("%s (+%d)" % (c, v[1] - v[2]) for c, v in worst))
    log("  reports written to %s/ (compare_extra.tsv, compare_missing.tsv, compare_by_category.tsv)" % report_dir)


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", required=True, help="GWAS Catalog associations TSV")
    ap.add_argument("--trait-col", default="DISEASE/TRAIT")
    ap.add_argument("--cues", required=True, help="category_cues.yaml (stage 1)")
    ap.add_argument("--supplement", default=None, help="supplement_memberships.tsv (stage 4)")
    ap.add_argument("--no-supplement", action="store_true", help="skip supplement (category criteria only)")
    ap.add_argument("--overrides", default=None, help="manual_overrides.tsv (stage 4)")
    ap.add_argument("--exclusions", default=None, help="optional TSV (trait, category, reason) of extra removals")
    ap.add_argument("--out", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--frozen", default=None, help="frozen vocabulary to compare against")
    ap.add_argument("--report-dir", default="compare")
    args = ap.parse_args()

    log = lambda s: print(s, file=sys.stderr)
    traits = read_catalog_traits(args.catalog, args.trait_col)

    cues = load_cues(args.cues)
    res = Result(list(cues))

    stage1_keyword_assignment(traits, cues, res)
    n1 = res.n()
    log("Stage 1: %d trait x category assignments from keyword cues" % n1)
    stage2_filter(res)
    n2 = res.n()
    log("Stage 2: %d removed (interaction / drug response / adverse events / misplaced survival)" % (n1 - n2))
    stage4_collision_rules(res)
    n4 = res.n()
    log("Stage 3a: %d removed (collision review)" % (n2 - n4))
    stage5_collision_rules(res)
    n5 = res.n()
    log("Stage 3b: %d removed (collision review)" % (n4 - n5))
    log("Stage 4: curation")
    if args.supplement and not args.no_supplement:
        stage6_supplement(res, traits, args.supplement, log)
    if args.overrides:
        stage6_overrides(res, traits, args.overrides, log)
    if args.exclusions:
        stage6_exclusions(res, args.exclusions, log)

    distinct = len({k for m in res.members.values() for k in m})
    log("Result: %d categories, %d memberships, %d distinct traits" % (len(res.members), res.n(), distinct))

    write_yaml(res, args.out, [
        "GWAS Catalog trait categories generated by build_gwas_categories.py",
        "catalog: %s | cues: %s | supplement: %s | overrides: %s"
        % (os.path.basename(args.catalog), os.path.basename(args.cues),
           "off" if (args.no_supplement or not args.supplement) else os.path.basename(args.supplement),
           os.path.basename(args.overrides) if args.overrides else "none"),
        "categories: %d | memberships: %d | distinct traits: %d" % (len(res.members), res.n(), distinct),
        "a trait may belong to several categories (pleiotropy is kept)",
    ])
    write_audit(res, args.audit)
    log("wrote %s and %s" % (args.out, args.audit))

    if args.frozen:
        compare(res, traits, args.frozen, args.report_dir, log)


if __name__ == "__main__":
    main()
