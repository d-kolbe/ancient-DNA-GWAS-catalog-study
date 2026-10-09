#!/usr/bin/env python3
"""
derive_category_cues.py
=======================
Reconstructs the keyword cues (Layer 1) that reproduce the category assignments of the
frozen vocabulary (gwas_categories_v2.yaml).

WHY THIS SCRIPT EXISTS
----------------------
The original keyword rules that assigned GWAS Catalog traits to categories were written
in an earlier round (LLM-drafted regexes, 108 categories) and were not kept. What we do
have is their result: the frozen vocabulary. This script works backwards from it:

  For every category it looks for short word/phrase cues ("hdl", "seropositivity",
  "chronic kidney disease", ...) that occur in the names of the category's members and
  are mostly specific to that category, and selects a small set of cues that together
  cover (nearly) all members (greedy set cover).

The result (category_cues.yaml) is a readable, editable rule table. Members that no
sufficiently specific cue covers are written to supplement_memberships.tsv (curation
layer, applied by build_gwas_categories.py).

HONESTY NOTE
------------
Cues are derived from the frozen answer, so they describe the vocabulary; they do not
prove that the original procedure was rule-based. How well they generalise is estimated
by cross-validation (--cv): cues are learned on 80% of the traits and their recall is
measured on the held-out 20%.

Usage
-----
  python derive_category_cues.py --frozen gwas_categories_v2.yaml \
      --defs category_definitions.yaml --out category_cues.yaml \
      --supplement supplement_memberships.tsv [--cv]
"""
import argparse
import collections
import random
import re

import yaml

# Single words that are too generic to be a cue on their own (they can still be part of
# a phrase, e.g. "chronic kidney disease").
GENERIC_UNIGRAMS = set("""
disease diseases disorder disorders syndrome risk response function functions factor
factors type index score count counts trait traits study analysis multivariate mtag
pleiotropy interaction adjusted time age years percentage percent concentration
concentrations plasma serum urine urinary blood protein proteins human cell cells
acid acids total mean maximum minimum baseline change rate volume size width length
levels level ratio measurement measurements chronic acute severe primary secondary
symptoms symptom self-reported reported medication treatment treated related related
""".split())
STOP = set("""the of and in to for with by on at a an or vs versus ukb data field phecode icd10
inv norm normal transformed after before from as is are be not no than that this these
those using use its it into per between within during other others""".split())


def tokens(text):
    """Words = maximal runs of letters/digits (punctuation and hyphens separate words)."""
    return re.findall(r"[a-z0-9]+", text.lower())


def grams(text):
    """All word 1-3-grams of a trait name that could serve as a cue."""
    w = tokens(text)
    out = set()
    for n in (1, 2, 3):
        for i in range(len(w) - n + 1):
            g = w[i:i + n]
            if g[0] in STOP or g[-1] in STOP:
                continue
            if n == 1 and (g[0] in GENERIC_UNIGRAMS or len(g[0]) < 3 or g[0].isdigit()):
                continue
            out.add(" ".join(g))
    return out


def learn_cues(members, universe_index, pmin, min_new, target):
    """Greedy set cover: repeatedly pick the cue that covers most still-uncovered members,
    among cues whose precision (share of all traits containing the cue that are members
    of this category) is at least pmin."""
    uncovered = set(members)
    cands = {g: ts & members for g, ts in universe_index.items() if len(ts & members) >= 2}
    chosen = []
    while uncovered and len(uncovered) > (1 - target) * len(members):
        best = None
        for g, hit in cands.items():
            new = len(hit & uncovered)
            if new < min_new:
                continue
            prec = len(hit) / len(universe_index[g])
            if prec < pmin:
                continue
            key = (new, prec, len(g))
            if best is None or key > best[0]:
                best = (key, g, hit, prec)
        if best is None:
            break
        _, g, hit, prec = best
        chosen.append({"cue": g, "covers": len(hit), "precision": round(prec, 2)})
        uncovered -= hit
    return chosen, uncovered


def build_index(traits):
    inv = collections.defaultdict(set)
    for t in traits:
        for g in grams(t):
            inv[g].add(t)
    return inv


def fit(frozen, pmin, min_new, target, keep_traits=None):
    cats = list(frozen)
    universe = sorted({t for ts in frozen.values() for t in ts})
    if keep_traits is not None:
        universe = [t for t in universe if t in keep_traits]
    inv = build_index(universe)
    result = {}
    for c in cats:
        members = set(frozen[c]) & set(universe)
        result[c] = learn_cues(members, inv, pmin, min_new, target)
    return result


def cue_regex(cue):
    """A cue matches whole words that follow each other, separated only by punctuation/spaces.
    (Same definition as cue_pattern() in build_gwas_categories.py.)"""
    return re.compile(r"(?<![a-z0-9])" + r"[^a-z0-9]+".join(re.escape(w) for w in cue.split()) + r"(?![a-z0-9])")


def matches(cues, text):
    tl = text.lower()
    return any(cue_regex(c["cue"]).search(tl) for c in cues)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--defs", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--supplement", required=True)
    ap.add_argument("--pmin", type=float, default=0.7, help="minimum cue precision")
    ap.add_argument("--min-new", type=int, default=3, help="minimum newly covered members per cue")
    ap.add_argument("--target", type=float, default=0.985, help="stop when this share is covered")
    ap.add_argument("--cv", action="store_true", help="also report 80/20 cross-validated recall")
    args = ap.parse_args()

    frozen = yaml.safe_load(open(args.frozen, encoding="utf-8"))
    defs = yaml.safe_load(open(args.defs, encoding="utf-8")) if args.defs else {}
    fitted = fit(frozen, args.pmin, args.min_new, args.target)

    n_all = sum(len(v) for v in frozen.values())
    n_unc = sum(len(u) for _, u in fitted.values())
    print("categories: %d | memberships: %d | covered by cues: %d (%.1f%%) | supplement: %d"
          % (len(frozen), n_all, n_all - n_unc, 100.0 * (n_all - n_unc) / n_all, n_unc))

    if args.cv:
        universe = sorted({t for ts in frozen.values() for t in ts})
        rnd = random.Random(1)
        rnd.shuffle(universe)
        cut = int(0.8 * len(universe))
        train, test = set(universe[:cut]), set(universe[cut:])
        cv_fit = fit(frozen, args.pmin, args.min_new, args.target, keep_traits=train)
        hit = tot = 0
        for c, ts in frozen.items():
            cues = cv_fit[c][0]
            for t in ts:
                if t in test:
                    tot += 1
                    hit += matches(cues, t)
        print("cross-validation: held-out memberships %d | recovered by cues learned on the other 80%%: %d (%.1f%%)"
              % (tot, hit, 100.0 * hit / tot))

    out = {}
    with open(args.supplement, "w", encoding="utf-8") as sup:
        sup.write("trait\tcategory\treason\n")
        for c in frozen:
            cues, unc = fitted[c]
            d = defs.get(c, {})
            out[c] = {
                "definition": d.get("definition", ""),
                "n_frozen_members": len(frozen[c]),
                "n_covered_by_cues": len(frozen[c]) - len(unc),
                "cues": cues,
            }
            for t in sorted(unc):
                sup.write("%s\t%s\tno sufficiently specific cue\n" % (t, c))

    header = (
        "# Keyword cues per category (Layer 1 of build_gwas_categories.py).\n"
        "# Reconstructed from the frozen vocabulary by derive_category_cues.py; see that script.\n"
        "# cue        : lower-case word or phrase, matched on word boundaries\n"
        "# covers     : number of frozen members of this category containing the cue\n"
        "# precision  : share of ALL frozen traits containing the cue that are members of this category\n"
        "# Edit freely: add cues to raise recall, delete cues that over-assign.\n\n"
    )
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump({"categories": out}, f, sort_keys=False, allow_unicode=True,
                       default_flow_style=False, width=200)
    print("wrote", args.out, "and", args.supplement)


if __name__ == "__main__":
    main()
