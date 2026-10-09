#!/usr/bin/env python3
"""
refine_cues_from_report.py
==========================
Second round of the cue reconstruction, driven by a real run of build_gwas_categories.py.

INPUT   compare_extra.tsv   memberships the generator produced that the frozen vocabulary does not have
WHAT    For every category, extra memberships whose trait DOES exist in the frozen vocabulary (under
        another category) are known collisions: the cue that fired is too broad. This script learns
        short "collision guards" (word or phrase) that remove many of these collisions and few (or
        no) true members of the category, and writes them as `exclude:` entries in category_cues.yaml.
        True members that a guard would lose are moved to the supplement (curation layer), so recall
        against the frozen vocabulary is unchanged.
NOT USED  extra memberships whose trait is absent from the frozen vocabulary altogether. Those are
        traits the vocabulary never contained (e.g. added to the catalog after the vocabulary was
        built); guarding against them would only memorise the catalog snapshot.

Usage
-----
  python refine_cues_from_report.py --cues category_cues.yaml --supplement supplement_memberships.tsv \
      --frozen ../resources/gwas_categories_v2.yaml --extra compare/compare_extra.tsv \
      --out-cues category_cues.yaml --out-supplement supplement_memberships.tsv
"""
import argparse
import collections
import csv
import re

import yaml

from build_gwas_categories import cue_pattern, norm
from derive_category_cues import grams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cues", required=True)
    ap.add_argument("--supplement", required=True)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--extra", required=True)
    ap.add_argument("--out-cues", required=True)
    ap.add_argument("--out-supplement", required=True)
    ap.add_argument("--max-member-loss", type=float, default=0.2,
                    help="a guard may remove at most this many true members per collision it removes")
    ap.add_argument("--min-benefit", type=int, default=2)
    args = ap.parse_args()

    cues = yaml.safe_load(open(args.cues, encoding="utf-8"))
    frozen = yaml.safe_load(open(args.frozen, encoding="utf-8"))
    fkeys = {norm(t) for ts in frozen.values() for t in ts}
    extras = collections.defaultdict(list)          # category -> [(trait, is_collision)]
    for r in csv.DictReader(open(args.extra, encoding="utf-8"), delimiter="\t"):
        extras[r["category"]].append((r["trait"], norm(r["trait"]) in fkeys))

    new_sup = []
    stats = collections.Counter()
    for cat, d in cues["categories"].items():
        inc = re.compile("|".join(cue_pattern(c["cue"]) for c in d["cues"])) if d["cues"] else None
        collisions = [t for t, is_col in extras.get(cat, []) if is_col]
        others = [t for t, is_col in extras.get(cat, []) if not is_col]
        members = [t for t in frozen[cat] if inc is not None and inc.search(norm(t))]   # members reached by cues
        if not collisions or inc is None:
            continue
        gram_sets = {t: grams(t) for t in collisions + members + others}
        chosen, left = [], set(collisions)
        while left:
            counts = collections.Counter(g for t in left for g in gram_sets[t])
            best = None
            for g, benefit in counts.items():
                if benefit < args.min_benefit:
                    continue
                cost = sum(1 for t in members if g in gram_sets[t])
                if cost > args.max_member_loss * benefit:
                    continue
                key = (benefit - 2 * cost, benefit, len(g))
                if best is None or key > best[0]:
                    best = (key, g, benefit, cost)
            if best is None:
                break
            _, g, benefit, cost = best
            chosen.append({"cue": g, "removes_collisions": benefit, "removes_members": cost,
                           "removes_other_extras": sum(1 for t in others if g in gram_sets[t])})
            left = {t for t in left if g not in gram_sets[t]}
        if chosen:
            d["exclude"] = chosen
            exc = re.compile("|".join(cue_pattern(c["cue"]) for c in chosen))
            for t in members:
                if exc.search(norm(t)):
                    new_sup.append((t, cat, "cue excluded by a collision guard; kept as curated member"))
            stats["collisions"] += len(collisions)
            stats["collisions_left"] += len(left)
            stats["guards"] += len(chosen)
            stats["members_to_supplement"] += sum(1 for t in members if exc.search(norm(t)))
            stats["other_extras_removed"] += sum(c["removes_other_extras"] for c in chosen)

    header = open(args.cues, encoding="utf-8").read().split("categories:")[0]
    header += ("# exclude   : collision guards learned by refine_cues_from_report.py from a real run;\n"
               "#              a trait containing one of them is not assigned to this category by cues.\n\n")
    with open(args.out_cues, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump({"categories": cues["categories"]}, f, sort_keys=False, allow_unicode=True,
                       default_flow_style=False, width=200)
    rows = [r for r in csv.DictReader(open(args.supplement, encoding="utf-8"), delimiter="\t")]
    have = {(norm(r["trait"]), r["category"]) for r in rows}
    with open(args.out_supplement, "w", encoding="utf-8") as f:
        f.write("trait\tcategory\treason\n")
        for r in rows:
            f.write("%s\t%s\t%s\n" % (r["trait"], r["category"], r["reason"]))
        for t, c, why in new_sup:
            if (norm(t), c) not in have:
                f.write("%s\t%s\t%s\n" % (t, c, why))
    print("collisions: %d | removed by guards: %d | left: %d | guards: %d"
          % (stats["collisions"], stats["collisions"] - stats["collisions_left"], stats["collisions_left"], stats["guards"]))
    print("true members moved to the supplement: %d | additional (non-collision) extras also removed: %d"
          % (stats["members_to_supplement"], stats["other_extras_removed"]))


if __name__ == "__main__":
    main()
