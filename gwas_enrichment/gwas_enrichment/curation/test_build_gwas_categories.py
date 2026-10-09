#!/usr/bin/env python3
"""
Tests for the category builder.

  python test_build_gwas_categories.py                       # rule tests only
  FROZEN=path/to/gwas_categories_v2.yaml python test_build_gwas_categories.py   # + end-to-end test

The end-to-end test builds a small pseudo-catalog from the frozen vocabulary plus a few
known 'trap' traits, runs build_gwas_categories.py and checks that
  * the frozen memberships are recovered (recall),
  * every trap is handled by the intended rule.
It cannot measure precision on the real catalog; use --frozen with the real catalog for that.
"""
import csv
import os
import subprocess
import sys
import tempfile
import unittest

import yaml

import curation_rules as R

HERE = os.path.dirname(os.path.abspath(__file__))


class RuleTests(unittest.TestCase):
    def test_stage2_drops_interactions_and_drug_response(self):
        bp = "Cardiovascular - Blood Pressure and Hypertension"
        self.assertTrue(R.filter_stage2("Diastolic blood pressure x alcohol consumption interaction (2df test)", bp))
        self.assertTrue(R.filter_stage2("Response to hydrochlorothiazide in hypertension", bp))

    def test_stage2_keeps_pleiotropy(self):
        self.assertIsNone(R.filter_stage2("Body mass index or breast cancer (pleiotropy)", "Anthropometry - Weight and BMI"))

    def test_stage2_adverse_events_only_outside_pharmacogenomics(self):
        t = "Cardiovascular adverse events in carfilzomib-treated multiple myeloma"
        self.assertTrue(R.filter_stage2(t, "Cancer - Hematological"))
        self.assertIsNone(R.filter_stage2(t, "Pharmacogenomics - Drug Response"))

    def test_stage4_hip_disease_not_body_composition(self):
        self.assertTrue(R.rule_hip_disease_in_body_comp("Hip fracture")[0])
        self.assertFalse(R.rule_hip_disease_in_body_comp("Hip circumference")[0])

    def test_stage4_diastolic_lv_not_blood_pressure(self):
        self.assertTrue(R.rule_lv_structure_in_bp("Left ventricular end-diastolic volume")[0])
        self.assertFalse(R.rule_lv_structure_in_bp("Diastolic blood pressure")[0])

    def test_stage5_craniofacial_height(self):
        self.assertEqual(R.stage5("Facial morphology (nose height)", "Anthropometry - Height and Growth")[0], "remove")
        self.assertIsNone(R.stage5("Standing height", "Anthropometry - Height and Growth"))

    def test_stage5_type1_in_t2d(self):
        cat = "Diabetes - Type 2 Diabetes and Glucose Metabolism"
        self.assertEqual(R.stage5("Type 1 diabetes nephropathy", cat)[0], "remove")
        self.assertIsNone(R.stage5("Type 1 or type 2 diabetes", cat))


FROZEN = os.environ.get("FROZEN")
TRAPS = ["Facial morphology (nose height)", "Diastolic blood pressure x alcohol consumption interaction (2df test)",
         "Response to hydrochlorothiazide in hypertension", "Hip fracture",
         "Anthracycline-induced cardiotoxicity in breast cancer", "Tanner 1 to Tanner 2 pubertal stage transition"]


@unittest.skipUnless(FROZEN and os.path.exists(FROZEN), "set FROZEN=<gwas_categories_v2.yaml> to run the end-to-end test")
class EndToEnd(unittest.TestCase):
    def test_pipeline(self):
        frozen = yaml.safe_load(open(FROZEN, encoding="utf-8"))
        traits = sorted({t for ts in frozen.values() for t in ts})
        with tempfile.TemporaryDirectory() as tmp:
            cat = os.path.join(tmp, "catalog.tsv")
            with open(cat, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f, delimiter="\t")
                w.writerow(["DISEASE/TRAIT", "P-VALUE"])
                for t in traits + TRAPS:
                    w.writerow([t, "1e-9"])
            out = os.path.join(tmp, "gen.yaml")
            cmd = [sys.executable, os.path.join(HERE, "build_gwas_categories.py"), "--catalog", cat,
                   "--cues", os.path.join(HERE, "category_cues.yaml"),
                   "--supplement", os.path.join(HERE, "supplement_memberships.tsv"),
                   "--overrides", os.path.join(HERE, "manual_overrides.tsv"),
                   "--out", out, "--audit", os.path.join(tmp, "audit.tsv"),
                   "--frozen", FROZEN, "--report-dir", os.path.join(tmp, "cmp")]
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
            self.assertEqual(r.returncode, 0, r.stderr)
            gen = yaml.safe_load(open(out, encoding="utf-8"))
            pairs = {(t, c) for c, ts in gen.items() for t in ts}
            fro = {(t, c) for c, ts in frozen.items() for t in ts}
            recall = len(pairs & fro) / len(fro)
            self.assertGreaterEqual(recall, 0.999)
            present = {t for _, t in pairs}
            for t in TRAPS:
                self.assertNotIn(t, present, t)
            print("\nend-to-end: recall %.2f%%, precision %.1f%% (inside the frozen trait universe)"
                  % (100 * recall, 100.0 * len(pairs & fro) / len(pairs)))


if __name__ == "__main__":
    unittest.main(verbosity=1)
