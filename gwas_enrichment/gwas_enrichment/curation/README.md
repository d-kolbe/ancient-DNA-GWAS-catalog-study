# curation

How the trait categories (`resources/gwas_categories_v2.yaml`, 65 categories) are built from the GWAS Catalog.
See `CATEGORY_CURATION.md` for the full process.

```
python build_gwas_categories.py --catalog gwas_catalog_GRCh37.tsv \
    --cues category_cues.yaml --supplement supplement_memberships.tsv --overrides manual_overrides.tsv \
    --exclusions exclusions.tsv \
    --out gwas_categories_generated.yaml --audit build_audit.tsv
```

| step | what | file |
|---|---|---|
| 1 | category criteria assign each trait name to one or more categories | `category_cues.yaml` |
| 2 | filter: interaction terms, drug response, adverse drug reactions, misplaced survival | `curation_rules.py` |
| 3 | collision review, e.g. "hip fracture" is not body composition, craniofacial "height" is not stature | `curation_rules.py` |
| 4 | curation: members the criteria miss, the Sensory split, specific keyword collisions | `supplement_memberships.tsv`, `manual_overrides.tsv`, `exclusions.tsv` |

Every removal and addition is written to the audit table (`--audit`).

`category_definitions.yaml` states what each category means. `derive_category_cues.py` is the tool used to build
and update `category_cues.yaml`. `test_build_gwas_categories.py` runs the rule tests. `--frozen <file>` compares
the output against another category file (internal QA; not part of the normal run).

Trait names are matched case- and whitespace-insensitively, as in the enrichment pipeline.
