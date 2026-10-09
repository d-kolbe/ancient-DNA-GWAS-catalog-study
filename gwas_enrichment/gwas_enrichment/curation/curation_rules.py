#!/usr/bin/env python3
"""
curation_rules.py
=================
The rule-based cleaning steps that were applied, in this order, to the keyword-assigned
categories. Every rule returns a reason, so every removal can be written to an audit log.

  Stage 2  filter_stage2()   drops gene x environment interaction terms, treatment-response and
                             drug-induced traits, adverse drug reactions (outside Pharmacogenomics)
                             and a few misplaced survival / protein-level traits.
                             Pleiotropy / MTAG / "A or B" traits are deliberately KEPT.
  Stage 3  STAGE4_RULES,     collision review: removes keyword matches that don't hold up on
           stage5()          inspection, e.g. 'hip fracture' matched 'hip' in Body Composition,
                             craniofacial 'nose height' matched 'height' in Height and Growth,
                             a type-1-only diabetes trait in the T2D bucket. 'flag' = logged, not removed.

Categories outside the 65 defined in category_cues.yaml (proteomics, metabolomics, gene
expression, genomic features, brain imaging) need no rule here: they are simply not in the
cue table, so no trait can be assigned to them.

Category names refer to the final vocabulary (gwas_categories_v2.yaml).
ASCII-only on purpose (cluster compile restriction).
"""
import re

# =============================================================================
# STAGE 2 - "moderate, keep pleiotropy" filter (filter_categories.py)
# =============================================================================
CKD_HYPERTENSION_PATTERN = re.compile(r'level in Chronic kidney disease with hypertension', re.I)
INTERACTION_PATTERN = re.compile(r'\s+x\s+.*interaction|\binteraction\b.*x\s+', re.I)
CANCER_SURVIVAL_PATTERN = re.compile(
    r'(?:cancer|carcinoma|lymphoma|melanoma|myeloma|glioma|osteosarcoma|leukemia).*survival|'
    r'survival.*(?:cancer|carcinoma|lymphoma|melanoma|myeloma|glioma|osteosarcoma|leukemia)', re.I)
TREATMENT_RESPONSE_PATTERN = re.compile(
    r'response\s+to\s+|efficacy\s+|(?:treated\s+with|treatment\s+with)\s+\w+|-induced\s+|drug[- ]induced', re.I)
ADVERSE_DRUG_PATTERN = re.compile(r'adverse\s+(?:response|effect|reaction|event|outcome)|toxicity|side\s+effect', re.I)
TANNER_PATTERN = re.compile(r'^Tanner\s+\d', re.I)
TRANSPLANT_SURVIVAL_PATTERN = re.compile(r'graft\s+survival|transplant.*survival|survival.*transplant', re.I)
PROTEIN_LEVEL_PATTERN = re.compile(r'\s+protein\s+level(?:s)?(?:\s|$|\()|\s+levels?\s*\([A-Z0-9.:_-]+\)', re.I)
DIS_SURV_KW = ['survival in sepsis','survival in sporadic','survival in progressive','survival in pancreatic',
 'survival in head and neck','survival in melanoma','survival in microsatellite','survival in coronary',
 'transplant-free survival','sepsis from pneumonia','disease free survival','disease-free survival',
 'event free survival','overall survival in','progression free survival','progression-free survival',
 'recurrence-free survival','relapse-free survival']
def in_aging(t):
    if CANCER_SURVIVAL_PATTERN.search(t) or TANNER_PATTERN.search(t) or TRANSPLANT_SURVIVAL_PATTERN.search(t): return True
    tl = t.lower()
    if any(k in tl for k in DIS_SURV_KW): return True
    if ('protein level' in tl or 'levels' in tl) and ('motor neuron' in tl or tl.endswith('levels')): return True
    if "Alzheimer's disease (survival time)" in t: return True
    return False
def in_bp(t):
    return bool(CKD_HYPERTENSION_PATTERN.search(t)) or "Alzheimer's disease in hypertension" in t
def in_cancer(t):
    if PROTEIN_LEVEL_PATTERN.search(t):
        return not any(k in t.lower() for k in ['antigen','tumor','cancer','carcinoma','oncogene'])
    return False
def filter_stage2(t, cat):
    """True if the stage-2 'moderate, keep pleiotropy' filter would drop this trait from this category."""
    if cat.startswith('Aging') and in_aging(t): return 'aging misplacement'
    if 'Blood Pressure' in cat and in_bp(t): return 'CKD protein level in BP'
    if INTERACTION_PATTERN.search(t): return 'interaction term'
    if TREATMENT_RESPONSE_PATTERN.search(t):
        if 'levels' in t.lower() and 'response' not in t.lower(): pass
        else: return 'treatment response'
    if not cat.startswith('Pharmacogenomics') and ADVERSE_DRUG_PATTERN.search(t): return 'adverse drug reaction'
    if cat.startswith('Cancer') and in_cancer(t): return 'protein level in cancer category'
    return None


# =============================================================================
# STAGE 4 - keyword-collision rules (verbatim from clean_categories.py)
# Each rule takes a trait name and returns (should_remove, reason).
# =============================================================================

def rule_ckd_no_diabetes_protein(trait):
    """Protein levels in CKD patients WITHOUT diabetes.  'no diabetes' triggered match."""
    if 'Chronic kidney disease with hypertension and no diabetes' in trait:
        return True, "Protein level in CKD without diabetes (matched 'diabetes' from 'no diabetes')"
    return False, ""


def rule_hip_disease_in_body_comp(trait):
    """Hip DISEASE traits that don't belong in Body Composition."""
    t = trait.lower()
    keep = ['hip circumference', 'hip width', 'hip index', 'waist-hip',
            'waist-to-hip', 'waist hip', 'thoracic-to-hip',
            'hip bone mineral density and total body fat',
            'gluteofemoral', 'trunk fat', 'body fat', 'adipose',
            'hip shape mode', 'hip shape (dxa',
            'body mass index or hip osteoarthritis',
            'body mass index or knee osteoarthritis',
            'body mass index or osteoarthritis',
            'body mass index or spine osteoarthritis']
    for p in keep:
        if p in t:
            return False, ""

    remove = [
        (r'\bhip fracture\b', "Hip fracture -> Bone/Musculoskeletal"),
        (r'\bhip pain\b', "Hip pain -> Pain"),
        (r'osteoarthritis.*\bhip\b|\bhip\b.*osteoarthritis', "Hip OA -> Musculoskeletal"),
        (r'arthroplasty.*\bhip\b|\bhip\b.*arthroplasty', "Hip arthroplasty -> Musculoskeletal"),
        (r'osteolysis.*total hip', "Hip osteolysis -> Musculoskeletal"),
        (r'developmental dysplasia.*hip', "Dysplasia of hip -> Musculoskeletal"),
        (r'hip minimal joint space', "Hip joint space -> Musculoskeletal"),
        (r'coxarthrosis|ICD10 M16', "Coxarthrosis -> Musculoskeletal"),
        (r'revision surgery.*hip', "Hip revision surgery -> Musculoskeletal"),
        (r'non-surgical hip osteo', "Hip OA -> Musculoskeletal"),
        (r'surgical hip osteo', "Hip OA -> Musculoskeletal"),
    ]
    for pattern, reason in remove:
        if re.search(pattern, t, re.IGNORECASE):
            return True, reason
    return False, ""


def rule_abdominal_disease_in_body_comp(trait):
    """Abdominal DISEASE traits in Body Composition."""
    t = trait.lower()
    keep = ['abdominal adipose', 'abdominal fat', 'abdominal subcutaneous',
            'abdominal size', 'visceral']
    for p in keep:
        if p in t:
            return False, ""
    remove = [
        (r'abdominal aortic aneurysm', "Aortic aneurysm -> Cardiovascular"),
        (r'abdominal aortic calcification', "Aortic calcification -> Cardiovascular"),
        (r'abdominal aortic diameter', "Aortic diameter -> Cardiovascular"),
        (r'abdominal hernia', "Abdominal hernia -> GI"),
        (r'abdominal infection', "Abdominal infection -> Infectious Disease"),
        (r'abdominal pain', "Abdominal pain -> Pain/GI"),
        (r"crohn.*abdominal surgery", "Crohn's surgery -> GI-IBD"),
        (r'intracranial.*abdominal.*aneurysm', "Aneurysm pleiotropy -> Cardiovascular"),
    ]
    for pattern, reason in remove:
        if re.search(pattern, t, re.IGNORECASE):
            return True, reason
    return False, ""


def rule_bone_in_body_comp(trait):
    """Bone measurements at hip that belong in Bone category."""
    t = trait.lower()
    if any(x in t for x in ['fat mass', 'fat adjusted', 'lean mass', 'lean adjusted']):
        return False, ""
    remove = [
        (r'^hip bone mineral content', "Hip BMC -> Bone"),
        (r'^peak hip bone mineral', "Peak hip BMC -> Bone"),
        (r'^hip bone size', "Hip bone size -> Bone"),
        (r'^hip geometry', "Hip bone geometry -> Bone"),
        (r'^pediatric bone mineral.*\(hip\)', "Pediatric hip BMD -> Bone"),
        (r'pediatric areal bone mineral.*hip', "Pediatric hip BMD -> Bone"),
    ]
    for pattern, reason in remove:
        if re.search(pattern, t, re.IGNORECASE):
            return True, reason
    return False, ""


def rule_cancer_in_body_comp(trait):
    """Cancer ICD codes matched via 'hip' in body composition."""
    if 'Carcinoma in situ' in trait and 'hip' in trait.lower():
        return True, "Carcinoma in situ -> Cancer (matched 'hip')"
    return False, ""


def rule_cardiac_indexed_bsa_in_body_comp(trait):
    """Cardiac measurements indexed to body surface area in Body Composition."""
    t = trait.lower()
    if 'body surface area' in t and ('ventricular' in t or 'left ventricular' in t):
        return True, "Cardiac measurement indexed to BSA -> Cardiac Structure (matched 'body surface area')"
    return False, ""


def rule_lv_structure_in_bp(trait):
    """LV structural measurements in Blood Pressure matched via 'diastolic'/'systolic'."""
    t = trait.lower()
    keep = ['blood pressure', 'hypertension', 'pulse pressure',
            'arterial pressure', 'diastolic blood', 'systolic blood',
            'mean arterial', 'white coat', 'night-to-day ratio',
            'preeclampsia', 'migraine', 'cognitive function and',
            'medication', 'takes medication', 'treatment effectiveness',
            'combination therapy', 'uncontrolled blood', 'variability',
            'change trajectory']
    for p in keep:
        if p in t:
            return False, ""

    remove = [
        (r'end.diastolic bi.ventricular shape', "Bi-ventricular shape -> Cardiac Structure (matched 'diastolic')"),
        (r'left ventricular concentricity', "LV concentricity -> Cardiac Structure"),
        (r'left ventricular end.diastolic mass', "LV mass -> Cardiac Structure (matched 'diastolic')"),
        (r'left ventricular end.diastolic volume', "LV volume -> Cardiac Structure (matched 'diastolic')"),
        (r'left ventricular end.systolic volume', "LV volume -> Cardiac Structure (matched 'systolic')"),
        (r'left ventricular mass.*indexed', "LV mass -> Cardiac Structure"),
        (r'left ventricular mass to end', "LV mass ratio -> Cardiac Structure"),
        (r'left ventricular mass \(', "LV mass -> Cardiac Structure"),
        (r'left ventricular stroke volume', "LV stroke volume -> Cardiac Structure"),
        (r'longitudinal peak diastolic strain', "LV strain -> Cardiac Structure (matched 'diastolic')"),
        (r'radial peak diastolic strain', "LV strain -> Cardiac Structure (matched 'diastolic')"),
        (r'right ventricular end.diastolic', "RV volume -> Cardiac Structure (matched 'diastolic')"),
        (r'right ventricular end systolic', "RV volume -> Cardiac Structure (matched 'systolic')"),
        (r'right ventricular peak', "RV function -> Cardiac Structure"),
        (r'right ventricular stroke volume', "RV stroke volume -> Cardiac Structure"),
        (r'left  ventricle systolic dysfunction', "LV dysfunction -> Heart Failure (matched 'systolic')"),
        (r'left ventricle diastolic internal', "LV dimension -> Cardiac Structure (matched 'diastolic')"),
        (r'left ventricular end systole\b', "LV wall thickness -> Cardiac Structure"),
        (r'left ventricular end diastole\b', "LV wall thickness -> Cardiac Structure"),
        (r'heart failure with preserved ef', "Diastolic heart failure -> Heart Failure (matched 'diastolic')"),
        (r'heart failure with reduced ef', "Systolic heart failure -> Heart Failure (matched 'systolic')"),
    ]
    for pattern, reason in remove:
        if re.search(pattern, t, re.IGNORECASE):
            return True, reason
    return False, ""


def rule_low_mw_protein_in_weight(trait):
    """Protein with 'weight' in name matched to Weight/BMI."""
    if 'Low molecular weight phosphotyrosine' in trait:
        return True, "Protein level (matched 'weight' from 'Low molecular weight')"
    return False, ""


def rule_brain_imaging_in_reproductive(trait):
    """Brain imaging 'lh' (left hemisphere) matched to Reproductive via 'LH'."""
    brain_patterns = [
        r'^T1 FreeSurfer',
        r'^BA-exvivo lh',
        r'^HippSubfield lh',
        r'^AmygNuclei lh',
        r'^ThalamNuclei lh',
        r'^aparc-',
        r'^aseg lh',
        r'^wg lh',
    ]
    for pattern in brain_patterns:
        if re.search(pattern, trait):
            return True, "Brain imaging (left hemisphere 'lh' matched luteinizing hormone 'LH')"
    return False, ""


def rule_immune_memory_in_cognitive(trait):
    """Immunological 'memory' cells matched to Cognitive via 'memory'."""
    t = trait.lower()
    cognitive_terms = [
        'memory loss', 'memory performance', 'memory decline',
        'memory dysfunction', 'verbal memory', 'visual memory',
        'episodic memory', 'spatial memory', 'face memory',
        'working memory', 'short-term memory', 'long-term memory',
        'logical memory', 'delayed recall', 'immediate recall',
        'word recall', 'memory impairment', 'baseline memory',
        'memory (pairs-matching)', 'takes medication for memory',
        'depression, cognition and memory', 'memory (delayed',
        'memory (immediate', 'static spatial working memory',
    ]
    for term in cognitive_terms:
        if term in t:
            return False, ""

    immune_patterns = [
        (r'\bmemory b cell\b', "Immunological memory B cell (not cognitive memory)"),
        (r'\bswitched memory b cell\b', "Immunological switched memory B cell"),
        (r'\bunswitched memory b cell\b', "Immunological unswitched memory B cell"),
        (r'\bcentral memory cd', "Immunological central memory T cell"),
        (r'\beffector memory cd', "Immunological effector memory T cell"),
        (r'\bmemory t.cell telomere\b', "Immunological memory T cell telomere"),
        (r'baff-r on.*memory', "BAFF-R on immune memory cell"),
        (r'cd\d+ on.*memory b', "CD marker on immune memory B cell"),
        (r'cd\d+ on.*switched memory', "CD marker on immune switched memory cell"),
        (r'cd\d+ on.*unswitched memory', "CD marker on immune unswitched memory cell"),
        (r'cd\d+ on central memory', "CD marker on immune central memory T cell"),
        (r'cd\d+ on effector memory', "CD marker on immune effector memory T cell"),
        (r'igd.*memory', "IgD on immune memory B cell"),
        (r'hvem on.*central memory', "HVEM on immune central memory T cell"),
        (r'hvem on.*effector memory', "HVEM on immune effector memory T cell"),
    ]
    for pattern, reason in immune_patterns:
        if re.search(pattern, t, re.IGNORECASE):
            return True, reason
    return False, ""


def rule_iq_protein_in_cognitive(trait):
    """IQ domain proteins matched to Cognitive via 'IQ'."""
    if 'IQ domain-containing protein' in trait:
        return True, "IQ domain protein (structural motif, not intelligence quotient)"
    return False, ""


def rule_lymphocytic_thyroiditis_in_cancer(trait):
    """Chronic lymphocytic thyroiditis in Cancer-Hematological via 'lymphocytic'."""
    if 'Chronic lymphocytic thyroiditis' in trait:
        return True, "Autoimmune thyroiditis, not cancer (matched 'lymphocytic')"
    return False, ""


# ============================================================================
# RULES PER CATEGORY
# ============================================================================

CATEGORY_RULES = {
    'Anthropometry - Weight and BMI': [
        rule_low_mw_protein_in_weight,
    ],
    'Anthropometry - Body Composition and Fat Distribution': [
        rule_hip_disease_in_body_comp,
        rule_abdominal_disease_in_body_comp,
        rule_bone_in_body_comp,
        rule_cancer_in_body_comp,
        rule_cardiac_indexed_bsa_in_body_comp,
    ],
    'Cardiovascular - Blood Pressure and Hypertension': [
        rule_lv_structure_in_bp,
    ],
    'Diabetes - Type 2 Diabetes and Glucose Metabolism': [
        rule_ckd_no_diabetes_protein,
    ],
    'Diabetes - Insulin and Insulin Resistance': [
        rule_ckd_no_diabetes_protein,
    ],
    'Reproductive - Sex Hormones and Fertility': [
        rule_brain_imaging_in_reproductive,
    ],
    'Neuropsychiatric - Cognitive Function and Intelligence': [
        rule_immune_memory_in_cognitive,
        rule_iq_protein_in_cognitive,
    ],
    'Cancer - Hematological': [
        rule_lymphocytic_thyroiditis_in_cancer,
    ],
}





# =============================================================================
# STAGE 5 - keyword-collision rules (clean_gwas_categories.py, stage A)
# =============================================================================

def stage5_rules():
    """(category, pattern, guard, action, reason). Same rules as clean_gwas_categories.py,
    with the retired category 'Sensory - Hearing and Smell' retargeted to 'Sensory - Hearing'."""
    R = lambda p: re.compile(p, re.I)
    return [
        ("Anthropometry - Height and Growth",
         R(r"facial|\bnasal\b|\bnose\b|\blip\b|philtrum|palpebrale|vermillion|gnathion|nasion|forehead|\bscar\b"),
         R(r"\bbody height\b|standing height|sitting height"),
         "remove", "craniofacial 'height', not stature"),
        ("Anthropometry - Height and Growth",
         R(r"\b(adipose|visceral|liver|pericardial|gluteofemoral|abdominal) .*adjust.*height|adjust.*height\)"),
         R(r"^height\b"),
         "remove", "height is only a covariate; trait is organ/adipose volume"),
        ("Diabetes - Type 2 Diabetes and Glucose Metabolism",
         R(r"type 1 diabet"),
         R(r"type 2 diabet"),
         "remove", "type-1-diabetes-specific trait in the T2D bucket"),
        ("Aging - Longevity and Healthspan",
         R(r"cortex|ventricular|hippocamp|eyeage|\bfacial\b|skin aging"),
         None,
         "remove", "imaging/dermatology trait, not a longevity/lifespan phenotype"),
        ("Sensory - Hearing",
         R(r"\btaste\b"),
         None,
         "flag", "taste perception is a separate sensory modality from hearing"),
        ("Pain - Chronic Pain Conditions",
         R(r"fear of .*pain|medication for pain relief"),
         None,
         "flag", "behavioral (fear) or medication-use trait, not a pain phenotype"),
    ]


_STAGE5 = None


def stage5(trait, category):
    """Return (action, reason) if a stage-5 rule fires for this trait in this category."""
    global _STAGE5
    if _STAGE5 is None:
        _STAGE5 = stage5_rules()
    for cat, pat, guard, action, reason in _STAGE5:
        if cat != category or not pat.search(trait):
            continue
        if guard is not None and guard.search(trait):
            continue
        return action, reason
    return None
