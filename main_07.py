import torch
import pandas as pd
import numpy as np
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt
import random
import warnings
import spacy
import requests
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_NAME     = "llama3.1:8b"
API_URL        = "http://ollama.warhol.informatik.rwth-aachen.de/api/generate"

NOTES_PATH     = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz"
DIAGNOSES_PATH = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/diagnoses_icd.csv.gz"
DICT_PATH      = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/d_icd_diagnoses.csv.gz"

FINAL_RESULTS  = "final_robustness2.csv"
HTML_REPORT    = "robustness_report2.html"
PLOT_FILE      = "robustness_plots2.png"
MAX_SAMPLES    = 500
DEVICE         = "cpu"

# Semantic similarity threshold used only as a DIAGNOSTIC metric now.
# The LLM equivalence check is the real gate (see is_baseline_correct).
BASELINE_SIM_THRESHOLD = 0.78

warnings.filterwarnings("ignore")


def debug_print(stage, message):
    print(f"\033[94m[{stage}]\033[0m {message}")


# ============================================================
# SECTION 1: LLM CONNECTOR
# ============================================================
def query_rwth_server(prompt, max_tokens=50):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": max_tokens}
    }
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.post(API_URL, headers=headers,
                                data=json.dumps(payload), timeout=60)
        response.raise_for_status()
        return response.json().get('response', '').strip()
    except requests.exceptions.RequestException as e:
        debug_print("API_ERROR", f"Connection failed: {e}")
        return ""


# ============================================================
# SECTION 2: DATA LOADER
# ============================================================
class MIMICLoader:
    @staticmethod
    def clean_noise(text):
        text = str(text)
        text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)
        text = re.sub(r"_{2,}", " ", text)
        text = text.replace("=", " ").replace("*", " ").replace("#", " ")
        return " ".join(text.split())

    @staticmethod
    def extract_blind_clinical_info(text):
        relevant_parts = []
        hpi = re.search(
            r"History of Present Illness:(.*?)(Past Medical History:|Social History:)",
            text, re.S | re.I)
        if hpi:
            relevant_parts.append(f"HPI: {hpi.group(1).strip()}")
        course = re.search(
            r"Brief Hospital Course:(.*?)(Medications on Admission:|Discharge Medications:|Discharge Diagnosis:)",
            text, re.S | re.I)
        if course:
            relevant_parts.append(f"Hospital Course: {course.group(1).strip()}")
        if relevant_parts:
            return "\n\n".join(relevant_parts)
        return text[:1500]

    @classmethod
    def load(cls, notes_path, diag_path, dict_path, n_samples):
        debug_print("DATA", "Loading Data Tables...")
        notes = pd.read_csv(notes_path, compression='gzip',
                            usecols=['subject_id', 'hadm_id', 'text'])
        diags = pd.read_csv(diag_path, compression='gzip')
        if os.path.exists(dict_path):
            icd_dict = pd.read_csv(dict_path, compression='gzip')
            diags = pd.merge(diags, icd_dict[['icd_code', 'icd_version', 'long_title']],
                             on=['icd_code', 'icd_version'], how='left')
        else:
            diags['long_title'] = "Unknown Description"
        primary_diags = diags[diags['seq_num'] == 1]
        debug_print("DATA", "Merging Notes with Diagnoses...")
        df = pd.merge(notes,
                      primary_diags[['hadm_id', 'icd_code', 'icd_version', 'long_title']],
                      on='hadm_id')
        debug_print("DATA", "Cleaning and Extracting Blind Clinical Info...")
        df['text'] = df['text'].apply(cls.clean_noise)
        df['processed_text'] = df['text'].apply(cls.extract_blind_clinical_info)
        return df.sample(n=min(len(df), n_samples), random_state=42)


# ============================================================
# SECTION 3: PERTURBATION — RELIABLE SYSTEMATIC APPROACH
#
# The core problem with the previous approach: we blocked specific words
# after seeing them fail. This is whack-a-mole — every run surfaces new
# bad synonyms. The real fix is a SYSTEMATIC VALIDATION LAYER that catches
# bad replacements regardless of which word they come from.
#
# The three-layer validation system:
#   Layer 1 — Entity category filter (skip entities that are inherently
#              unsafe to replace: polarities, laterality, measurements,
#              section headers, negations)
#   Layer 2 — Structural validation of the replacement itself (length,
#              word count, grammar type)
#   Layer 3 — LLM self-check: ask the LLM "is this replacement clinically
#              safe?" before accepting it. This catches semantic errors the
#              rule-based layers miss (like benign→hypertension) without
#              needing a hardcoded word list.
# ============================================================

# ── LAYER 1: Entity category patterns that are ALWAYS unsafe to replace ──────
# These are defined by PATTERN not by specific word — so no word list needed.

# Regex patterns for entity text that should never be replaced
UNSAFE_ENTITY_PATTERNS = [
    # Negation words and phrases
    r'^(no|not|denies|denied|without|negative|none|absent|absence)\b',
    # Lab result polarity (positive/negative as standalone)
    r'^(positive|negative)$',
    # Directional/laterality — left/right/bilateral etc.
    r'^(left|right|bilateral|unilateral|ipsilateral|contralateral|'
    r'anterior|posterior|proximal|distal|superior|inferior|medial|lateral)$',
    # Quantitative polarity — any word describing increase/decrease direction
    r'^(elevated|increased|decreased|reduced|low|high|normal|abnormal|'
    r'mild|moderate|severe|minimal|significant|benign|malignant|acute|chronic|'
    r'stable|unstable|resolved|worsening|improving)$',
    # Time units
    r'^(minute|minutes|second|seconds|hour|hours|day|days|week|weeks|'
    r'month|months|year|years)s?$',
    # Vital sign terms — replacing these changes clinical meaning
    r'^(afebrile|normotensive|tachycardic|bradycardic|hypertensive|hypotensive|'
    r'tachypneic|bradypneic|normocardic)$',
    # Negation-prefix words
    r'^(non|no)[a-z]+$',
    # All-caps section headers (HISTORY OF PRESENT ILLNESS etc.)
    r'^[A-Z][A-Z\s]{10,}$',
    # Pure numbers or measurements
    r'^\d',
    # Single letters (abbreviations like L, R, V, Q)
    r'^[A-Z]$',
]

_UNSAFE_COMPILED = [re.compile(p, re.IGNORECASE) for p in UNSAFE_ENTITY_PATTERNS]


def entity_is_safe_to_replace(entity_text: str) -> tuple:
    """
    Layer 1: Pattern-based safety check on the entity itself.
    Returns (is_safe, reason).
    No hardcoded word lists — uses regex patterns that generalise.
    """
    text = entity_text.strip()

    for pattern in _UNSAFE_COMPILED:
        if pattern.match(text):
            return False, f"matches unsafe pattern: {pattern.pattern[:40]}"

    # Multi-word check: if it's a pure negation phrase
    lower = text.lower()
    NEGATION_STARTS = ('no ', 'not ', 'denies ', 'denied ', 'without ',
                       'negative for ', 'absence of ', 'no evidence of ')
    if lower.startswith(NEGATION_STARTS):
        return False, "negation phrase"

    # All-caps multi-word = section header
    tokens = text.split()
    if len(tokens) >= 2 and all(t.isupper() and t.isalpha() for t in tokens):
        return False, "section header"

    return True, "ok"


def replacement_is_structurally_valid(original: str, replacement: str) -> tuple:
    """
    Layer 2: Structural validation of the replacement string.
    Catches expansions, grammar changes, and format violations.
    Returns (is_valid, reason).
    No hardcoded word lists — pure structural rules.
    """
    orig_words = original.strip().split()
    repl_words = replacement.strip().split()
    n_orig = len(orig_words)
    n_repl = len(repl_words)

    # Rule 1: Replacement cannot be more than 2x the word count of original
    if n_repl > max(n_orig * 2, n_orig + 3):
        return False, f"too long ({n_orig} words → {n_repl} words)"

    # Rule 2: Single-word original → replacement must be ≤ 3 words
    # This blocks "palpation → Per rectal examination" type errors
    if n_orig == 1 and n_repl > 3:
        return False, f"single-word original expanded to {n_repl} words"

    # Rule 3: Replacement must not be empty or same as original
    if not replacement.strip():
        return False, "empty replacement"
    if replacement.strip().lower() == original.strip().lower():
        return False, "identical to original"

    # Rule 4: Replacement must not be drastically shorter (avoids deletions)
    if n_orig > 2 and n_repl < n_orig // 2:
        return False, f"too short ({n_orig} words → {n_repl} words)"

    # Rule 5: Character length ratio check
    char_ratio = len(replacement) / max(len(original), 1)
    if char_ratio > 5.0:
        return False, f"character length ratio too high ({char_ratio:.1f}x)"

    # Rule 6: Replacement should not introduce sentence-ending punctuation
    # mid-replacement (suggests the LLM output a sentence, not a term)
    if re.search(r'[.!?](?!\s*$)', replacement):
        return False, "contains sentence-ending punctuation mid-text"

    return True, "ok"


def validate_replacement_with_llm(original: str, replacement: str,
                                   context: str) -> tuple:
    """
    Layer 3: LLM self-check for clinical safety.
    Ask the LLM: "Is this replacement clinically equivalent and safe?"
    This catches semantic errors that pattern rules cannot — like
    'benign → hypertension' which passes all structural checks but is
    clinically wrong.

    Returns (is_safe, reason).
    This costs one extra API call per candidate replacement, but it
    completely removes the need for any hardcoded word blocklist.
    """
    prompt = (
        f"Context sentence: \"{context}\"\n"
        f"Original term: '{original}'\n"
        f"Proposed replacement: '{replacement}'\n\n"
        f"Answer YES only if ALL of the following are true:\n"
        f"1. The replacement is medically/clinically equivalent to the original "
        f"in this specific context.\n"
        f"2. The replacement does NOT change the clinical meaning, polarity "
        f"(e.g. positive vs negative), severity, laterality, or diagnosis.\n"
        f"3. The replacement is grammatically the same type as the original "
        f"(noun→noun, adjective→adjective, etc.).\n"
        f"4. The replacement fits naturally in the sentence without making it "
        f"grammatically incorrect.\n\n"
        f"Answer with only YES or NO."
    )
    response = query_rwth_server(prompt, max_tokens=5)
    is_safe = response.strip().upper().startswith("YES")
    return is_safe, ("LLM approved" if is_safe else "LLM rejected")


class GenerativePerturber:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_sci_md")
        except OSError:
            self.nlp = spacy.load("en_core_sci_sm")

    def is_negated(self, token):
        negation_terms = {'no', 'not', 'denies', 'negative',
                          'without', 'none', 'denied'}
        window = [t.text.lower()
                  for t in list(token.doc)[max(0, token.i - 3):token.i]]
        return any(term in window for term in negation_terms)

    def get_synonym_from_llm(self, word: str, context_sentence: str) -> str | None:
        """Generate a synonym candidate. Prompt is tightly constrained to
        minimise hallucination and expansion. The LLM is only asked to produce
        the replacement term — validation is handled separately."""
        is_acronym = word.isupper() and len(word) <= 5
        prompt = (
            f"Context: \"{context_sentence}\"\n"
            f"Term: '{word}'\n\n"
            f"Task: Provide ONE medically equivalent synonym for '{word}' "
            f"as used in this specific clinical context.\n\n"
            f"Rules:\n"
            f"- Respond with the replacement term ONLY. No explanation.\n"
            f"- Do NOT change clinical meaning, polarity, or severity.\n"
            f"- Do NOT expand abbreviations into full sentences.\n"
            f"- If '{word}' is a single word, your answer must be 1-3 words max.\n"
            f"- If no safe synonym exists, respond with the word: NONE\n\n"
            f"Replacement:"
        )
        raw = query_rwth_server(prompt, max_tokens=20)
        if not raw:
            return None
        # Clean up
        synonym = raw.replace('"', '').replace("'", '').strip()
        # Strip trailing punctuation
        synonym = re.sub(r'[.!?,;]+$', '', synonym).strip()
        # If the LLM said NONE, skip
        if synonym.upper() == "NONE" or not synonym:
            return None
        # If it's the same as the original, skip
        if synonym.lower() == word.lower():
            return None
        return synonym

    def perturb(self, text: str, perturbation_rate: float = 0.10) -> tuple:
        doc = self.nlp(text)

        # Pre-filter: only entities that are not negated
        candidate_entities = [ent for ent in doc.ents
                               if not self.is_negated(ent[0])]
        if not candidate_entities:
            return text, []

        num_to_perturb = max(1, int(len(candidate_entities) * perturbation_rate))
        target_entities = random.sample(
            candidate_entities, min(num_to_perturb, len(candidate_entities)))
        # Process right-to-left so character offsets stay valid
        target_entities.sort(key=lambda x: x.start_char, reverse=True)

        perturbed_text = text
        changes_log = []

        for ent in target_entities:
            original = ent.text

            # ── LAYER 1: Entity category safety check ────────────────────
            safe, reason = entity_is_safe_to_replace(original)
            if not safe:
                print(f"     LAYER1 BLOCKED ({reason}): '{original}'")
                continue

            # Get context window for synonym generation
            context = (ent.sent.text if ent.sent
                       else text[max(0, ent.start_char - 60):ent.end_char + 60])

            # Generate synonym candidate
            replacement = self.get_synonym_from_llm(original, context)
            if not replacement:
                print(f"     NO SYNONYM: '{original}'")
                continue

            # ── LAYER 2: Structural validation ───────────────────────────
            struct_ok, struct_reason = replacement_is_structurally_valid(
                original, replacement)
            if not struct_ok:
                print(f"     LAYER2 BLOCKED ({struct_reason}): "
                      f"'{original}' → '{replacement}'")
                continue

            # ── LAYER 3: LLM clinical safety self-check ──────────────────
            llm_ok, llm_reason = validate_replacement_with_llm(
                original, replacement, context)
            if not llm_ok:
                print(f"     LAYER3 BLOCKED ({llm_reason}): "
                      f"'{original}' → '{replacement}'")
                continue

            # All three layers passed — apply replacement
            perturbed_text = (perturbed_text[:ent.start_char]
                              + replacement
                              + perturbed_text[ent.end_char:])
            changes_log.append((original, replacement))
            print(f"     ACCEPTED: '{original}' → '{replacement}'")

        return perturbed_text, changes_log


# ============================================================
# SECTION 4: SEMANTIC EVALUATOR (ClinicalBERT)
# ============================================================
class SemanticEvaluator:
    def __init__(self):
        debug_print("EVAL_INIT", "Loading ClinicalBERT (CPU)...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT")
        self.model = AutoModel.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT").to(DEVICE)
        self.model.eval()

    def get_sim(self, text1: str, text2: str) -> float:
        with torch.no_grad():
            inputs = self.tokenizer(
                [text1, text2], padding=True, truncation=True,
                max_length=512, return_tensors="pt").to(DEVICE)
            outputs = self.model(**inputs)
            e1 = outputs.last_hidden_state[0, 0, :].unsqueeze(0)
            e2 = outputs.last_hidden_state[1, 0, :].unsqueeze(0)
            sim = cosine_similarity(e1.numpy(), e2.numpy())[0][0]
        return float(sim)

    def descriptions_match(self, desc1: str, desc2: str,
                            threshold: float = BASELINE_SIM_THRESHOLD) -> tuple:
        sim = self.get_sim(desc1, desc2)
        return sim >= threshold, sim


# ============================================================
# SECTION 5: BASELINE CORRECTNESS — LLM IS THE PRIMARY GATE
# ============================================================

def check_clinical_equivalence(pred_icd: str, gt_icd: str,
                                gt_desc: str) -> bool:
    """Ask the LLM whether the predicted ICD is the same condition as GT.
    This handles ICD-9 vs ICD-10 mismatches (MIMIC has both), synonym
    descriptions, and specificity differences — none of which string
    matching can handle reliably."""
    if pred_icd in ("UNKNOWN", ""):
        return False
    prompt = (
        f"Ground truth diagnosis: '{gt_desc}' (ICD code: {gt_icd})\n"
        f"Predicted ICD code: {pred_icd}\n\n"
        f"Question: Does the predicted ICD code represent the same primary "
        f"medical condition as the ground truth diagnosis?\n"
        f"Ignore differences in ICD version (ICD-9 vs ICD-10) and minor "
        f"specificity differences. Focus only on whether the core disease "
        f"or condition is the same.\n"
        f"Answer with only YES or NO."
    )
    response = query_rwth_server(prompt, max_tokens=5)
    return response.strip().upper().startswith("YES")


def is_baseline_correct(scorer: SemanticEvaluator,
                         orig_desc: str, gt_desc: str,
                         orig_icd: str, gt_icd: str) -> tuple:
    """
    Two-gate correctness check. PRIMARY gate is the LLM clinical equivalence
    check. Semantic similarity is logged as supporting evidence only.

    A sample passes ONLY if llm_equiv=YES.
    Semantic similarity alone is NOT sufficient — it caused false passes
    in previous runs (AMI passing as CKD because embedding similarity
    happened to exceed the threshold).

    Both results are returned for transparency in the report.
    """
    sim_ok, baseline_sim = scorer.descriptions_match(orig_desc, gt_desc)
    llm_ok = check_clinical_equivalence(orig_icd, gt_icd, gt_desc)

    gate_parts = [
        f"sem_sim={baseline_sim:.4f}{'✓' if sim_ok else '✗'}",
        f"llm_equiv={'YES✓' if llm_ok else 'NO✗'}"
    ]

    # LLM equivalence is the sole deciding gate
    passed = llm_ok
    return passed, baseline_sim, llm_ok, " | ".join(gate_parts)


# ============================================================
# SECTION 6: ICD PREDICTION & DESCRIPTION
# ============================================================
def get_icd_prediction(text: str) -> str:
    """Two-attempt ICD extraction with fallback regex.
    Returns 'UNKNOWN' instead of blank so downstream comparisons
    don't silently fail on empty strings."""
    primary_prompt = (
        f"Read the clinical note below. Return ONLY the single most likely "
        f"ICD-10 diagnosis code (example format: J18.9 or I21.0). "
        f"Do not write the disease name. Do not explain. "
        f"Output the code only.\n\n"
        f"Note:\n{text}\n\nICD-10 Code:"
    )
    fallback_prompt = (
        f"Output a single ICD-10 code for this note. "
        f"Format: letter + 2 digits + optional dot + digits. "
        f"Examples: J18.9, I21.0, K56.9\n\n"
        f"Note:\n{text[:800]}\n\nCode:"
    )
    for attempt, prompt in enumerate([primary_prompt, fallback_prompt]):
        response = query_rwth_server(prompt)
        if response:
            match = re.search(r'[A-Z]\d{2}\.\d+', response.strip().upper())
            if match:
                return match.group(0)
            match = re.search(r'[A-Z]\d{2,4}', response.strip().upper())
            if match:
                return match.group(0)
    return "UNKNOWN"


def get_diagnosis_description(icd_code: str, icd_version: int) -> str:
    if icd_code in ("UNKNOWN", ""):
        return "Unknown"
    prompt = (
        f"What medical condition does ICD-{icd_version} code "
        f"'{icd_code}' represent? "
        f"Reply with ONLY the condition name in 2-5 words. No explanation."
    )
    return query_rwth_server(prompt)


# ============================================================
# SECTION 7: DRIFT CLASSIFICATION
# ============================================================
def classify_drift(orig_icd: str, pert_icd: str) -> dict:
    """Two-level drift: categorical (different disease family) vs
    subcategory (same family, different specificity code).
    This separates real errors from minor specificity noise."""
    orig_norm = re.sub(r'[^A-Z0-9]', '', orig_icd.upper())
    pert_norm = re.sub(r'[^A-Z0-9]', '', pert_icd.upper())

    # Handle UNKNOWN predictions
    if "UNKNOWN" in (orig_norm, pert_norm):
        return {
            "icd_codes_differ" : True,
            "drift_level"      : "unknown",
            "categorical_drift": False,
            "subcategory_drift": False,
        }

    exact_match  = orig_norm == pert_norm
    prefix_match = (len(orig_norm) >= 3 and len(pert_norm) >= 3 and
                    orig_norm[:3] == pert_norm[:3])

    if exact_match:
        level = "none"
    elif prefix_match:
        level = "subcategory"
    else:
        level = "categorical"

    return {
        "icd_codes_differ" : not exact_match,
        "drift_level"      : level,
        "categorical_drift": level == "categorical",
        "subcategory_drift": level == "subcategory",
    }


# ============================================================
# SECTION 8: PLOTS
# ============================================================
def plot_results(df: pd.DataFrame):
    debug_print("PLOT", "Generating plots...")
    valid_df = df[df['baseline_correct'] == True]
    if len(valid_df) == 0:
        print("No valid samples to plot.")
        return

    plt.figure(figsize=(22, 5))

    plt.subplot(1, 4, 1)
    sns.histplot(valid_df['semantic_similarity'], kde=True,
                 bins=15, color='teal')
    plt.title("Semantic Preservation\n(correct baselines only)")
    plt.xlabel("Cosine Similarity")

    plt.subplot(1, 4, 2)
    drift_counts = valid_df['drift_level'].value_counts()
    colors_map = {'none': '#90EE90', 'subcategory': '#FFD700',
                  'categorical': '#FF6B6B', 'unknown': '#CCCCCC'}
    bar_colors = [colors_map.get(k, 'grey') for k in drift_counts.index]
    drift_counts.plot(kind='bar', color=bar_colors)
    cat_rate = valid_df['categorical_drift'].mean() * 100
    plt.title(f"Drift Level Breakdown\nCategorical drift: {cat_rate:.1f}%")
    plt.xlabel("Drift Level")
    plt.xticks(rotation=0)

    plt.subplot(1, 4, 3)
    plot_df = valid_df[valid_df['drift_level'].isin(
        ['none', 'subcategory', 'categorical'])]
    if len(plot_df) > 0:
        sns.scatterplot(data=plot_df, x='changes_count',
                        y='semantic_similarity', hue='drift_level',
                        palette={'none': 'green', 'subcategory': 'orange',
                                 'categorical': 'red'})
    plt.title("Changes vs Similarity\ncoloured by drift level")
    plt.xlabel("Words Changed")

    plt.subplot(1, 4, 4)
    counts = df['baseline_correct'].value_counts()
    labels, colors_pie = [], []
    if True in counts.index:
        labels.append(f'Correct Baseline (n={counts[True]})')
        colors_pie.append('#90EE90')
    if False in counts.index:
        labels.append(f'Wrong Baseline (n={counts[False]})')
        colors_pie.append('#FFB6C1')
    plt.pie(counts, labels=labels, colors=colors_pie, autopct='%1.1f%%')
    plt.title("Baseline Quality\n(gate: LLM clinical equivalence)")

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150)
    print(f"Plot saved: {PLOT_FILE}")


# ============================================================
# SECTION 9: HTML REPORT
# ============================================================
def generate_html_report(df: pd.DataFrame, filename: str = HTML_REPORT):

    DRIFT_COLORS = {
        'none'       : '#ccffcc',
        'subcategory': '#fff3cd',
        'categorical': '#ffcccc',
        'unknown'    : '#eeeeee',
        'skipped'    : '#f9f9f9',
    }
    DRIFT_LABELS = {
        'none'       : 'STABLE',
        'subcategory': 'SUBCATEGORY DRIFT',
        'categorical': 'CATEGORICAL DRIFT',
        'unknown'    : 'UNKNOWN ICD',
        'skipped'    : 'SKIPPED',
    }

    def highlight_changes(original_text, perturbed_text, changes_str):
        if not changes_str or changes_str in ('None', '', 'SKIPPED'):
            return original_text, perturbed_text
        orig_h = original_text
        pert_h = perturbed_text
        for pair in changes_str.split("|"):
            pair = pair.strip()
            if " -> " not in pair:
                continue
            orig_word, new_word = pair.split(" -> ", 1)
            orig_word = orig_word.strip()
            new_word  = new_word.strip()
            orig_h = re.sub(
                rf'\b{re.escape(orig_word)}\b',
                f'<span style="background:#FF6B6B;color:white;'
                f'padding:1px 4px;border-radius:3px;'
                f'font-weight:bold;">{orig_word}</span>',
                orig_h, flags=re.IGNORECASE)
            pert_h = re.sub(
                rf'\b{re.escape(new_word)}\b',
                f'<span style="background:#51C878;color:white;'
                f'padding:1px 4px;border-radius:3px;'
                f'font-weight:bold;">{new_word}</span>',
                pert_h, flags=re.IGNORECASE)
        return orig_h, pert_h

    rows_html = ""
    for _, row in df.iterrows():
        is_skipped  = str(row.get('perturbed_text', 'SKIPPED')) == 'SKIPPED'
        drift_level = row.get('drift_level', 'none')
        drift_color = DRIFT_COLORS.get(drift_level, '#ccffcc')
        drift_label = DRIFT_LABELS.get(drift_level, 'STABLE')
        sim_display = (f"{float(row['semantic_similarity']):.4f}"
                       if row.get('semantic_similarity') else 'N/A')
        baseline_disp  = f"{float(row.get('baseline_sim', 0)):.4f}"
        pred_desc      = row.get('original_pred_desc', '')
        changes_display = (row['changes_made']
                           if row.get('changes_made') not in
                           ['SKIPPED', '', None] else 'None')
        gate_display = row.get('baseline_gate', '')

        if is_skipped:
            text_section = (
                f'<p style="color:#999;font-style:italic;padding:10px;'
                f'background:#f9f9f9;border-radius:4px;">'
                f'Skipped — LLM determined prediction does not match '
                f'ground truth condition.<br>'
                f'<small>Gates: {gate_display}</small></p>')
        else:
            oh, ph = highlight_changes(
                row.get('original_text', ''),
                row.get('perturbed_text', ''),
                row.get('changes_made', ''))
            text_section = f"""
            <div style="margin-top:10px;">
                <div style="margin-bottom:8px;font-size:0.85em;">
                    <span style="background:#FF6B6B;color:white;
                    padding:2px 8px;border-radius:3px;">Original words</span>&nbsp;
                    <span style="background:#51C878;color:white;
                    padding:2px 8px;border-radius:3px;">Validated replacements</span>
                </div>
                <table style="width:100%;border-collapse:collapse;">
                <tr>
                    <th style="width:50%;background:#e8f4f8;padding:8px;
                    border:1px solid #ccc;">Original Text</th>
                    <th style="width:50%;background:#fff8e8;padding:8px;
                    border:1px solid #ccc;">Perturbed Text</th>
                </tr>
                <tr>
                    <td style="padding:10px;border:1px solid #ccc;
                    vertical-align:top;white-space:pre-wrap;
                    font-size:0.85em;line-height:1.6em;">{oh}</td>
                    <td style="padding:10px;border:1px solid #ccc;
                    vertical-align:top;white-space:pre-wrap;
                    font-size:0.85em;line-height:1.6em;">{ph}</td>
                </tr>
                <tr>
                    <td style="padding:8px;border:1px solid #ccc;
                    background:#f0f0f0;text-align:center;">
                    <b>ICD: {row['original_icd_pred']}</b></td>
                    <td style="padding:8px;border:1px solid #ccc;
                    background:#f0f0f0;text-align:center;">
                    <b>ICD: {row['perturbed_icd_pred']}</b></td>
                </tr>
                </table>
            </div>"""

        rows_html += f"""
        <div style="border:1px solid #ccc;margin:20px 0;
        padding:15px;border-radius:8px;">
            <h3 style="margin:0 0 10px 0;">
                Patient ID: {row['hadm_id']}
                <span style="background:{drift_color};padding:4px 10px;
                border-radius:4px;font-size:0.9em;">{drift_label}</span>
            </h3>
            <table style="font-size:0.9em;border-collapse:collapse;
            margin-bottom:10px;">
                <tr>
                    <td style="padding:3px 12px 3px 0;"><b>Ground Truth:</b></td>
                    <td>{row['ground_truth_desc']}
                    ({row['ground_truth_icd']} v{row['ground_truth_version']})</td>
                </tr>
                <tr>
                    <td style="padding:3px 12px 3px 0;"><b>Prediction:</b></td>
                    <td>{row['original_icd_pred']} — {pred_desc}
                    (baseline sim: {baseline_disp})</td>
                </tr>
                <tr>
                    <td style="padding:3px 12px 3px 0;"><b>Baseline Gates:</b></td>
                    <td><small>{gate_display}</small></td>
                </tr>
                <tr>
                    <td style="padding:3px 12px 3px 0;"><b>Changes:</b></td>
                    <td>{changes_display} ({row['changes_count']} words)</td>
                </tr>
                <tr>
                    <td style="padding:3px 12px 3px 0;"><b>Text Similarity:</b></td>
                    <td>{sim_display}</td>
                </tr>
            </table>
            {text_section}
        </div>"""

    valid_df    = df[df['baseline_correct'] == True]
    cat_rate    = (valid_df['categorical_drift'].mean() * 100
                   if len(valid_df) > 0 else 0)
    sub_rate    = (valid_df['subcategory_drift'].mean() * 100
                   if len(valid_df) > 0 else 0)
    avg_sim     = (f"{valid_df['semantic_similarity'].mean():.4f}"
                   if len(valid_df) > 0 else 'N/A')
    avg_changes = (f"{valid_df['changes_count'].mean():.1f}"
                   if len(valid_df) > 0 else 'N/A')

    html = f"""<!DOCTYPE html>
<html><head><title>Robustness Report</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1400px;margin:0 auto;
     padding:20px;background:#fafafa;}}
h1{{color:#333;}} h3{{margin:0 0 10px 0;}}
.summary{{background:#f0f0f0;padding:15px;border-radius:8px;
          margin-bottom:30px;}}
.summary p{{margin:4px 0;}}
.legend{{display:flex;gap:16px;margin-bottom:16px;font-size:0.85em;flex-wrap:wrap;}}
.legend span{{padding:3px 10px;border-radius:4px;}}
.method-note{{background:#e8f4f8;padding:12px;border-radius:6px;
              margin-bottom:20px;font-size:0.9em;color:#555;}}
</style></head><body>
<h1>Clinical Note Robustness Report</h1>
<p style="color:#666;">Model: {MODEL_NAME} | Synonym validation: 3-layer
(pattern filter → structural check → LLM self-verification)</p>

<div class="method-note">
  <b>Perturbation validation method:</b> Each synonym passes three independent
  checks before being applied: (1) entity category pattern filter blocks
  inherently unsafe entity types (negations, polarities, laterality, etc.),
  (2) structural rules enforce word-count and length constraints, (3) LLM
  self-check asks the model to verify clinical equivalence and grammatical
  fit. This replaces hardcoded word blocklists with a generalizable system.
</div>

<div class="legend">
  <span style="background:#ccffcc;">STABLE — exact ICD match</span>
  <span style="background:#fff3cd;">SUBCATEGORY DRIFT — same 3-char family</span>
  <span style="background:#ffcccc;">CATEGORICAL DRIFT — different disease (primary metric)</span>
</div>
<div class="summary">
    <h2>Summary</h2>
    <p><b>Total Samples:</b> {len(df)}</p>
    <p><b>Valid for Analysis:</b> {len(valid_df)}
       <span style="color:#888;">(gate: LLM clinical equivalence = YES)</span></p>
    <p><b>Skipped (wrong baseline):</b> {len(df) - len(valid_df)}</p>
    <p><b>Categorical Drift Rate:</b> {cat_rate:.1f}%
       <span style="color:#888;">(primary robustness metric)</span></p>
    <p><b>Subcategory Drift Rate:</b> {sub_rate:.1f}%
       <span style="color:#888;">(minor specificity shifts — not counted as true errors)</span></p>
    <p><b>Avg Semantic Similarity:</b> {avg_sim}</p>
    <p><b>Avg Changes Per Note:</b> {avg_changes}</p>
</div>
{rows_html}
</body></html>"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML report saved: {filename}")


# ============================================================
# SECTION 10: MAIN
# ============================================================
def main():
    debug_print("MAIN", f"Starting Run | LLM: {MODEL_NAME}")

    debug_print("MAIN", "Testing API connection...")
    if not query_rwth_server("Say hello"):
        debug_print("ERROR", "Cannot connect to RWTH server. Check VPN.")
        return
    print("LLM API OK")

    try:
        df = MIMICLoader.load(NOTES_PATH, DIAGNOSES_PATH, DICT_PATH, MAX_SAMPLES)
    except Exception as e:
        debug_print("ERROR", f"Data load failed: {e}")
        return

    perturber = GenerativePerturber()
    scorer    = SemanticEvaluator()
    results   = []

    print("\n" + "=" * 80)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            hadm_id   = row['hadm_id']
            orig_text = row['processed_text']
            gt_desc   = row['long_title']
            gt_icd    = row['icd_code']
            icd_ver   = row['icd_version']

            print(f"\n[ID: {hadm_id}] Processing...")
            print(f"   > Ground Truth  : {gt_desc} ({gt_icd} v{icd_ver})")

            # ── Step 1: Original ICD prediction ───────────────────────────
            orig_icd  = get_icd_prediction(orig_text)
            orig_desc = get_diagnosis_description(orig_icd, 10)
            print(f"   > Original ICD  : {orig_icd} — {orig_desc}")

            # ── Step 2: Baseline correctness (LLM-gated) ──────────────────
            baseline_ok, baseline_sim, llm_ok, gate_str = is_baseline_correct(
                scorer, orig_desc, gt_desc, orig_icd, gt_icd)
            print(f"   > Baseline Gates: {gate_str}")
            print(f"   > Decision      : "
                  f"{'MATCH — proceeding' if baseline_ok else 'MISMATCH — skipping'}")

            if not baseline_ok:
                results.append({
                    "hadm_id"             : hadm_id,
                    "ground_truth_icd"    : gt_icd,
                    "ground_truth_version": icd_ver,
                    "ground_truth_desc"   : gt_desc,
                    "original_text"       : orig_text,
                    "perturbed_text"      : "SKIPPED",
                    "changes_made"        : "SKIPPED",
                    "changes_count"       : 0,
                    "original_icd_pred"   : orig_icd,
                    "original_pred_desc"  : orig_desc,
                    "baseline_sim"        : baseline_sim,
                    "baseline_llm_ok"     : llm_ok,
                    "baseline_gate"       : gate_str,
                    "baseline_correct"    : False,
                    "perturbed_icd_pred"  : "SKIPPED",
                    "icd_codes_differ"    : False,
                    "drift_level"         : "skipped",
                    "categorical_drift"   : False,
                    "subcategory_drift"   : False,
                    "suspicious_drift"    : False,
                    "semantic_similarity" : 0.0
                })
                print("-" * 40)
                continue

            # ── Step 3: Three-layer validated perturbation ─────────────────
            print(f"   > Perturbing...")
            pert_text, changes_log = perturber.perturb(orig_text)

            if changes_log:
                print(f"   > Attack Status : {len(changes_log)} validated changes.")
            else:
                print("   > Attack Status : No valid synonyms found — "
                      "all candidates blocked by validation layers.")

            # ── Step 4: Perturbed ICD prediction ──────────────────────────
            pert_icd = get_icd_prediction(pert_text)
            print(f"   > Perturbed ICD : {pert_icd}")

            # ── Step 5: Drift classification + similarity ──────────────────
            drift_info = classify_drift(orig_icd, pert_icd)
            sim        = scorer.get_sim(orig_text, pert_text)
            suspicious = (drift_info['categorical_drift']
                          and sim > 0.995 and len(changes_log) <= 3)

            print(f"   > Drift Level   : {drift_info['drift_level']}")
            print(f"   > Text Sim      : {sim:.4f}")
            if suspicious:
                print(f"   > WARNING       : Suspicious drift — high sim + "
                      f"few changes suggests LLM instability, not perturbation effect")

            changes_str = " | ".join([f"{o} -> {n}" for o, n in changes_log])

            results.append({
                "hadm_id"             : hadm_id,
                "ground_truth_icd"    : gt_icd,
                "ground_truth_version": icd_ver,
                "ground_truth_desc"   : gt_desc,
                "original_text"       : orig_text,
                "perturbed_text"      : pert_text,
                "changes_made"        : changes_str,
                "changes_count"       : len(changes_log),
                "original_icd_pred"   : orig_icd,
                "original_pred_desc"  : orig_desc,
                "baseline_sim"        : baseline_sim,
                "baseline_llm_ok"     : llm_ok,
                "baseline_gate"       : gate_str,
                "baseline_correct"    : True,
                "perturbed_icd_pred"  : pert_icd,
                **drift_info,
                "suspicious_drift"    : suspicious,
                "semantic_similarity" : sim
            })
            print("-" * 40)

        except Exception as e:
            debug_print("ERROR", f"Row {idx} failed: {e}")
            continue

    if not results:
        print("No results to save.")
        return

    final_df = pd.DataFrame(results)

    csv_df = final_df.drop(
        columns=['original_text', 'perturbed_text'], errors='ignore')
    csv_df.to_csv(FINAL_RESULTS, index=False)

    plot_results(final_df)
    generate_html_report(final_df)

    valid_df = final_df[final_df['baseline_correct'] == True]
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print(f"  Total samples       : {len(final_df)}")
    print(f"  Valid (LLM matched) : {len(valid_df)}")
    print(f"  Skipped             : {len(final_df) - len(valid_df)}")
    if len(valid_df) > 0:
        print(f"  Categorical drift   : "
              f"{valid_df['categorical_drift'].mean() * 100:.1f}%  ← primary metric")
        print(f"  Subcategory drift   : "
              f"{valid_df['subcategory_drift'].mean() * 100:.1f}%  ← minor noise")
        print(f"  Avg text sim        : "
              f"{valid_df['semantic_similarity'].mean():.4f}")
        print(f"  Avg changes/note    : "
              f"{valid_df['changes_count'].mean():.1f}")
        print(f"  Suspicious drifts   : "
              f"{valid_df['suspicious_drift'].sum()}")
    print(f"  CSV saved           : {FINAL_RESULTS}")
    print(f"  HTML saved          : {HTML_REPORT}")
    print(f"  Plot saved          : {PLOT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()