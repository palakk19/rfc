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

FINAL_RESULTS  = "final_robustness.csv"
HTML_REPORT    = "robustness_report.html"
PLOT_FILE      = "robustness_plots.png"
MAX_SAMPLES    = 20
DEVICE         = "cpu"

# Semantic similarity threshold for description matching.
# Intentionally softer (0.78) because we compare free-text description
# strings that may use synonyms or different specificity levels.
# The LLM equivalence check acts as the second, smarter gate.
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
# SECTION 3: PERTURBATION
# ============================================================
NEVER_REPLACE_PHRASES = (
    'no ', 'not ', 'denies ', 'denied ', 'without ',
    'negative for ', 'absence of ', 'no evidence of ',
)

NEVER_REPLACE_WORDS = {
    'personal', 'family',
    'minutes', 'seconds', 'hours', 'days', 'weeks', 'months',
    'proximal', 'distal', 'anterior', 'posterior',
    'bilateral', 'unilateral', 'ipsilateral', 'contralateral',
    'normotensive', 'afebrile', 'atraumatic', 'anicteric',
    'nontender', 'nonsmoker', 'nonradiating', 'asymptomatic',
    'denies', 'denied', 'stable', 'unstable',
}


def is_section_header(text: str) -> bool:
    """Return True if the entity looks like an all-caps section header
    e.g. HISTORY OF PRESENT ILLNESS, ASSESSMENT AND PLAN.
    Replacing these breaks the document structure the LLM uses for parsing."""
    tokens = text.strip().split()
    if len(tokens) < 2:
        return False
    return all(t.isupper() and t.isalpha() for t in tokens)


def is_valid_replacement(original: str, replacement: str) -> bool:
    """Guard against bad LLM expansions like 'palpation' -> 'Per rectal examination'.
    Rules:
      - Replacement cannot be more than 3x longer than original
      - If original is a single word, replacement must be ≤ 3 words
    """
    orig_words = len(original.split())
    repl_words = len(replacement.split())
    if repl_words > orig_words * 3 or repl_words > orig_words + 4:
        return False
    if orig_words == 1 and repl_words > 3:
        return False
    return True


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

    def get_generative_synonym(self, word, context_sentence):
        is_acronym = word.isupper() and len(word) <= 5
        prompt = (
            f"Context: \"{context_sentence}\"\n"
            f"Term: '{word}'\n\n"
            f"Instructions:\n"
            f"1. Replace '{word}' with one medically equivalent synonym.\n"
            f"2. CATEGORY RULE: A fistula is a surgical/pathological connection; "
            f"a 'sinus tract' is NOT a synonym. Maintain exact pathology.\n"
            f"3. ACRONYM RULE: If the term is an abbreviation (like CTA, VTX, PE), "
            f"ONLY provide the full formal expansion. If you are unsure of the "
            f"expansion in this context, return the original term '{word}'.\n"
            f"4. LENGTH RULE: If the original term is one word, reply with at most "
            f"three words. Do not expand into full phrases or sentences.\n"
            f"5. Respond with ONLY the replacement text.\n"
            f"Answer:"
        )
        synonym = query_rwth_server(prompt)
        if synonym:
            synonym = synonym.replace('"', '').replace('.', '').strip()
            if len(word) > 0 and (len(synonym) / len(word)) > 5:
                return None
            if is_acronym and len(synonym.split()) == 1 \
                    and synonym.lower() != word.lower():
                return None
            if not synonym or synonym.lower() == word.lower():
                return None
        return synonym

    def perturb(self, text, perturbation_rate=0.10):
        doc = self.nlp(text)
        valid_entities = [ent for ent in doc.ents
                          if not self.is_negated(ent[0])]
        if not valid_entities:
            return text, []
        num_to_perturb = max(1, int(len(valid_entities) * perturbation_rate))
        target_entities = random.sample(
            valid_entities, min(num_to_perturb, len(valid_entities)))
        target_entities.sort(key=lambda x: x.start_char, reverse=True)
        perturbed_text = text
        changes_log = []
        for ent in target_entities:
            if ent.text.lower().startswith(NEVER_REPLACE_PHRASES):
                print(f"     BLOCKED (negation phrase): '{ent.text}'")
                continue
            if ent.text.lower().strip() in NEVER_REPLACE_WORDS:
                print(f"     BLOCKED (safety blocklist): '{ent.text}'")
                continue
            if is_section_header(ent.text):
                print(f"     BLOCKED (section header): '{ent.text}'")
                continue

            context_window = (ent.sent.text if ent.sent
                              else text[max(0, ent.start_char-50):ent.end_char+50])
            replacement = self.get_generative_synonym(ent.text, context_window)

            if replacement and replacement.lower() != ent.text.lower():
                if len(replacement) > (len(ent.text) * 4):
                    continue
                if not is_valid_replacement(ent.text, replacement):
                    print(f"     BLOCKED (bad expansion): '{ent.text}' -> '{replacement}'")
                    continue
                perturbed_text = (perturbed_text[:ent.start_char]
                                  + replacement
                                  + perturbed_text[ent.end_char:])
                changes_log.append((ent.text, replacement))
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

    def get_sim(self, text1, text2):
        with torch.no_grad():
            inputs = self.tokenizer(
                [text1, text2], padding=True, truncation=True,
                max_length=512, return_tensors="pt").to(DEVICE)
            outputs = self.model(**inputs)
            e1 = outputs.last_hidden_state[0, 0, :].unsqueeze(0)
            e2 = outputs.last_hidden_state[1, 0, :].unsqueeze(0)
            sim = cosine_similarity(e1.numpy(), e2.numpy())[0][0]
        return float(sim)

    def descriptions_match(self, desc1, desc2, threshold=BASELINE_SIM_THRESHOLD):
        """Semantic similarity only — no ICD string comparison.
        Threshold is intentionally softer (0.78) because free-text descriptions
        of the same condition may use synonyms or different specificity levels.
        The LLM equivalence check in check_clinical_equivalence() acts as the
        second, smarter correctness gate."""
        sim = self.get_sim(desc1, desc2)
        return sim >= threshold, sim


# ============================================================
# SECTION 5: BASELINE CORRECTNESS CHECK
# ============================================================

def check_clinical_equivalence(pred_icd: str, gt_icd: str, gt_desc: str) -> bool:
    """Ask the LLM whether the predicted ICD code represents the same primary
    condition as the ground truth.

    This is the key fix for the ICD-9 vs ICD-10 mismatch problem:
    MIMIC has mixed ICD-9 and ICD-10 ground truth codes, but the LLM always
    outputs ICD-10. A string/prefix comparison will always fail for ICD-9
    ground truths even when the prediction is clinically correct.
    e.g. ground truth '56081 v9' (intestinal obstruction ICD-9)
         prediction   'K56.9'    (intestinal obstruction ICD-10)
         → prefix '560' ≠ 'K56' → old code skipped this → WRONG
         → LLM says YES          → new code proceeds     → CORRECT
    """
    if pred_icd == "UNKNOWN":
        return False
    prompt = (
        f"Ground truth diagnosis: '{gt_desc}' (ICD code: {gt_icd})\n"
        f"Predicted ICD code: {pred_icd}\n\n"
        f"Question: Does the predicted ICD code represent the same primary "
        f"medical condition as the ground truth diagnosis? "
        f"Ignore differences in code version (ICD-9 vs ICD-10) and minor "
        f"specificity differences. Focus only on whether the core condition is "
        f"the same.\n"
        f"Answer with only YES or NO."
    )
    response = query_rwth_server(prompt, max_tokens=5)
    return response.strip().upper().startswith("YES")


def is_baseline_correct(scorer: SemanticEvaluator,
                         orig_desc: str, gt_desc: str,
                         orig_icd: str, gt_icd: str) -> tuple:
    """Combined correctness gate using two independent checks:
      1. ClinicalBERT semantic similarity between description strings (≥ 0.78)
      2. LLM clinical equivalence check (handles ICD-9/10 mismatch)

    A sample passes if EITHER check is True — this is intentionally generous
    because we want to catch genuine correctness, not penalise vocabulary
    differences or version mismatches.

    Both sim score and individual check results are returned for logging.
    """
    sim_ok, baseline_sim = scorer.descriptions_match(orig_desc, gt_desc)
    llm_ok = check_clinical_equivalence(orig_icd, gt_icd, gt_desc)

    # Log which gate passed for transparency
    gate_str = []
    if sim_ok:
        gate_str.append(f"sem_sim={baseline_sim:.4f}✓")
    else:
        gate_str.append(f"sem_sim={baseline_sim:.4f}✗")
    gate_str.append("llm_equiv=YES✓" if llm_ok else "llm_equiv=NO✗")

    passed = sim_ok or llm_ok
    return passed, baseline_sim, llm_ok, " | ".join(gate_str)


# ============================================================
# SECTION 6: ICD PREDICTION & DESCRIPTION
# ============================================================
def get_icd_prediction(text):
    """Two-attempt ICD extraction with fallback regex.
    Returns 'UNKNOWN' instead of blank so downstream comparisons don't
    silently fail on empty strings."""
    primary_prompt = (
        f"Read the clinical note below. Return ONLY the single most likely "
        f"ICD-10 diagnosis code (example format: J18.9 or I21.0). "
        f"Do not write the disease name. Do not explain. "
        f"Output the code only.\n\n"
        f"Note:\n{text}\n\nICD-10 Code:"
    )
    for attempt in range(2):
        prompt = primary_prompt if attempt == 0 else (
            f"Output a single ICD-10 code for this note. "
            f"Format: letter + 2 digits + optional dot + digits. "
            f"Examples: J18.9, I21.0, K56.9\n\n"
            f"Note:\n{text[:800]}\n\nCode:"
        )
        response = query_rwth_server(prompt)
        if response:
            # Primary: strict dot-format e.g. J18.9
            match = re.search(r'[A-Z]\d{2}\.\d+', response.strip().upper())
            if match:
                return match.group(0)
            # Fallback: no-dot format e.g. J189 or J18
            match = re.search(r'[A-Z]\d{2,4}', response.strip().upper())
            if match:
                return match.group(0)
    return "UNKNOWN"


def get_diagnosis_description(icd_code, icd_version):
    if icd_code == "UNKNOWN":
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
    """Two-level drift classification instead of a single binary flag.
      - categorical : 3-char ICD prefix changed → different disease family
                      this is the meaningful robustness metric
      - subcategory : exact code changed but same 3-char prefix
                      e.g. K74.9 → K74.7 (both liver cirrhosis)
                      this is minor specificity noise, not true drift
      - none        : codes identical
    """
    orig_norm = re.sub(r'[^A-Z0-9]', '', orig_icd.upper())
    pert_norm = re.sub(r'[^A-Z0-9]', '', pert_icd.upper())

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
def plot_results(df):
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
    colors = {'none': '#90EE90', 'subcategory': '#FFD700', 'categorical': '#FF6B6B'}
    bar_colors = [colors.get(k, 'grey') for k in drift_counts.index]
    drift_counts.plot(kind='bar', color=bar_colors)
    cat_rate = valid_df['categorical_drift'].mean() * 100
    plt.title(f"Drift Level Breakdown\nCategorical drift: {cat_rate:.1f}%")
    plt.xlabel("Drift Level")
    plt.xticks(rotation=0)

    plt.subplot(1, 4, 3)
    sns.scatterplot(data=valid_df, x='changes_count',
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
    plt.title("Baseline Quality\n(sim≥0.78 OR LLM equiv=YES)")

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150)
    print(f"Plot saved: {PLOT_FILE}")


# ============================================================
# SECTION 9: HTML REPORT
# ============================================================
def generate_html_report(df, filename=HTML_REPORT):

    DRIFT_COLORS = {
        'none'       : '#ccffcc',
        'subcategory': '#fff3cd',
        'categorical': '#ffcccc',
    }
    DRIFT_LABELS = {
        'none'       : 'STABLE',
        'subcategory': 'SUBCATEGORY DRIFT',
        'categorical': 'CATEGORICAL DRIFT',
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
                       if row['semantic_similarity'] else 'N/A')
        baseline_disp  = f"{float(row.get('baseline_sim', 0)):.4f}"
        pred_desc      = row.get('original_pred_desc', '')
        changes_display = (row['changes_made']
                           if row['changes_made'] not in ['SKIPPED', '', None]
                           else 'None')
        gate_display = row.get('baseline_gate', '')

        if is_skipped:
            text_section = (
                f'<p style="color:#999;font-style:italic;padding:10px;'
                f'background:#f9f9f9;border-radius:4px;">'
                f'Skipped — original prediction did not match ground truth.<br>'
                f'<small>Gates: {gate_display}</small>'
                f'</p>')
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
                    padding:2px 8px;border-radius:3px;">LLM replacements</span>
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
    cat_rate    = (valid_df['categorical_drift'].mean()*100
                   if len(valid_df) > 0 else 0)
    sub_rate    = (valid_df['subcategory_drift'].mean()*100
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
</style></head><body>
<h1>Clinical Note Robustness Report</h1>
<p style="color:#666;">Perturbations generated by {MODEL_NAME} via LLM synonym substitution.</p>
<div class="legend">
  <span style="background:#ccffcc;">STABLE — exact ICD match</span>
  <span style="background:#fff3cd;">SUBCATEGORY DRIFT — same 3-char family, different specificity</span>
  <span style="background:#ffcccc;">CATEGORICAL DRIFT — different disease category (true error)</span>
</div>
<div class="summary">
    <h2>Summary</h2>
    <p><b>Total Samples:</b> {len(df)}</p>
    <p><b>Valid for Analysis (correct baseline):</b> {len(valid_df)}
       <span style="color:#888;">(sem_sim ≥ 0.78 OR LLM clinical equivalence = YES)</span></p>
    <p><b>Skipped (wrong baseline):</b> {len(df)-len(valid_df)}</p>
    <p><b>Categorical Drift Rate:</b> {cat_rate:.1f}%
       <span style="color:#888;">(different disease category — primary robustness metric)</span></p>
    <p><b>Subcategory Drift Rate:</b> {sub_rate:.1f}%
       <span style="color:#888;">(same category, different specificity — minor noise)</span></p>
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

    print("\n" + "="*80)

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

            # ── Step 2: Two-gate baseline correctness check ───────────────
            # Gate 1: ClinicalBERT semantic similarity ≥ 0.78
            # Gate 2: LLM says predicted ICD is clinically equivalent to GT
            # Passes if EITHER gate is True.
            # This correctly handles ICD-9 ground truths (MIMIC has both
            # ICD-9 and ICD-10) where a prefix string match always fails
            # even when the prediction is clinically correct.
            baseline_ok, baseline_sim, llm_ok, gate_str = is_baseline_correct(
                scorer, orig_desc, gt_desc, orig_icd, gt_icd)

            print(f"   > Baseline Gates: {gate_str}")
            print(f"   > Baseline      : {'MATCH — proceeding' if baseline_ok else 'MISMATCH — skipping'}")

            # ── Step 3: Skip if both gates failed ─────────────────────────
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
                print("-"*40)
                continue

            # ── Step 4: LLM perturbation ───────────────────────────────────
            pert_text, changes_log = perturber.perturb(orig_text)

            if changes_log:
                print(f"   > Attack Status : {len(changes_log)} words changed.")
                for orig_word, new_word in changes_log:
                    print(f"     * '{orig_word}' -> '{new_word}'")
            else:
                print("   > Attack Status : FAILED (no synonyms found).")

            # ── Step 5: Perturbed ICD prediction ──────────────────────────
            pert_icd = get_icd_prediction(pert_text)
            print(f"   > Perturbed ICD : {pert_icd}")

            # ── Step 6: Two-level drift + similarity ──────────────────────
            drift_info = classify_drift(orig_icd, pert_icd)
            sim        = scorer.get_sim(orig_text, pert_text)
            # Suspicious = categorical drift happened but text barely changed
            # — likely LLM instability rather than real perturbation effect
            suspicious = (drift_info['categorical_drift']
                          and sim > 0.995 and len(changes_log) <= 3)

            print(f"   > Drift Level   : {drift_info['drift_level']}")
            print(f"   > Text Sim      : {sim:.4f}")
            if suspicious:
                print(f"   > WARNING       : Suspicious categorical drift "
                      f"(high sim + few changes — possible LLM instability)")

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
            print("-"*40)

        except Exception as e:
            debug_print("ERROR", f"Row {idx} failed: {e}")
            continue

    if results:
        final_df = pd.DataFrame(results)

        csv_df = final_df.drop(
            columns=['original_text', 'perturbed_text'], errors='ignore')
        csv_df.to_csv(FINAL_RESULTS, index=False)

        plot_results(final_df)
        generate_html_report(final_df)

        valid_df = final_df[final_df['baseline_correct'] == True]
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print(f"  Total samples       : {len(final_df)}")
        print(f"  Valid (matched)     : {len(valid_df)}")
        print(f"  Skipped             : {len(final_df)-len(valid_df)}")
        if len(valid_df) > 0:
            print(f"  Categorical drift   : "
                  f"{valid_df['categorical_drift'].mean()*100:.1f}%  ← primary metric")
            print(f"  Subcategory drift   : "
                  f"{valid_df['subcategory_drift'].mean()*100:.1f}%  ← minor noise")
            print(f"  Avg text sim        : "
                  f"{valid_df['semantic_similarity'].mean():.4f}")
            print(f"  Avg words changed   : "
                  f"{valid_df['changes_count'].mean():.1f}")
            print(f"  Suspicious drifts   : "
                  f"{valid_df['suspicious_drift'].sum()}")
        print(f"  CSV saved           : {FINAL_RESULTS}")
        print(f"  HTML saved          : {HTML_REPORT}")
        print(f"  Plot saved          : {PLOT_FILE}")
        print("="*80)


if __name__ == "__main__":
    main()