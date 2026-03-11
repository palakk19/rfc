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

# --- CONFIGURATION ---
MODEL_NAME = "llama3.1:8b"
API_URL = "http://ollama.warhol.informatik.rwth-aachen.de/api/generate"

NOTES_PATH = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz"
DIAGNOSES_PATH = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/diagnoses_icd.csv.gz"
DICT_PATH = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/d_icd_diagnoses.csv.gz"

FINAL_RESULTS       = "final_robustness_api.csv"
SPOT_CHECK_REPORT   = "spot_check_report.csv"   # NEW: for manual/paper review
MAX_SAMPLES         = 8
DEVICE              = "cpu"

warnings.filterwarnings("ignore")

def debug_print(stage, message):
    print(f"\033[94m[{stage}]\033[0m {message}")


# ============================================================
# SECTION 1: API CONNECTOR — No changes
# ============================================================
def query_rwth_server(prompt):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 50}
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
# SECTION 2: DATA LOADER — No changes
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
# SECTION 3: SEMANTIC EVALUATOR — Added descriptions_match()
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

    def descriptions_match(self, desc1, desc2, threshold=0.72):
        sim = self.get_sim(desc1, desc2)
        return sim >= threshold, sim


# ============================================================
# SECTION 4: PERTURBATION VALIDATOR
# Uses ClinicalBERT similarity + antonym check via LLM
# ============================================================
class PerturbationValidator:
    def __init__(self, evaluator, threshold=0.72):
        self.evaluator = evaluator
        self.threshold = threshold

    def validate_synonym(self, original_word, replacement, context_sentence):
        """
        Step 1: Fast pre-checks (no API, no embedding)
        Step 2: ClinicalBERT similarity
        Step 3: Antonym check via LLM (only if similarity passes)
        Returns (is_valid, sim_score, reason)
        """
        # ── PRE-CHECK: formatting artifacts ──────────────────────────────
        if any(p in replacement for p in ['->', '→', '\n', '**', '##']):
            return False, 0.0, "formatting artifact"

        # ── PRE-CHECK: too verbose ────────────────────────────────────────
        if len(replacement.split()) > len(original_word.split()) + 4:
            return False, 0.0, "too verbose"

        # ── CLINICALBERT SIMILARITY ───────────────────────────────────────
        sim = self.evaluator.get_sim(original_word, replacement)
        if sim < self.threshold:
            return False, sim, f"low similarity ({sim:.3f} < {self.threshold})"

        # ── ANTONYM CHECK: only called if similarity passed ───────────────
        # ClinicalBERT scores antonyms highly because they appear in same
        # clinical contexts (normotensive/hypertensive, denies/reports etc.)
        # A focused yes/no antonym question to LLM is more reliable here
        if self.is_antonym(original_word, replacement):
            return False, sim, f"antonyms detected (sim={sim:.3f} was misleading)"

        return True, sim, f"sim={sim:.3f}"

    def is_antonym(self, word1, word2):
        """
        Strictly checks for TRUE opposites only.
        Only catches cases where meaning is directly reversed —
        NOT near-synonyms or related terms.
        """
        prompt = (
            f"Are '{word1}' and '{word2}' TRUE opposites with DIRECTLY "
            f"OPPOSITE meanings?\n\n"
            f"Answer YES only for TRUE opposites like:\n"
            f"- normotensive/hypertensive (opposite blood pressure states)\n"
            f"- increased/decreased (opposite directions)\n"
            f"- positive/negative (opposite test results)\n"
            f"- present/absent, alive/dead\n\n"
            f"Answer NO for these — they are synonyms NOT opposites:\n"
            f"- vomiting/emesis (same symptom, different word)\n"
            f"- Lasix/Furosemide (brand and generic of same drug)\n"
            f"- hospital stay/admission (same concept)\n"
            f"- leg swelling/edema (same condition)\n"
            f"- dilated/distended (same meaning)\n"
            f"- exploratory surgery/diagnostic laparotomy (same procedure)\n"
            f"- renal function/kidney function (same thing)\n\n"
            f"Answer only YES (true opposites) or NO (not opposites)."
        )
        response = query_rwth_server(prompt)
        return "YES" in response.strip().upper()


# ============================================================
# SECTION 5: NEVER-REPLACE BLOCKLIST
# NEW: Hard-coded linguistic rules — no API, no model needed.
# These words should NEVER be replaced regardless of any
# similarity score because:
# - Negations: flipping them reverses clinical meaning entirely
#   "denies chest pain" != "reports chest pain"
# - Non/a-prefix words: defined by their negation
#   normotensive != hypertensive
# - Severity words: changing them changes clinical severity
#   mild != severe
# - Temporal words: changing them changes patient history
#   "never smoked" != "previously smoked"
# ============================================================
NEVER_REPLACE = {
    # Negation words
    'no', 'not', 'never', 'denies', 'denied', 'without',
    'none', 'negative', 'absent', 'unremarkable',
    # Words defined by their negation prefix
    'normotensive', 'afebrile', 'atraumatic', 'anicteric',
    'nonradiating', 'nonsmoker', 'nontender', 'nonverbal',
    'asymptomatic', 'afebrile',
    # Severity/degree words — changing these changes clinical meaning
    'mild', 'moderate', 'severe', 'acute', 'chronic',
    'stable', 'unstable', 'worsening', 'improving',
    # Temporal/history words — "never" vs "prior" completely differs
    'prior', 'previous', 'former', 'current', 'active', 'ongoing',
    # Quantity words
    'few', 'many', 'multiple', 'single', 'bilateral', 'unilateral',
}


# ============================================================
# SECTION 6: GENERATIVE PERTURBER
# CHANGES:
#   1. Accepts evaluator for ClinicalBERT validation
#   2. Checks NEVER_REPLACE blocklist before generating
#   3. Improved prompt with explicit prohibitions
#   4. Three-layer validation: blocklist → similarity → antonym
#   5. Logs ALL decisions (accepted/rejected/blocked) for spot check
# ============================================================
class GenerativePerturber:
    def __init__(self, evaluator):
        self.validator = PerturbationValidator(evaluator, threshold=0.72)
        try:
            self.nlp = spacy.load("en_core_sci_md")
        except OSError:
            self.nlp = spacy.load("en_core_sci_sm")

    def is_negated(self, token):
        negation_terms = {'no', 'not', 'denies', 'negative', 'without',
                          'none', 'denied'}
        window = [t.text.lower()
                  for t in list(token.doc)[max(0, token.i - 3):token.i]]
        return any(term in window for term in negation_terms)

    def get_generative_synonym(self, word, context_sentence):
        is_acronym = word.isupper() and len(word) <= 5
        prompt = (
            f"You are a medical scribe. Replace ONE term with a safe synonym "
            f"that preserves the EXACT clinical meaning.\n\n"
            f"Context: \"{context_sentence}\"\n"
            f"Term to replace: '{word}'\n\n"
            f"STRICT RULES:\n"
            f"1. SAFE: brand->generic drug (Tylenol->Acetaminophen), "
            f"Latin equivalents (kidney->renal), "
            f"direct synonyms (fever->pyrexia, vomiting->emesis)\n"
            f"2. NEVER change body part specificity "
            f"(foot != toe, small intestine != jejunum)\n"
            f"3. NEVER add information "
            f"(pain != severe pain, negative != normal)\n"
            f"4. NEVER swap drugs for different drugs "
            f"(methotrexate != folinic acid, methotrexate != leucovorin)\n"
            f"5. NEVER change test types "
            f"(urine culture != urinalysis)\n"
            f"6. NEVER replace negation words "
            f"(denies, never, no, without, negative)\n"
            f"7. NEVER replace antonyms as synonyms "
            f"(normotensive != hypertensive, increased != decreased)\n"
            f"8. If NO safe synonym exists, reply exactly: NONE\n"
            f"9. Reply with ONLY the synonym. No explanations. No arrows.\n\n"
            f"Answer:"
        )
        synonym = query_rwth_server(prompt)
        if not synonym:
            return None
        synonym = synonym.replace('"', '').replace('.', '').strip()
        if synonym.upper() == "NONE" or not synonym:
            return None
        if any(p in synonym for p in ['->', '→', '\n', '**']):
            return None
        if len(word) > 0 and (len(synonym) / len(word)) > 5:
            return None
        if is_acronym and len(synonym.split()) == 1 and \
           synonym.lower() != word.lower():
            return None
        if synonym.lower() == word.lower():
            return None
        return synonym

    def perturb(self, text, perturbation_rate=0.10):
        doc = self.nlp(text)
        valid_entities = [ent for ent in doc.ents if not self.is_negated(ent[0])]

        if not valid_entities:
            return text, [], []  # text, changes_log, decisions_log

        num_to_perturb = max(1, int(len(valid_entities) * perturbation_rate))
        target_entities = random.sample(
            valid_entities, min(num_to_perturb, len(valid_entities)))
        target_entities.sort(key=lambda x: x.start_char, reverse=True)

        perturbed_text = text
        changes_log   = []
        decisions_log = []  # NEW: records every decision for spot check report

        for ent in target_entities:
            context_window = ent.sent.text if ent.sent else \
                text[max(0, ent.start_char - 50):ent.end_char + 50]

            # ── LAYER 1: BLOCKLIST CHECK ──────────────────────────────────
            # Zero API calls, zero embeddings — pure string check
            if ent.text.lower().strip() in NEVER_REPLACE:
                print(f"     🚫 BLOCKED:   '{ent.text}' (in never-replace list)")
                decisions_log.append({
                    "original"   : ent.text,
                    "replacement": "N/A",
                    "decision"   : "BLOCKED",
                    "reason"     : "in never-replace blocklist",
                    "sim_score"  : "N/A"
                })
                continue

            # ── GENERATE SYNONYM ──────────────────────────────────────────
            replacement = self.get_generative_synonym(ent.text, context_window)

            if not replacement or replacement.lower() == ent.text.lower():
                decisions_log.append({
                    "original"   : ent.text,
                    "replacement": replacement or "NONE",
                    "decision"   : "SKIPPED",
                    "reason"     : "no synonym generated",
                    "sim_score"  : "N/A"
                })
                continue

            if len(replacement) > (len(ent.text) * 4):
                decisions_log.append({
                    "original"   : ent.text,
                    "replacement": replacement,
                    "decision"   : "REJECTED",
                    "reason"     : "length ratio too high",
                    "sim_score"  : "N/A"
                })
                continue

            # ── LAYER 2 + 3: CLINICALBERT + ANTONYM CHECK ────────────────
            is_valid, sim_score, reason = self.validator.validate_synonym(
                ent.text, replacement, context_window)

            if not is_valid:
                print(f"     ⚠️  REJECTED:  '{ent.text}' -> '{replacement}' ({reason})")
                decisions_log.append({
                    "original"   : ent.text,
                    "replacement": replacement,
                    "decision"   : "REJECTED",
                    "reason"     : reason,
                    "sim_score"  : round(sim_score, 4)
                })
                continue

            # ── ACCEPTED ──────────────────────────────────────────────────
            print(f"     ✅ ACCEPTED:  '{ent.text}' -> '{replacement}' (sim={sim_score:.3f})")
            decisions_log.append({
                "original"   : ent.text,
                "replacement": replacement,
                "decision"   : "ACCEPTED",
                "reason"     : reason,
                "sim_score"  : round(sim_score, 4)
            })

            perturbed_text = (perturbed_text[:ent.start_char]
                              + replacement
                              + perturbed_text[ent.end_char:])
            changes_log.append((ent.text, replacement))

        return perturbed_text, changes_log, decisions_log


# ============================================================
# SECTION 7: ICD PREDICTION — No changes
# ============================================================
def get_icd_prediction(text):
    prompt = (
        f"Read the clinical note below. Return ONLY the single most likely "
        f"ICD-10 diagnosis code (example format: J18.9 or I21.0). "
        f"Do not write the disease name. Do not explain. Output the code only.\n\n"
        f"Note:\n{text}\n\nICD-10 Code:"
    )
    response = query_rwth_server(prompt)
    match = re.search(r'[A-Z]\d{2}\.?\d*', response.strip().upper())
    return match.group(0) if match else response.strip()


# ============================================================
# SECTION 8: DIAGNOSIS DESCRIPTION — No changes
# ============================================================
def get_diagnosis_description(icd_code, icd_version):
    prompt = (
        f"What medical condition does ICD-{icd_version} code "
        f"'{icd_code}' represent? "
        f"Reply with ONLY the condition name in 2-5 words. No explanation."
    )
    return query_rwth_server(prompt)


# ============================================================
# SECTION 9: SPOT CHECK REPORT
# NEW: Generates a CSV with every single perturbation decision
# (accepted, rejected, blocked) across all samples.
#
# HOW TO USE FOR YOUR PAPER:
# 1. Open spot_check_report.csv in Excel
# 2. Filter column "decision" to show only "ACCEPTED" rows
# 3. For each row, ask yourself: does the replacement preserve meaning?
#    You do NOT need medical knowledge for most — linguistic sense is enough:
#    - "normotensive -> Hypertensive" — obviously wrong (opposites)
#    - "fever -> pyrexia" — obviously correct (same word different language)
#    - "vomiting -> emesis" — obviously correct
#    - "methotrexate -> leucovorin" — flag as suspicious (different drugs)
# 4. Add a column "manual_valid" and mark Y/N for each accepted row
# 5. Count: valid_accepted / total_accepted = your perturbation quality rate
# 6. In your paper write:
#    "Manual spot check of N perturbation pairs confirmed X% were valid
#     medical synonyms. The remaining Y% were flagged as potentially
#     altering clinical meaning and excluded from drift analysis."
# ============================================================
def generate_spot_check_report(all_decisions, filename=SPOT_CHECK_REPORT):
    """
    Saves every perturbation decision to CSV for manual review.
    Columns:
      hadm_id     — patient case
      original    — word that was considered for replacement
      replacement — proposed synonym
      decision    — ACCEPTED / REJECTED / BLOCKED / SKIPPED
      reason      — why (sim score, antonym, blocklist etc.)
      sim_score   — ClinicalBERT similarity (N/A if not computed)
      manual_valid — empty column for you to fill in during spot check
    """
    if not all_decisions:
        return

    rows = []
    for hadm_id, decisions in all_decisions:
        for d in decisions:
            rows.append({
                "hadm_id"     : hadm_id,
                "original"    : d["original"],
                "replacement" : d["replacement"],
                "decision"    : d["decision"],
                "reason"      : d["reason"],
                "sim_score"   : d["sim_score"],
                "manual_valid": ""   # ← you fill this in during spot check
            })

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)

    # Print summary
    total     = len(df)
    accepted  = len(df[df['decision'] == 'ACCEPTED'])
    rejected  = len(df[df['decision'] == 'REJECTED'])
    blocked   = len(df[df['decision'] == 'BLOCKED'])
    skipped   = len(df[df['decision'] == 'SKIPPED'])

    print(f"\n📋 SPOT CHECK REPORT SUMMARY")
    print(f"   Total decisions : {total}")
    print(f"   ✅ Accepted     : {accepted} ({100*accepted/total:.1f}%)")
    print(f"   ⚠️  Rejected     : {rejected} ({100*rejected/total:.1f}%)")
    print(f"   🚫 Blocked      : {blocked}  ({100*blocked/total:.1f}%)")
    print(f"   ℹ️  Skipped      : {skipped}  ({100*skipped/total:.1f}%)")
    print(f"   Saved to        : {filename}")
    print(f"   → Open in Excel, filter decision=ACCEPTED, fill manual_valid column")


# ============================================================
# SECTION 10: VISUALIZATION — Filter to valid samples only
# ============================================================
def plot_results(df):
    debug_print("PLOT", "Generating plots...")
    valid_df = df[df['baseline_correct'] == True]
    if len(valid_df) == 0:
        print("⚠️  No valid samples to plot.")
        return

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    sns.histplot(valid_df['semantic_similarity'], kde=True, bins=15, color='teal')
    plt.title("Semantic Preservation\n(correct baselines only)")
    plt.xlabel("Cosine Similarity")

    plt.subplot(1, 3, 2)
    drift_rate = valid_df['icd_codes_differ'].mean() * 100
    sns.countplot(x='icd_codes_differ', data=valid_df, palette='viridis')
    plt.title(f"ICD Prediction Stability\nDrift Rate: {drift_rate:.1f}%")
    plt.xlabel("ICD Codes Differ After Perturbation")
    plt.xticks([0, 1], ['Stable', 'Drifted'])

    plt.subplot(1, 3, 3)
    counts = df['baseline_correct'].value_counts()
    labels, colors = [], []
    if True in counts.index:
        labels.append(f'Correct Baseline\n(n={counts[True]})')
        colors.append('#90EE90')
    if False in counts.index:
        labels.append(f'Wrong Baseline\n(n={counts[False]})')
        colors.append('#FFB6C1')
    plt.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%')
    plt.title("Baseline Prediction Quality")

    plt.tight_layout()
    plt.savefig("robustness_api_plots.png")
    print("✅ Plot saved to robustness_api_plots.png")


# ============================================================
# SECTION 11: HTML REPORT — Highlights changed words
# ============================================================
def generate_html_report(df, filename="robustness_report.html"):

    def highlight_changes(original_text, perturbed_text, changes_str):
        if not changes_str or changes_str == "SKIPPED":
            return original_text, perturbed_text
        highlighted_orig = original_text
        highlighted_pert = perturbed_text
        pairs = [p.strip() for p in changes_str.split("|")]
        for pair in pairs:
            if " -> " not in pair:
                continue
            orig_word, new_word = pair.split(" -> ", 1)
            orig_word = orig_word.strip()
            new_word  = new_word.strip()
            highlighted_orig = re.sub(
                rf'\b{re.escape(orig_word)}\b',
                f'<span style="background-color:#FF6B6B; color:white; '
                f'padding:1px 4px; border-radius:3px; font-weight:bold;">'
                f'{orig_word}</span>',
                highlighted_orig, flags=re.IGNORECASE)
            highlighted_pert = re.sub(
                rf'\b{re.escape(new_word)}\b',
                f'<span style="background-color:#51C878; color:white; '
                f'padding:1px 4px; border-radius:3px; font-weight:bold;">'
                f'{new_word}</span>',
                highlighted_pert, flags=re.IGNORECASE)
        return highlighted_orig, highlighted_pert

    rows_html = ""
    for _, row in df.iterrows():
        is_skipped  = str(row.get('perturbed_text', 'SKIPPED')) == 'SKIPPED'
        drift_color = "#ffcccc" if row['icd_codes_differ'] else "#ccffcc"
        drift_label = "❌ DRIFTED" if row['icd_codes_differ'] else "✅ STABLE"
        skipped_badge = (
            '<span style="background:#aaa; color:white; padding:4px 10px; '
            'border-radius:4px; font-size:0.9em; margin-left:8px;">'
            '⚠️ SKIPPED — wrong baseline</span>' if is_skipped else "")

        # Pre-compute display values — f-strings cannot handle
        # expressions like {value:.4f if condition else 'N/A'}
        sim_display      = (f"{float(row['semantic_similarity']):.4f}"
                            if row['semantic_similarity'] else 'N/A')
        baseline_display = f"{float(row.get('baseline_sim', 0)):.4f}"
        pred_desc        = row.get('original_pred_desc', '')
        changes_display  = (row['changes_made']
                            if row['changes_made'] not in ['SKIPPED', '', None]
                            else 'None')

        if is_skipped:
            text_section = """
            <p style="color:#999; font-style:italic; padding:10px;
            background:#f9f9f9; border-radius:4px;">
                ⚠️ Perturbation skipped — original prediction did not match
                ground truth so robustness testing would be meaningless.
            </p>"""
        else:
            orig_h, pert_h = highlight_changes(
                row.get('original_text', ''),
                row.get('perturbed_text', ''),
                row.get('changes_made', ''))
            text_section = f"""
            <div style="margin-top:10px;">
                <div style="margin-bottom:8px; font-size:0.85em;">
                    <span style="background:#FF6B6B; color:white;
                    padding:2px 8px; border-radius:3px;">
                    ● Original words (changed)</span>&nbsp;
                    <span style="background:#51C878; color:white;
                    padding:2px 8px; border-radius:3px;">
                    ● Replacement words</span>
                </div>
                <table style="width:100%; border-collapse:collapse;">
                    <tr>
                        <th style="width:50%; background:#e8f4f8; padding:8px;
                        border:1px solid #ccc;">Original Text</th>
                        <th style="width:50%; background:#fff8e8; padding:8px;
                        border:1px solid #ccc;">Perturbed Text</th>
                    </tr>
                    <tr>
                        <td style="padding:10px; border:1px solid #ccc;
                        vertical-align:top; white-space:pre-wrap;
                        font-size:0.85em; line-height:1.6em;">{orig_h}</td>
                        <td style="padding:10px; border:1px solid #ccc;
                        vertical-align:top; white-space:pre-wrap;
                        font-size:0.85em; line-height:1.6em;">{pert_h}</td>
                    </tr>
                    <tr>
                        <td style="padding:8px; border:1px solid #ccc;
                        background:#f0f0f0; text-align:center;">
                        <b>Predicted ICD: {row['original_icd_pred']}</b></td>
                        <td style="padding:8px; border:1px solid #ccc;
                        background:#f0f0f0; text-align:center;">
                        <b>Predicted ICD: {row['perturbed_icd_pred']}</b></td>
                    </tr>
                </table>
            </div>"""

        rows_html += f"""
        <div class="case" style="border:1px solid #ccc; margin:20px 0;
        padding:15px; border-radius:8px;">
            <h3>Patient ID: {row['hadm_id']}
                <span style="background:{drift_color}; padding:4px 10px;
                border-radius:4px; font-size:0.9em;">{drift_label}</span>
                {skipped_badge}
            </h3>
            <table style="font-size:0.9em; border-collapse:collapse;
            margin-bottom:10px;">
                <tr><td style="padding:3px 12px 3px 0;"><b>Ground Truth:</b></td>
                    <td>{row['ground_truth_desc']}
                        ({row['ground_truth_icd']} v{row['ground_truth_version']})
                    </td></tr>
                <tr><td style="padding:3px 12px 3px 0;"><b>Original Prediction:</b></td>
                    <td>{row['original_icd_pred']} — {pred_desc}
                        (baseline sim: {baseline_display})
                    </td></tr>
                <tr><td style="padding:3px 12px 3px 0;"><b>Changes Made:</b></td>
                    <td>{changes_display} ({row['changes_count']} words)
                    </td></tr>
                <tr><td style="padding:3px 12px 3px 0;"><b>Text Similarity:</b></td>
                    <td>{sim_display}
                    </td></tr>
            </table>
            {text_section}
        </div>"""

    valid_df   = df[df['baseline_correct'] == True]
    drift_rate = valid_df['icd_codes_differ'].mean() * 100 if len(valid_df) > 0 else 0
    avg_sim_display     = f"{valid_df['semantic_similarity'].mean():.4f}" if len(valid_df) > 0 else 'N/A'
    avg_changes_display = f"{valid_df['changes_count'].mean():.1f}" if len(valid_df) > 0 else 'N/A'

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Robustness Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1400px;
               margin: 0 auto; padding: 20px; background: #fafafa; }}
        h1 {{ color: #333; }}
        h3 {{ margin: 0 0 10px 0; }}
        .summary {{ background: #f0f0f0; padding: 15px;
                   border-radius: 8px; margin-bottom: 30px; }}
        .summary p {{ margin: 4px 0; }}
    </style>
</head>
<body>
    <h1>🏥 Clinical Note Robustness Report</h1>
    <div class="summary">
        <h2>Summary</h2>
        <p><b>Total Samples:</b> {len(df)}</p>
        <p><b>Valid for Analysis (correct baseline):</b> {len(valid_df)}</p>
        <p><b>Skipped (wrong baseline):</b> {len(df) - len(valid_df)}</p>
        <p><b>ICD Drift Rate:</b> {drift_rate:.1f}%
           <span style="color:#888;">(valid samples only)</span></p>
        <p><b>Avg Semantic Similarity:</b> {avg_sim_display}</p>
        <p><b>Avg Changes Per Note:</b> {avg_changes_display}</p>
    </div>
    {rows_html}
</body>
</html>"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ HTML report saved to {filename} — open in any browser")


# ============================================================
# SECTION 12: MAIN LOOP
# ============================================================
def main():
    debug_print("MAIN", f"Starting Run on {DEVICE} with API Model: {MODEL_NAME}")

    debug_print("MAIN", "Testing API connection...")
    test_resp = query_rwth_server("Say 'hello'")
    if not test_resp:
        debug_print("ERROR", "❌ Cannot connect to RWTH server. Check VPN/URL.")
        return
    print("✅ API Connection Successful.")

    try:
        df = MIMICLoader.load(NOTES_PATH, DIAGNOSES_PATH, DICT_PATH, MAX_SAMPLES)
    except Exception as e:
        debug_print("ERROR", f"Data load failed: {e}")
        return

    scorer   = SemanticEvaluator()
    perturber = GenerativePerturber(evaluator=scorer)
    results       = []
    all_decisions = []  # NEW: collects decisions from every sample for spot check

    print("\n" + "=" * 80)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            hadm_id   = row['hadm_id']
            orig_text = row['processed_text']
            gt_desc   = row['long_title']
            gt_icd    = row['icd_code']
            icd_ver   = row['icd_version']

            print(f"\n[ID: {hadm_id}] Processing...")
            print(f"   > Ground Truth  : {gt_desc} ({gt_icd}, v{icd_ver})")

            # ── STEP 1: Original prediction ───────────────────────────────
            orig_icd = get_icd_prediction(orig_text)
            print(f"   > Original ICD  : {orig_icd}")

            # ── STEP 2: Baseline correctness check ────────────────────────
            orig_desc = get_diagnosis_description(orig_icd, 10)
            print(f"   > Pred Desc     : {orig_desc}")
            baseline_correct, baseline_sim = scorer.descriptions_match(gt_desc, orig_desc)
            print(f"   > Baseline Sim  : {baseline_sim:.4f} → "
                  f"{'✅ MATCH — proceeding' if baseline_correct else '❌ MISMATCH — skipping'}")

            # ── STEP 3: Skip if baseline wrong ────────────────────────────
            if not baseline_correct:
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
                    "baseline_correct"    : False,
                    "perturbed_icd_pred"  : "SKIPPED",
                    "icd_codes_differ"    : False,
                    "suspicious_drift"    : False,
                    "semantic_similarity" : 0.0
                })
                print("-" * 40)
                continue

            # ── STEP 4: Perturb ───────────────────────────────────────────
            # Now returns decisions_log too for spot check report
            pert_text, changes_log, decisions_log = perturber.perturb(orig_text)
            all_decisions.append((hadm_id, decisions_log))  # collect for spot check

            if changes_log:
                print(f"   > Attack Status : {len(changes_log)} words changed.")
                for orig_word, new_word in changes_log:
                    print(f"     * '{orig_word}' -> '{new_word}'")
            else:
                print("   > Attack Status : FAILED (No valid synonyms found).")

            # ── STEP 5: Perturbed prediction ──────────────────────────────
            pert_icd = get_icd_prediction(pert_text)
            print(f"   > Perturbed ICD : {pert_icd}")

            # ── STEP 6: Robustness check ──────────────────────────────────
            icd_drifted = orig_icd.upper().strip() != pert_icd.upper().strip()
            print(f"   > ICD Drift     : "
                  f"{'❌ YES — prediction changed' if icd_drifted else '✅ NO — prediction stable'}")

            # ── STEP 7: Text similarity ───────────────────────────────────
            sim = scorer.get_sim(orig_text, pert_text)
            print(f"   > Text Sim      : {sim:.4f}")

            # ── STEP 8: Flag suspicious drifts ────────────────────────────
            suspicious = (icd_drifted and sim > 0.995 and len(changes_log) <= 3)
            if suspicious:
                print(f"   > ⚠️  Suspicious drift flagged")

            changes_str = " | ".join([f"{o} -> {p}" for o, p in changes_log])

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
                "baseline_correct"    : True,
                "perturbed_icd_pred"  : pert_icd,
                "icd_codes_differ"    : icd_drifted,
                "suspicious_drift"    : suspicious,
                "semantic_similarity" : sim
            })

            print("-" * 40)

        except Exception as e:
            debug_print("ERROR", f"Row {idx} failed: {e}")
            continue

    if results:
        final_df = pd.DataFrame(results)

        # Drop text columns from CSV — they live in HTML report
        csv_df = final_df.drop(
            columns=['original_text', 'perturbed_text'], errors='ignore')
        csv_df.to_csv(FINAL_RESULTS, index=False)

        plot_results(final_df)
        generate_html_report(final_df)

        # NEW: Generate spot check report for paper
        generate_spot_check_report(all_decisions)

        # Summary on valid samples only
        valid_df = final_df[final_df['baseline_correct'] == True]
        skipped  = len(final_df) - len(valid_df)

        print("\n" + "=" * 80)
        print(f"📊 SUMMARY")
        print(f"   Total samples           : {len(final_df)}")
        print(f"   Skipped (wrong baseline): {skipped}")
        print(f"   Valid for analysis      : {len(valid_df)}")
        if len(valid_df) > 0:
            drift_rate    = valid_df['icd_codes_differ'].mean() * 100
            avg_sim       = valid_df['semantic_similarity'].mean()
            avg_changes   = valid_df['changes_count'].mean()
            suspicious_n  = valid_df['suspicious_drift'].sum()
            print(f"   ICD drift rate          : {drift_rate:.1f}%")
            print(f"   Avg text similarity     : {avg_sim:.4f}")
            print(f"   Avg words changed       : {avg_changes:.1f}")
            print(f"   Suspicious drifts       : {suspicious_n}")
        print(f"   Results saved to        : {FINAL_RESULTS}")
        print(f"   Spot check saved to     : {SPOT_CHECK_REPORT}")
        print("=" * 80)
        debug_print("MAIN", "Done.")

if __name__ == "__main__":
    main()