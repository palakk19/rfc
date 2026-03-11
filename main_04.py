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

FINAL_RESULTS = "final_robustness_api.csv"
MAX_SAMPLES = 8
DEVICE = "cpu"

warnings.filterwarnings("ignore")

def debug_print(stage, message):
    print(f"\033[94m[{stage}]\033[0m {message}")

# ============================================================
# SECTION 1: API CONNECTOR — No changes here
# ============================================================
def query_rwth_server(prompt):
    """Sends the prompt to the University Server and returns the text."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 50
        }
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
# SECTION 2: DATA LOADER — Only change: capture icd_version
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
        # ✅ CHANGE: Added icd_version to the merge so we can store it in results
        # This tells us whether the ground truth code is ICD-9 or ICD-10
        df = pd.merge(notes,
                      primary_diags[['hadm_id', 'icd_code', 'icd_version', 'long_title']],
                      on='hadm_id')

        debug_print("DATA", "Cleaning and Extracting Blind Clinical Info...")
        df['text'] = df['text'].apply(cls.clean_noise)
        df['processed_text'] = df['text'].apply(cls.extract_blind_clinical_info)

        return df.sample(n=min(len(df), n_samples), random_state=42)


# ============================================================
# SECTION 3: PERTURBATION VALIDATOR — Entirely new class
# ============================================================
# WHY: Your perturber generates synonyms but never checks if they
# actually preserve clinical meaning. This class asks the LLM to
# verify its own synonym before it gets accepted. If the LLM says
# the replacement changes the meaning, the word is NOT swapped.
class PerturbationValidator:
    @staticmethod
    def validate_synonym(original_word, replacement, context_sentence):
        
        # Fast rule-based pre-checks before wasting an API call
        artifact_patterns = ['->', '→', '\n', '**', '##']
        for artifact in artifact_patterns:
            if artifact in replacement:
                return False
        
        if len(replacement.split()) > len(original_word.split()) + 5:
            return False

        prompt = (
            f"You are a medical terminology expert. Decide if a word replacement "
            f"is safe in a clinical note.\n\n"
            f"A replacement is SAFE if:\n"
            f"- It is a direct synonym (fever = pyrexia, vomiting = emesis)\n"
            f"- It is a brand/generic drug swap (Tylenol = Acetaminophen)\n"
            f"- It is a Latin/English equivalent (kidney = renal, liver = hepatic)\n"
            f"- Minor phrasing difference (patient = individual, elevated = increased)\n\n"
            f"A replacement is UNSAFE if:\n"
            f"- It changes the specific body part (small intestine -> jejunum)\n"
            f"- It changes the severity (pain -> severe pain)\n"
            f"- It changes the type of condition (acidosis -> acidemia are different)\n"
            f"- It adds information not in the original (negative -> normal)\n\n"
            f"Examples of SAFE: fever->pyrexia, vomiting->emesis, "
            f"white blood cell->leukocyte, Tylenol->Acetaminophen, "
            f"shortness of breath->dyspnea, fatigue->lethargy\n"
            f"Examples of UNSAFE: small intestine->jejunum, "
            f"pain->severe pain, negative->normal\n\n"
            f"Context: \"{context_sentence}\"\n"
            f"Original: '{original_word}'\n"
            f"Replacement: '{replacement}'\n\n"
            f"Is this replacement SAFE? Answer only SAFE or UNSAFE."
        )
        
        response = query_rwth_server(prompt)
        return "SAFE" in response.strip().upper()


# ============================================================
# SECTION 4: GENERATIVE PERTURBER
# Only change: added validator call before accepting a synonym
# ============================================================
class GenerativePerturber:
    def __init__(self):
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
            f"Context: \"{context_sentence}\"\n"
            f"Term: '{word}'\n\n"
            f"Instructions:\n"
            f"1. Replace '{word}' with one medically equivalent synonym.\n"
            f"2. CATEGORY RULE: A fistula is a surgical/pathological connection; "
            f"a 'sinus tract' is NOT a synonym. Maintain exact pathology.\n"
            f"3. ACRONYM RULE: If the term is an abbreviation (like CTA, VTX, PE), "
            f"ONLY provide the full formal expansion. If unsure, return '{word}'.\n"
            f"4. Respond with ONLY the replacement text.\n"
            f"Answer:"
        )
        synonym = query_rwth_server(prompt)

        if synonym:
            synonym = synonym.replace('"', '').replace('.', '').strip()
            if len(word) > 0 and (len(synonym) / len(word)) > 5:
                return None
            if is_acronym and len(synonym.split()) == 1 and \
               synonym.lower() != word.lower():
                return None
            if not synonym or synonym.lower() == word.lower():
                return None

        return synonym

    def perturb(self, text, perturbation_rate=0.10):
        doc = self.nlp(text)
        valid_entities = [ent for ent in doc.ents if not self.is_negated(ent[0])]

        if not valid_entities:
            return text, []

        num_to_perturb = max(1, int(len(valid_entities) * perturbation_rate))
        target_entities = random.sample(
            valid_entities, min(num_to_perturb, len(valid_entities)))
        target_entities.sort(key=lambda x: x.start_char, reverse=True)

        perturbed_text = text
        changes_log = []

        for ent in target_entities:
            context_window = ent.sent.text if ent.sent else \
                text[max(0, ent.start_char - 50):ent.end_char + 50]

            replacement = self.get_generative_synonym(ent.text, context_window)

            if replacement and replacement.lower() != ent.text.lower():
                if len(replacement) > (len(ent.text) * 4):
                    continue

                if '->' in replacement or '→' in replacement or '\n' in replacement:
                    print(f"     ⚠️  REJECTED: '{ent.text}' -> '{replacement}' (formatting artifact)")
                    continue

                # ✅ CHANGE: Validate before accepting
                # Without this, bad synonyms like "fistula -> sinus tract"
                # would silently corrupt your perturbations
                is_valid = PerturbationValidator.validate_synonym(
                    ent.text, replacement, context_window)

                if not is_valid:
                    print(f"     ⚠️  REJECTED: '{ent.text}' -> '{replacement}' "
                          f"(meaning changed per validator)")
                    continue

                perturbed_text = (perturbed_text[:ent.start_char]
                                  + replacement
                                  + perturbed_text[ent.end_char:])
                changes_log.append((ent.text, replacement))

        return perturbed_text, changes_log


# ============================================================
# SECTION 5: SEMANTIC EVALUATOR — No changes
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
        return sim


# ============================================================
# SECTION 6: ICD PREDICTION — Replaces get_api_prediction()
# ============================================================
# WHY: Free text like "Pneumonia" vs "lung infection" are the same
# disease but string comparison treats them as different — giving
# false positives for drift. ICD-10 codes like J18.9 are standardized
# so comparison is unambiguous. The regex at the end strips any
# extra text the LLM adds and keeps only the code pattern.
def get_icd_prediction(text):
    """
    Asks the LLM to return a standardized ICD-10 code.
    Replaces get_api_prediction() which returned free text descriptions.
    """
    prompt = (
        f"Read the clinical note below. Return ONLY the single most likely "
        f"ICD-10 diagnosis code (example format: J18.9 or I21.0). "
        f"Do not write the disease name. Do not explain. Output the code only.\n\n"
        f"Note:\n{text}\n\nICD-10 Code:"
    )
    response = query_rwth_server(prompt)

    # ✅ Clean the response: extract just the ICD code pattern
    # e.g. if LLM says "The code is J18.9 for pneumonia" → we extract "J18.9"
    match = re.search(r'[A-Z]\d{2}\.?\d*', response.strip().upper())
    return match.group(0) if match else response.strip()


# ============================================================
# SECTION 7: VISUALIZATION — Small fix for new column names
# ============================================================
def plot_results(df):
    debug_print("PLOT", "Generating plots...")
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(df['semantic_similarity'], kde=True, bins=15, color='teal')
    plt.title("Semantic Preservation After Perturbation")
    plt.xlabel("Cosine Similarity")

    # ✅ CHANGE: Use icd_codes_differ column directly instead of
    # recomputing from free-text predictions (old approach was fragile)
    drift_rate = df['icd_codes_differ'].mean() * 100

    plt.subplot(1, 2, 2)
    sns.countplot(x='icd_codes_differ', data=df, palette='viridis')
    plt.title(f"ICD Prediction Stability\n(Drift Rate: {drift_rate:.1f}%)")
    plt.xlabel("ICD Codes Differ After Perturbation")
    plt.xticks([0, 1], ['Same (Robust)', 'Different (Drifted)'])

    plt.tight_layout()
    plt.savefig("robustness_api_plots.png")
    print("✅ Plot saved to robustness_api_plots.png")

# ============================================================
# SECTION 8.5: HTML REPORT — New addition
# Generates a readable side-by-side view of every case
# ============================================================
def generate_html_report(df, filename="robustness_report.html"):
    rows_html = ""
    for _, row in df.iterrows():
        drift_color = "#ffcccc" if row['icd_codes_differ'] else "#ccffcc"
        drift_label = "❌ DRIFTED" if row['icd_codes_differ'] else "✅ STABLE"
        
        rows_html += f"""
        <div class="case" style="border:1px solid #ccc; margin:20px 0; padding:15px; border-radius:8px;">
            <h3>Patient ID: {row['hadm_id']} 
                <span style="background:{drift_color}; padding:4px 10px; 
                border-radius:4px; font-size:0.9em;">{drift_label}</span>
            </h3>
            <p><b>Ground Truth:</b> {row['ground_truth_desc']} 
               ({row['ground_truth_icd']} v{row['ground_truth_version']})</p>
            <p><b>Changes Made:</b> {row['changes_made'] if row['changes_made'] else 'None'}</p>
            <p><b>Semantic Similarity:</b> {row['semantic_similarity']:.4f}</p>
            
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
                    font-size:0.85em;">{row['original_text']}</td>
                    <td style="padding:10px; border:1px solid #ccc; 
                    vertical-align:top; white-space:pre-wrap; 
                    font-size:0.85em;">{row['perturbed_text']}</td>
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
        </div>
        """
    
    drift_rate = df['icd_codes_differ'].mean() * 100
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Robustness Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1400px; 
               margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f5f5f5; padding: 15px; 
                   border-radius: 8px; margin-bottom: 30px; }}
    </style>
</head>
<body>
    <h1>🏥 Clinical Note Robustness Report</h1>
    <div class="summary">
        <h2>Summary</h2>
        <p><b>Total Samples:</b> {len(df)}</p>
        <p><b>ICD Drift Rate:</b> {drift_rate:.1f}%</p>
        <p><b>Avg Semantic Similarity:</b> {df['semantic_similarity'].mean():.4f}</p>
        <p><b>Avg Changes Per Note:</b> {df['changes_count'].mean():.1f}</p>
    </div>
    {rows_html}
</body>
</html>"""
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ HTML report saved to {filename} — open in any browser")




# ============================================================
# SECTION 8: MAIN LOOP — Updated results dict and comparisons
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

    perturber = GenerativePerturber()
    scorer = SemanticEvaluator()
    results = []

    print("\n" + "=" * 80)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            hadm_id    = row['hadm_id']
            orig_text  = row['processed_text']
            gt_desc    = row['long_title']
            gt_icd     = row['icd_code']
            icd_ver    = row['icd_version']   # ✅ NEW: ICD-9 vs ICD-10 flag

            print(f"\n[ID: {hadm_id}] Processing...")
            print(f"   > Ground Truth  : {gt_desc} ({gt_icd}, v{icd_ver})")

            # Step 1: Get ICD prediction for original text
            orig_icd = get_icd_prediction(orig_text)
            print(f"   > Original ICD  : {orig_icd}")

            # Step 2: Perturb the text
            pert_text, changes_log = perturber.perturb(orig_text)

            if changes_log:
                print(f"   > Attack Status : {len(changes_log)} words changed.")
                for orig_word, new_word in changes_log:
                    print(f"     * '{orig_word}' -> '{new_word}'")
            else:
                print("   > Attack Status : FAILED (No valid synonyms found).")

            # Step 3: Get ICD prediction for perturbed text
            pert_icd = get_icd_prediction(pert_text)
            print(f"   > Perturbed ICD : {pert_icd}")

            # Step 4: Compare — this is your core research metric
            # ✅ NEW: Simple, unambiguous boolean. No fuzzy string matching needed.
            icd_drifted = orig_icd.upper().strip() != pert_icd.upper().strip()
            print(f"   > ICD Drift     : {'❌ YES — prediction changed' if icd_drifted else '✅ NO — prediction stable'}")

            # Step 5: Semantic similarity of the texts themselves
            sim = scorer.get_sim(orig_text, pert_text)
            print(f"   > Similarity    : {sim:.4f}")

            # ✅ CHANGE: Format changes as a readable string for CSV
            # e.g. "fever -> pyrexia | edema -> oedema"
            changes_str = " | ".join([f"{o} -> {p}" for o, p in changes_log])

            results.append({
                "hadm_id"            : hadm_id,
                # Ground truth info
                "ground_truth_icd"   : gt_icd,
                "ground_truth_version": icd_ver,        # ✅ NEW
                "ground_truth_desc"  : gt_desc,
                # The actual texts
                "original_text"      : orig_text,       # ✅ NEW
                "perturbed_text"     : pert_text,       # ✅ NEW
                # What the perturber changed
                "changes_made"       : changes_str,     # ✅ NEW
                "changes_count"      : len(changes_log),
                # LLM predictions
                "original_icd_pred"  : orig_icd,        # ✅ RENAMED
                "perturbed_icd_pred" : pert_icd,        # ✅ RENAMED
                # Core metric
                "icd_codes_differ"   : icd_drifted,     # ✅ NEW
                # Text similarity
                "semantic_similarity": sim
            })

            print("-" * 40)

        except Exception as e:
            debug_print("ERROR", f"Row {idx} failed: {e}")
            continue

    if results:
        final_df = pd.DataFrame(results)
        final_df.to_csv(FINAL_RESULTS, index=False)
        plot_results(final_df)
        generate_html_report(final_df)

        # ✅ NEW: Print a quick summary at the end
        drift_rate = final_df['icd_codes_differ'].mean() * 100
        avg_sim    = final_df['semantic_similarity'].mean()
        avg_changes = final_df['changes_count'].mean()
        print("\n" + "=" * 80)
        print(f"📊 SUMMARY")
        print(f"   Samples processed : {len(final_df)}")
        print(f"   Avg words changed : {avg_changes:.1f} per note")
        print(f"   ICD drift rate    : {drift_rate:.1f}%")
        print(f"   Avg text similarity: {avg_sim:.4f}")
        print(f"   Results saved to  : {FINAL_RESULTS}")
        print("=" * 80)

        debug_print("MAIN", "Done.")

if __name__ == "__main__":
    main()