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
import requests # Added for API
import json     # Added for API
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
# ✅ UPDATED: Using the Model and URL from your successful screenshot
MODEL_NAME = "llama3.1:8b" 
API_URL = "http://ollama.warhol.informatik.rwth-aachen.de/api/generate"

# PATHS (Keep your existing paths)
NOTES_PATH = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz"
DIAGNOSES_PATH = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/diagnoses_icd.csv.gz"
DICT_PATH = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/d_icd_diagnoses.csv.gz" 

FINAL_RESULTS = "final_robustness_api.csv"
MAX_SAMPLES = 8  # Set low for testing
DEVICE = "cpu"   # ✅ UPDATED: Forced CPU

warnings.filterwarnings("ignore")

def debug_print(stage, message):
    print(f"\033[94m[{stage}]\033[0m {message}")

# --- HELPER: API CONNECTOR ---
def query_rwth_server(prompt):
    """Sends the prompt to the University Server and returns the text."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,       
        "options": {
            "temperature": 0.1, 
            "num_predict": 50  # Limit output length
        }
    }
    
    # ✅ FIX: Create a session that ignores system proxies (fixes VPN issues)
    session = requests.Session()
    session.trust_env = False 
    
    try:
        response = session.post(API_URL, headers=headers, data=json.dumps(payload), timeout=60)
        response.raise_for_status()
        return response.json().get('response', '').strip()
    except requests.exceptions.RequestException as e:
        debug_print("API_ERROR", f"Connection failed: {e}")
        return ""

class MIMICLoader:
    @staticmethod
    def clean_noise(text):
        """Removes de-identification masks and artifacts."""
        text = str(text)
        # Remove [** ... **] and ___
        text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)
        text = re.sub(r"_{2,}", " ", text)
        # Remove specific MIMIC formatting noise to help RegEx find headers
        text = text.replace("=", " ").replace("*", " ").replace("#", " ")
        return " ".join(text.split())

    @staticmethod
    def extract_blind_clinical_info(text):
        """
        Extracts HPI and Hospital Course but EXCLUDES 
        Discharge Diagnosis and Family History to prevent ground truth leakage.
        """
        relevant_parts = []
        
        # 1. History of Present Illness (The 'Story')
        # Stops before PMH or Social History to avoid 'Lupus/Family' hallucinations
        hpi = re.search(r"History of Present Illness:(.*?)(Past Medical History:|Social History:)", text, re.S | re.I)
        if hpi:
            relevant_parts.append(f"HPI: {hpi.group(1).strip()}")
        
        # 2. Brief Hospital Course (The 'Action')
        # Stops before Medications or Discharge sections to hide the 'Answer'
        course = re.search(r"Brief Hospital Course:(.*?)(Medications on Admission:|Discharge Medications:|Discharge Diagnosis:)", text, re.S | re.I)
        if course:
            relevant_parts.append(f"Hospital Course: {course.group(1).strip()}")

        if relevant_parts:
            return "\n\n".join(relevant_parts)
        
        # Fallback: If headers are missing, take the first 1500 chars 
        # (Usually contains the HPI but stops before the bottom-page diagnosis)
        return text[:1500]

    @classmethod
    def load(cls, notes_path, diag_path, dict_path, n_samples):
        debug_print("DATA", "Loading Data Tables...")
        
        # 1. Load the raw dataframes
        notes = pd.read_csv(notes_path, compression='gzip', usecols=['subject_id', 'hadm_id', 'text'])
        diags = pd.read_csv(diag_path, compression='gzip')
        
        # 2. Load and merge ICD descriptions
        if os.path.exists(dict_path):
            icd_dict = pd.read_csv(dict_path, compression='gzip')
            diags = pd.merge(diags, icd_dict[['icd_code', 'icd_version', 'long_title']], 
                             on=['icd_code', 'icd_version'], how='left')
        else:
            diags['long_title'] = "Unknown Description"

        # 3. Filter for Primary Diagnoses (seq_num 1)
        primary_diags = diags[diags['seq_num'] == 1]
        
        debug_print("DATA", "Merging Notes with Diagnoses...")
        # Create the main dataframe by merging notes with primary diagnosis
        df = pd.merge(notes, primary_diags[['hadm_id', 'icd_code', 'long_title']], on='hadm_id')
        
        # 4. Process the text: Clean first, then Extract
        debug_print("DATA", "Cleaning and Extracting Blind Clinical Info...")
        df['text'] = df['text'].apply(cls.clean_noise)
        df['processed_text'] = df['text'].apply(cls.extract_blind_clinical_info)
        
        # 5. Return the sample
        return df.sample(n=min(len(df), n_samples), random_state=42)
    
    
# --- 2. GENERATIVE ATTACKER (VIA API) ---

class GenerativePerturber:
    def __init__(self):
        try:
            # Using the medium model for better dependency parsing
            self.nlp = spacy.load("en_core_sci_md")
        except OSError:
            self.nlp = spacy.load("en_core_sci_sm")

    def is_negated(self, token):
        """Checks if a word is part of a negative finding."""
        negation_terms = {'no', 'not', 'denies', 'negative', 'without', 'none', 'denied'}
        # Check previous 3 words for negation
        window = [t.text.lower() for t in list(token.doc)[max(0, token.i - 3):token.i]]
        return any(term in window for term in negation_terms)

    def get_generative_synonym(self, word, context_sentence):
        # Determine if the word is an acronym (all caps or short mixed case)
        is_acronym = word.isupper() and len(word) <= 5
        
        prompt = (
            f"Context: \"{context_sentence}\"\n"
            f"Term: '{word}'\n\n"
            f"Instructions:\n"
            f"1. Replace '{word}' with one medically equivalent synonym.\n"
            f"2. CATEGORY RULE: A fistula is a surgical/pathological connection; a 'sinus tract' is NOT a synonym. Maintain exact pathology.\n"
            f"3. ACRONYM RULE: If the term is an abbreviation (like CTA, VTX, PE), ONLY provide the full formal expansion. If you are unsure of the expansion in this context, return the original term '{word}'.\n"
            f"4. Respond with ONLY the replacement text.\n"
            f"Answer:"
        )
        
        synonym = query_rwth_server(prompt)
        
        if synonym:
            synonym = synonym.replace('"', '').replace('.', '').strip()
            
            # --- NEW VALIDATION FILTERS ---
            # 1. Reject if length ratio is suspicious (e.g., word is 3 chars, synonym is 30)
            if len(word) > 0 and (len(synonym) / len(word)) > 5:
                return None
            
            # 2. Hallucination check for common acronym errors
            if is_acronym and len(synonym.split()) == 1 and synonym.lower() != word.lower():
                # If an acronym was replaced by a single different word, it's likely a hallucination
                # (e.g., CTA -> Tachypnea). Real expansions should be multiple words.
                return None
                
            if not synonym or synonym.lower() == word.lower():
                return None
                
        return synonym

    def perturb(self, text, perturbation_rate=0.10):
        doc = self.nlp(text)
        # Filter: Only perturb entities that are NOT negated
        valid_entities = [ent for ent in doc.ents if not self.is_negated(ent[0])]
        
        if not valid_entities:
            return text, []

        num_to_perturb = max(1, int(len(valid_entities) * perturbation_rate))
        target_entities = random.sample(valid_entities, min(num_to_perturb, len(valid_entities)))
        
        # Sort reverse to maintain string indices
        target_entities.sort(key=lambda x: x.start_char, reverse=True)
        
        perturbed_text = text
        changes_log = [] 
        
        for ent in target_entities:
            # Send surrounding context (current sentence + surrounding text)
            context_window = ent.sent.text if ent.sent else text[max(0, ent.start_char-50):ent.end_char+50]
            
            replacement = self.get_generative_synonym(ent.text, context_window)
            
            if replacement and replacement.lower() != ent.text.lower():
                # Final check: Don't let it replace a single word with something wildly different length-wise
                if len(replacement) > (len(ent.text) * 4): 
                    continue
                    
                perturbed_text = perturbed_text[:ent.start_char] + replacement + perturbed_text[ent.end_char:]
                changes_log.append((ent.text, replacement))
                
        return perturbed_text, changes_log
# --- 3. EVALUATOR (CPU Friendly) ---
class SemanticEvaluator:
    def __init__(self):
        debug_print("EVAL_INIT", "Loading ClinicalBERT (CPU)...")
        # Load small BERT model locally on CPU
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(DEVICE)
        self.model.eval()

    def get_sim(self, text1, text2):
        with torch.no_grad():
            inputs = self.tokenizer([text1, text2], padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            outputs = self.model(**inputs)
            e1 = outputs.last_hidden_state[0, 0, :].unsqueeze(0)
            e2 = outputs.last_hidden_state[1, 0, :].unsqueeze(0)
            sim = cosine_similarity(e1.numpy(), e2.numpy())[0][0]
        return sim

# --- 4. PREDICTION (VIA API) ---
def get_api_prediction(text):
    prompt = (
        f"Read the clinical note below. Identify the primary diagnosis. "
        f"Output ONLY the disease name. Do not write complete sentences.\n\n"
        f"Note:\n{text}\n\nDiagnosis:"
    )
    response = query_rwth_server(prompt)
    return response.strip()

# --- 5. VISUALIZATION ---
def plot_results(df):
    debug_print("PLOT", "Generating plots...")
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(df['semantic_similarity'], kde=True, bins=15, color='teal')
    plt.title("Semantic Preservation")
    
    df['clean_orig'] = df['original_pred'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
    df['clean_pert'] = df['perturbed_pred'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
    df['drift'] = df['clean_orig'] != df['clean_pert']
    drift_rate = df['drift'].mean() * 100

    plt.subplot(1, 2, 2)
    sns.countplot(x='drift', data=df, palette='viridis')
    plt.title(f"Prediction Stability (Drift: {drift_rate:.1f}%)")
    plt.savefig("robustness_api_plots.png")

# --- MAIN ---
def main():
    debug_print("MAIN", f"Starting Run on {DEVICE} with API Model: {MODEL_NAME}")
    
    # 1. Test API connection
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

    print("\n" + "="*80)
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            hadm_id = row['hadm_id']
            orig_text = row['processed_text']
            ground_truth_desc = row['long_title']
            
            print(f"\n[ID: {hadm_id}] Processing...")
            print(f"   > Ground Truth: {ground_truth_desc}")

            # 1. Original Prediction (API)
            orig_pred = get_api_prediction(orig_text)
            print(f"   > Original Pred : {orig_pred}")
            
            # 2. Generative Attack (API)
            pert_text, changes_log = perturber.perturb(orig_text)
            
            if len(changes_log) > 0:
                print(f"   > Attack Status : {len(changes_log)} words changed.")
                for orig_word, new_word in changes_log:
                    print(f"     * '{orig_word}' -> '{new_word}'")
            else:
                print("   > Attack Status : FAILED (No synonyms found).")

            # 3. Perturbed Prediction (API)
            pert_pred = get_api_prediction(pert_text)
            print(f"   > Perturbed Pred: {pert_pred}")
            
            # 4. Eval (Local CPU)
            sim = scorer.get_sim(orig_text, pert_text)
            print(f"   > Similarity    : {sim:.4f}")

            results.append({
                "hadm_id": hadm_id,
                "ground_truth": ground_truth_desc,
                "original_pred": orig_pred,
                "perturbed_pred": pert_pred,
                "changes_count": len(changes_log),
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
        debug_print("MAIN", f"Done. Results saved to {FINAL_RESULTS}")

if __name__ == "__main__":
    main()