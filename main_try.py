import torch
import pandas as pd
import numpy as np
import os
import re
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

# PATHS (Removed Dictionary Path)
NOTES_PATH = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz"
DIAGNOSES_PATH = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/diagnoses_icd.csv.gz"

FINAL_RESULTS = "final_results_api.csv"
MAX_SAMPLES = 10

# --- DEVICE SELECTION ---
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"✅ Hardware Detected: GPU ({torch.cuda.get_device_name(0)})")
else:
    DEVICE = "cpu"
    print("⚠️ Hardware Detected: CPU (Running evaluation locally)")

warnings.filterwarnings("ignore")

# --- HELPER: API CONNECTOR ---
# --- HELPER: API CONNECTOR (UPDATED) ---
# --- HELPER: API CONNECTOR (Use this version) ---
def query_rwth_server(prompt, model_name):
    """Sends the prompt to the University Server and returns the text."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,       
        "options": {
            "temperature": 0.1, 
            "num_predict": 200  
        }
    }
    
    # [FIX] Create a session that ignores system proxies
    session = requests.Session()
    session.trust_env = False  # <--- THIS DISABLES PROXY INTERFERENCE
    
    try:
        response = session.post(API_URL, headers=headers, data=json.dumps(payload), timeout=60)
        response.raise_for_status()
        return response.json().get('response', '')
    except requests.exceptions.RequestException as e:
        print(f"\n❌ API Error: {e}")
        return ""
# --- 1. DATA LOADER (NO DICTIONARY) ---
class MIMICLoader:
    @staticmethod
    def clean_mimic_text(text):
        """Helper to clean clinical notes."""
        if pd.isna(text): return ""
        text = str(text).replace("[**", "").replace("**]", "")
        text = re.sub(r"\n+", " ", text)
        return " ".join(text.split())

    @staticmethod
    def load(notes_path, diag_path, n_samples):
        print("DATA: Loading Data Tables...")
        
        # 1. Load Notes
        if not os.path.exists(notes_path): raise FileNotFoundError(f"File missing: {notes_path}")
        notes = pd.read_csv(notes_path, compression='gzip', usecols=['subject_id', 'hadm_id', 'text'])

        # 2. Load Diagnoses
        if not os.path.exists(diag_path): raise FileNotFoundError(f"File missing: {diag_path}")
        diags = pd.read_csv(diag_path, compression='gzip')
        
        # 3. Filter for Primary Diagnosis
        primary_diags = diags[diags['seq_num'] == 1].copy()
        
        # 4. MANUAL OVERRIDE: Create 'long_title' from the Code itself
        # This prevents the "key error" because we are creating the column right here.
        primary_diags['long_title'] = "ICD Code: " + primary_diags['icd_code'].astype(str)
        
        print("DATA: Merging Notes with Diagnoses...")
        # Merge
        df = pd.merge(notes, primary_diags[['hadm_id', 'icd_code', 'long_title']], on='hadm_id')
        
        # Clean text
        df['text'] = df['text'].apply(MIMICLoader.clean_mimic_text)
        df['processed_text'] = df['text'].apply(lambda x: x[:3000])
        
        print(f"DATA: Successfully loaded {len(df)} rows.")
        return df.sample(n=min(len(df), n_samples), random_state=42)

# --- 2. GENERATIVE ATTACKER ---
class GenerativePerturber:
    def __init__(self):
        print("ATTACK_INIT: Loading SciSpacy...")
        try:
            self.nlp = spacy.load("en_core_sci_sm")
        except OSError:
            raise OSError("Model 'en_core_sci_sm' not found. Run: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz")
            
    def get_generative_synonym(self, word, context_sentence):
        prompt = (
            f"You are a medical editor. Replace the term '{word}' with a valid clinical synonym "
            f"that fits the context below. \n"
            f"Context: \"{context_sentence}\"\n\n"
            f"Rules:\n1. Output ONLY the new word.\n2. Do not change meaning.\n3. Do not output original word.\n"
            f"Answer:"
        )
        
        response = query_rwth_server(prompt, MODEL_NAME)
        synonym = response.strip().replace('"', '').replace('.', '')
        
        if not synonym or synonym.lower() == word.lower() or len(synonym.split()) > 3:
            return None
        return synonym

    def perturb(self, text, perturbation_rate=0.10):
        doc = self.nlp(text)
        entities = list(doc.ents)
        if not entities: return text, []

        num_to_perturb = min(10, max(1, int(len(entities) * perturbation_rate)))
        target_entities = random.sample(entities, min(num_to_perturb, len(entities)))
        target_entities.sort(key=lambda x: x.start_char, reverse=True)
        
        perturbed_text = text
        changes_log = [] 
        
        for ent in target_entities:
            context_sent = ent.sent.text if ent.sent else text[:200]
            replacement = self.get_generative_synonym(ent.text, context_sent)
            
            if replacement:
                perturbed_text = perturbed_text[:ent.start_char] + replacement + perturbed_text[ent.end_char:]
                changes_log.append((ent.text, replacement))
                
        return perturbed_text, changes_log

# --- 3. EVALUATOR ---
class SemanticEvaluator:
    def __init__(self):
        print("EVAL_INIT: Loading ClinicalBERT (Local)...")
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(DEVICE)
        self.model.eval()

    def get_sim(self, text1, text2):
        with torch.no_grad():
            inputs = self.tokenizer([text1, text2], padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            outputs = self.model(**inputs)
            e1 = outputs.last_hidden_state[0, 0, :].unsqueeze(0)
            e2 = outputs.last_hidden_state[1, 0, :].unsqueeze(0)
            sim = cosine_similarity(e1.cpu().numpy(), e2.cpu().numpy())[0][0]
        return sim

# --- 4. PREDICTION ---
def get_prediction(text):
    prompt = f"Read the note below. Identify the primary diagnosis. Output ONLY the disease name.\n\nNote:\n{text}\n\nDiagnosis:"
    return query_rwth_server(prompt, MODEL_NAME).strip()

# --- MAIN ---
def main():
    print(f"MAIN: Starting Run using API Model: {MODEL_NAME}")
    
    # 1. Test API connection
    print("MAIN: Testing API connection...")
    test_resp = query_rwth_server("Say 'hello'", MODEL_NAME)
    if not test_resp:
        print("❌ CRITICAL: Cannot connect to RWTH server. Check VPN or URL.")
        return
    print("✅ API Connection Successful.")

    try:
        # NOTE: Removed dict_path argument from call
        df = MIMICLoader.load(NOTES_PATH, DIAGNOSES_PATH, MAX_SAMPLES)
    except Exception as e:
        print(f"ERROR: Data load failed: {e}")
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
            
            orig_pred = get_prediction(orig_text)
            print(f"   > Original Pred : {orig_pred}")
            
            pert_text, changes_log = perturber.perturb(orig_text)
            print(f"   > Attack Status : {len(changes_log)} words changed.")

            pert_pred = get_prediction(pert_text)
            print(f"   > Perturbed Pred: {pert_pred}")
            
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
            print(f"ERROR: Row {idx} failed: {e}")
            continue

    if results:
        final_df = pd.DataFrame(results)
        final_df.to_csv(FINAL_RESULTS, index=False)
        print(f"MAIN: Done. Results saved to {FINAL_RESULTS}")

if __name__ == "__main__":
    main()