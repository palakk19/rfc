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
import requests # <--- NEW IMPORT
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel  # Removed AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
# IMPORTANT: Replace this with the EXACT name you found in the /api/tags link
MODEL_NAME = "llama3.1:8b" 

# API ENDPOINT (From your screenshot context)
# The documentation link in your image suggests an Ollama setup.
API_URL = "http://warhol.informatik.rwth-aachen.de:11434/api/generate"

# PATHS (Keep your existing paths)
NOTES_PATH = "C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz"
DIAGNOSES_PATH = "C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/diagnoses_icd.csv.gz"
DICT_PATH = "C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/d_icd_diagnoses.csv.gz"

FINAL_RESULTS = "final_results_api.csv"
MAX_SAMPLES = 2

# --- DEVICE SELECTION (Only for the Evaluator now) ---
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"✅ Hardware Detected: GPU ({torch.cuda.get_device_name(0)}) - Used for ClinicalBERT evaluation")
else:
    DEVICE = "cpu"
    print("⚠️ Hardware Detected: CPU - Used for ClinicalBERT evaluation")

warnings.filterwarnings("ignore")

# --- HELPER: API CONNECTOR ---
def query_rwth_server(prompt, model_name):
    """Sends the prompt to the University Server and returns the text."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,       # We want the full response at once
        "options": {
            "temperature": 0.1, # Low temp for consistent medical answers
            "num_predict": 200  # Max tokens to generate
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Check for errors
        return response.json()['response']
    except requests.exceptions.RequestException as e:
        print(f"\n❌ API Error: {e}")
        return ""

# --- 1. DATA LOADER (Unchanged) ---
class MIMICLoader:
    @staticmethod
    def load(notes_path, diag_path, dict_path, n_samples):
        print("DATA: Loading Data Tables...")
        if not os.path.exists(notes_path): raise FileNotFoundError(f"File missing: {notes_path}")
        
        notes = pd.read_csv(notes_path, compression='gzip', usecols=['subject_id', 'hadm_id', 'text'])
        diags = pd.read_csv(diag_path, compression='gzip')
        
        if os.path.exists(dict_path):
            icd_dict = pd.read_csv(dict_path, compression='gzip')
            diags = pd.merge(diags, icd_dict[['icd_code', 'icd_version', 'long_title']], 
                             on=['icd_code', 'icd_version'], how='left')
        
        primary_diags = diags[diags['seq_num'] == 1]
        df = pd.merge(notes, primary_diags[['hadm_id', 'icd_code', 'long_title']], on='hadm_id')
        
        def clean_mimic_text(text):
            text = text.replace("[**", "").replace("**]", "")
            text = re.sub(r"\n+", " ", text) # Simple cleanup
            return " ".join(text.split())

        df['text'] = df['text'].apply(clean_mimic_text)
        df['processed_text'] = df['text'].apply(lambda x: x[:3000])
        return df.sample(n=min(len(df), n_samples), random_state=42)

# --- 2. GENERATIVE ATTACKER (Modified for API) ---
class GenerativePerturber:
    def __init__(self):
        print("ATTACK_INIT: Loading SciSpacy...")
        try:
            self.nlp = spacy.load("en_core_sci_sm")
        except OSError:
            raise OSError("Please install scispacy model: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz")
            
    def get_generative_synonym(self, word, context_sentence):
        prompt = (
            f"You are a medical editor. Replace the term '{word}' with a valid clinical synonym "
            f"that fits the context below. \n"
            f"Context: \"{context_sentence}\"\n\n"
            f"Rules:\n1. Output ONLY the new word.\n2. Do not change meaning.\n3. Do not output original word.\n"
            f"Answer:"
        )
        
        # CALL API INSTEAD OF LOCAL MODEL
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

# --- 3. EVALUATOR (Unchanged - Keeps ClinicalBERT Local) ---
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

# --- 4. PREDICTION (Modified for API) ---
def get_prediction(text):
    prompt = f"Read the note below. Identify the primary diagnosis. Output ONLY the disease name.\n\nNote:\n{text}\n\nDiagnosis:"
    
    # CALL API
    response = query_rwth_server(prompt, MODEL_NAME)
    return response.strip()

# --- MAIN ---
def main():
    print(f"MAIN: Starting Run using API Model: {MODEL_NAME}")
    
    try:
        df = MIMICLoader.load(NOTES_PATH, DIAGNOSES_PATH, DICT_PATH, MAX_SAMPLES)
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

            # 1. Original Prediction (API)
            orig_pred = get_prediction(orig_text)
            print(f"   > Original Pred : {orig_pred}")
            
            # 2. Generative Attack (API)
            pert_text, changes_log = perturber.perturb(orig_text)
            print(f"   > Attack Status : {len(changes_log)} words changed.")

            # 3. Perturbed Prediction (API)
            pert_pred = get_prediction(pert_text)
            print(f"   > Perturbed Pred: {pert_pred}")
            
            # 4. Eval (Local)
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