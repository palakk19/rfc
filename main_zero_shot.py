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
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

# UPDATE PATHS
BASE_PATH = "/home/kulkarni/projects/palakrfc/dataset/physionet.org/files"
NOTES_PATH = f"{BASE_PATH}/mimic-iv-note/2.2/note/discharge.csv.gz"
DIAGNOSES_PATH = f"{BASE_PATH}/mimic-iv-note/2.2/note/diagnoses_icd.csv.gz" 
DICT_PATH = f"{BASE_PATH}/mimic-iv-note/2.2/note/d_icd_diagnoses.csv.gz" 

FINAL_RESULTS = "final_robustness_generative.csv"
MAX_SAMPLES = 6
DEVICE = "cuda"

warnings.filterwarnings("ignore")

def debug_print(stage, message):
    print(f"\033[94m[{stage}]\033[0m {message}")

# --- 1. DATA LOADER ---
# --- 1. DATA LOADER (UPDATED) ---
class MIMICLoader:
    @staticmethod
    def load(notes_path, diag_path, dict_path, n_samples):
        debug_print("DATA", "Loading Data Tables...")
        if not os.path.exists(notes_path) or not os.path.exists(diag_path):
            raise FileNotFoundError(f"Files missing. Checked: {notes_path}")

        notes = pd.read_csv(notes_path, compression='gzip', usecols=['subject_id', 'hadm_id', 'text'])
        diags = pd.read_csv(diag_path, compression='gzip')
        
        if os.path.exists(dict_path):
            icd_dict = pd.read_csv(dict_path, compression='gzip')
            diags = pd.merge(diags, icd_dict[['icd_code', 'icd_version', 'long_title']], 
                             on=['icd_code', 'icd_version'], how='left')
        else:
            diags['long_title'] = "Unknown Description"

        primary_diags = diags[diags['seq_num'] == 1]
        
        debug_print("DATA", "Merging Notes with Diagnoses...")
        df = pd.merge(notes, primary_diags[['hadm_id', 'icd_code', 'long_title']], on='hadm_id')
        
        def clean_mimic_text(text):
            # 1. Remove brackets
            text = text.replace("[**", "").replace("**]", "")
            
            # 2. Remove Headers BUT KEEP 'Sex:'
            # I removed r"Sex:.*?\n" from this list so it stays in the text.
            headers_to_remove = [
                r"Name:.*?\n", 
                r"Unit No:.*?\n", 
                r"Admission Date:.*?\n", 
                r"Discharge Date:.*?\n", 
                r"Date of Birth:.*?\n",
                r"Service:.*?\n"  # Removed Service to clean it up further
            ]
            for h in headers_to_remove:
                text = re.sub(h, " ", text)
            
            # 3. Collapse whitespace
            return " ".join(text.split())

        df['text'] = df['text'].apply(clean_mimic_text)
        
        # Truncate: First 3000 chars (Symptoms)
        df['processed_text'] = df['text'].apply(lambda x: x[:3000])
        
        return df.sample(n=min(len(df), n_samples), random_state=42)

# --- 2. GENERATIVE ATTACKER (Using Phi-3) ---
class GenerativePerturber:
    def __init__(self):
        debug_print("ATTACK_INIT", "Loading SciSpacy for target identification...")
        try:
            self.nlp = spacy.load("en_core_sci_sm")
        except OSError:
            raise OSError("Please install scispacy model.")
            
    def get_generative_synonym(self, model, tokenizer, word, context_sentence):
        """Asks the LLM to provide a medical synonym."""
        # Strict prompt to force a single word output
        prompt = (
            f"<|user|>\n"
            f"You are a medical editor. Replace the term '{word}' with a valid clinical synonym "
            f"that fits the context below. \n"
            f"Context: \"{context_sentence}\"\n\n"
            f"Rules:\n"
            f"1. Output ONLY the new word or phrase.\n"
            f"2. Do not change the meaning.\n"
            f"3. Do not output the original word.\n"
            f"<|end|>\n<|assistant|>"
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, use_cache=False)
            
        new_tokens = outputs[0][inputs.input_ids.shape[-1]:]
        synonym = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        # Cleanup: Remove extra quotes or periods
        synonym = synonym.replace('"', '').replace('.', '').strip()
        
        # Validation: Don't return the same word or empty strings
        if not synonym or synonym.lower() == word.lower() or len(synonym.split()) > 3:
            return None
            
        return synonym

    def perturb(self, text, model, tokenizer, perturbation_rate=0.10):
        doc = self.nlp(text)
        entities = list(doc.ents)
        if not entities: 
            return text, []

        # Limit to 5 perturbations max to save time (LLM generation is slower than BERT)
        num_to_perturb = min(10, max(1, int(len(entities) * perturbation_rate)))
        
        target_entities = random.sample(entities, min(num_to_perturb, len(entities)))
        target_entities.sort(key=lambda x: x.start_char, reverse=True)
        
        perturbed_text = text
        changes_log = [] 
        
        for ent in target_entities:
            # Get the sentence context for better suggestions
            context_sent = ent.sent.text if ent.sent else text[:200]
            
            # Ask Phi-3 for a synonym
            replacement = self.get_generative_synonym(model, tokenizer, ent.text, context_sent)
            
            if replacement:
                perturbed_text = perturbed_text[:ent.start_char] + replacement + perturbed_text[ent.end_char:]
                changes_log.append((ent.text, replacement))
                
        return perturbed_text, changes_log

# --- 3. EVALUATOR ---
class SemanticEvaluator:
    def __init__(self):
        debug_print("EVAL_INIT", "Loading ClinicalBERT...")
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
def clean_llm_output(text):
    text = text.strip()
    patterns = [r"^Based on the.*?diagnosis is", r"^The most likely diagnosis is", 
                r"^The patient has", r"^Diagnosis:", r"^Assessment:", r"^Most likely diagnosis:"]
    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE).strip()
    return text.strip(".").strip()

def get_phi3_prediction(model, tokenizer, text):
    prompt = f"<|user|>\nRead the note below. Identify the primary diagnosis. Output ONLY the disease name. Do not write complete sentences.\n\nNote:\n{text}<|end|>\n<|assistant|>"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, use_cache=False)
    
    new_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return clean_llm_output(response)

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
    plt.savefig("robustness_gpu_plots.png")

# --- MAIN ---
def main():
    debug_print("MAIN", f"Starting Run on {torch.cuda.get_device_name(0)}...")
    
    try:
        df = MIMICLoader.load(NOTES_PATH, DIAGNOSES_PATH, DICT_PATH, MAX_SAMPLES)
    except Exception as e:
        debug_print("ERROR", f"Data load failed: {e}")
        return

    debug_print("MAIN", "Loading Phi-3 (Used for Attack AND Defense)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            device_map="cuda",            
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="eager" 
        )
    except Exception as e:
         debug_print("ERROR", f"Model load failed: {e}")
         return

    perturber = GenerativePerturber() # Updated Class
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

            # 1. Original Prediction
            orig_pred = get_phi3_prediction(model, tokenizer, orig_text)
            print(f"   > Original Pred : {orig_pred}")
            
            # 2. Generative Attack (Pass model/tokenizer here)
            pert_text, changes_log = perturber.perturb(orig_text, model, tokenizer)
            
            if len(changes_log) > 0:
                print(f"   > Attack Status : {len(changes_log)} words changed.")
                print("   > Changes Made  :")
                for orig_word, new_word in changes_log:
                    print(f"     * '{orig_word}' -> '{new_word}'")
            else:
                print("   > Attack Status : FAILED (No suitable synonyms generated).")

            # 3. Perturbed Prediction
            pert_pred = get_phi3_prediction(model, tokenizer, pert_text)
            print(f"   > Perturbed Pred: {pert_pred}")
            
            # 4. Eval
            sim = scorer.get_sim(orig_text, pert_text)
            print(f"   > Similarity    : {sim:.4f}")

            results.append({
                "hadm_id": hadm_id,
                "ground_truth": ground_truth_desc,
                "original_pred": orig_pred,
                "perturbed_pred": pert_pred,
                "changes_count": len(changes_log),
                "changes_list": str(changes_log),
                "semantic_similarity": sim,
                "original_text": orig_text,
                "perturbed_text": pert_text
            })
            
            print("-" * 40)
            if idx % 5 == 0: torch.cuda.empty_cache()

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