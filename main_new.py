import torch
import pandas as pd
import re, random, spacy, requests, json, os, warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
MODEL_NAME = "llama3.1:8b"
API_URL = "http://ollama.warhol.informatik.rwth-aachen.de/api/generate"
NOTES_PATH = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz"
DIAGNOSES_PATH = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/diagnoses_icd.csv.gz"
DICT_PATH = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/d_icd_diagnoses.csv.gz"
FINAL_RESULTS = "robustness_analysis_detailed.csv"
DEVICE = "cpu"
warnings.filterwarnings("ignore")

# --- REAL-TIME VERBOSE LOGGER ---
def log_step(hadm_id, stage, message, color="\033[94m"):
    print(f"{color}[HADM: {hadm_id}] [{stage}]\033[0m {message}")

# --- RELEVANT SECTION EXTRACTOR ---
def extract_clinical_narrative(text):
    """Isolates Chief Complaint and History of Present Illness using Regex."""
    relevant_text = ""
    # Look for common MIMIC headers
    patterns = [
        r"(Chief Complaint:.*?)(?=[A-Z]{2,}:|$)",
        r"(History of Present Illness:.*?)(?=[A-Z]{2,}:|$)"
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            relevant_text += match.group(1).strip() + " "
    
    # Fallback if specific headers aren't found
    return relevant_text[:2000] if len(relevant_text) > 50 else text[:1500]

# --- ADVERSARIAL LLM CONNECTOR ---
def query_llm(prompt):
    session = requests.Session()
    session.trust_env = False
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}}
    try:
        response = session.post(API_URL, json=payload, timeout=60)
        return response.json().get('response', '').strip()
    except: return "ERROR"

# --- MAIN ENGINE ---
def run_robustness_test():
    log_step("SYSTEM", "INIT", "Loading medical models and data...")
    nlp = spacy.load("en_core_sci_sm")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(DEVICE)
    
    # Load and merge data
    notes = pd.read_csv(NOTES_PATH, compression='gzip', nrows=10)
    diags = pd.read_csv(DIAGNOSES_PATH, compression='gzip')
    icd_dict = pd.read_csv(DICT_PATH, compression='gzip')
    primary = diags[diags['seq_num'] == 1]
    df = notes.merge(primary, on='hadm_id').merge(icd_dict, on=['icd_code', 'icd_version'])

    results = []
    print("\n" + "="*80 + "\nSTARTING REAL-TIME ANALYSIS\n" + "="*80)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        hid = row['hadm_id']
        clinical_text = extract_clinical_narrative(row['text'])
        ground_truth = row['long_title']

        # 1. Baseline Accuracy Check
        orig_pred = query_llm(f"Identify the primary diagnosis in 3 words from this note: {clinical_text}")
        log_step(hid, "BASELINE", f"GT: {ground_truth} | AI: {orig_pred}", "\033[92m")

        # 2. ScispaCy Entity Targeting & Perturbation
        doc = nlp(clinical_text)
        entities = [ent.text for ent in doc.ents if len(ent.text) > 5][:2] # Targeted attack on 2 key terms
        perturbed_text = clinical_text
        changes = []

        for ent in entities:
            synonym = query_llm(f"Provide one clinical synonym for '{ent}':")
            if synonym and synonym.lower() != ent.lower() and len(synonym.split()) < 4:
                perturbed_text = perturbed_text.replace(ent, synonym)
                changes.append(f"{ent}->{synonym}")

        if not changes:
            log_step(hid, "ATTACK", "Failed (No synonyms found)", "\033[93m")
            continue

        log_step(hid, "ATTACK", f"Changed: {', '.join(changes)}", "\033[95m")

        # 3. Robustness Re-Evaluation
        pert_pred = query_llm(f"Identify the primary diagnosis in 3 words from this note: {perturbed_text}")
        
        # 4. Semantic Similarity (Bio_ClinicalBERT)
        inputs = tokenizer([clinical_text, perturbed_text], padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            emb = bert_model(**inputs).last_hidden_state[:, 0, :]
        sim = float(cosine_similarity(emb[0:1], emb[1:2])[0][0])
        
        drift = 1 if orig_pred.lower() != pert_pred.lower() else 0
        log_step(hid, "RESULT", f"Sim: {sim:.4f} | Drift: {'YES' if drift else 'NO'}", "\033[96m")

        results.append({
            "hadm_id": hid, "ground_truth": ground_truth, "original_input": clinical_text,
            "perturbed_input": perturbed_text, "original_pred": orig_pred, "perturbed_pred": pert_pred,
            "similarity": sim, "drift": drift, "perturbations": "|".join(changes)
        })

    # Saving Results
    final_df = pd.DataFrame(results)
    final_df.to_csv(FINAL_RESULTS, index=False)
    generate_summary_plots(final_df)

def generate_summary_plots(df):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df['similarity'], kde=True, color='teal')
    plt.title("Distribution of Semantic Similarity")
    
    plt.subplot(1, 2, 2)
    drift_pct = df['drift'].mean() * 100
    sns.barplot(x=["Stable", "Drifted"], y=[100-drift_pct, drift_pct], palette="viridis")
    plt.title(f"Prediction Stability (Drift Rate: {drift_pct:.1f}%)")
    plt.savefig("robustness_summary.png")
    print(f"\n[DONE] Plots saved. Detailed CSV at {FINAL_RESULTS}")

if __name__ == "__main__":
    run_robustness_test()