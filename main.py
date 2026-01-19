import pandas as pd
from data_loader import MIMICDataLoader
from attacker import MedicalPerturber
from evaluator import RobustnessEvaluator
from tqdm import tqdm

# --- MOCK VICTIM MODEL ---
# Replace this class with your actual Llama-3 / Model inference code
class DummyVictimLLM:
    def predict(self, text):
        # Returns a fake ICD code for demonstration
        return "428.0" 

def main():
    # 1. Load Data
    # Update paths to where you saved your MIMIC files
    data_loader = MIMICDataLoader(
        notes_path='/home/kulkarni/projects/palakrfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz', 
        diagnosis_path='/home/kulkarni/projects/palakrfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/diagnoses_icd.csv.gz',
        max_samples=10 # Keep small for testing
    )
    df = data_loader.load_data()
    print(f"Loaded {len(df)} samples.")

    # 2. Initialize Components
    perturber = MedicalPerturber()
    evaluator = RobustnessEvaluator()
    victim_model = DummyVictimLLM()

    results = []

    # 3. Execution Loop
    print("Starting robustness evaluation...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        original_text = row['text']
        true_label = row['icd_code']
        
        # A. Original Prediction
        orig_pred = victim_model.predict(original_text)
        
        # B. Attack (Perturbation)
        try:
            perturbed_text = perturber.perturb(original_text)
        except Exception as e:
            print(f"Error perturbing index {index}: {e}")
            perturbed_text = original_text # Fallback

        # C. Perturbed Prediction
        pert_pred = victim_model.predict(perturbed_text)
        
        results.append({
            "original_text": original_text,
            "perturbed_text": perturbed_text,
            "original_pred": orig_pred,
            "perturbed_pred": pert_pred
        })

    # 4. Evaluation
    orig_texts = [r['original_text'] for r in results]
    pert_texts = [r['perturbed_text'] for r in results]
    orig_preds = [r['original_pred'] for r in results]
    pert_preds = [r['perturbed_pred'] for r in results]

    metrics = evaluator.calculate_metrics(orig_texts, pert_texts, orig_preds, pert_preds)
    
    print("\n=== Robustness Metrics ===")
    print(f"Mean Semantic Preservation (ClinicalBERT): {metrics['mean_semantic_similarity']:.4f}")
    print(f"Prediction Drift Rate: {metrics['prediction_drift_rate']:.4f}")

    # Save details to CSV for analysis
    pd.DataFrame(results).to_csv("robustness_results.csv", index=False)
    print("Results saved to robustness_results.csv")

if __name__ == "__main__":
    main()
