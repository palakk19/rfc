# import pandas as pd
# import re

# class MIMICDataLoader:
#     def __init__(self, notes_path, diagnosis_path, max_samples=100):
#         self.notes_path = notes_path
#         self.diagnosis_path = diagnosis_path
#         self.max_samples = max_samples

#     def load_data(self):
#         print("Loading MIMIC-IV Data... this may take a while.")
        
#         # Load Discharge Summaries (Notes)
#         # We only need subject_id, hadm_id, and text
#         notes_df = pd.read_csv(self.notes_path, compression='gzip', usecols=['subject_id', 'hadm_id', 'text'])
        
#         # Load Diagnoses (Labels)
#         diag_df = pd.read_csv(self.diagnosis_path, compression='gzip')
        
#         # Filter for top priority diagnoses (seq_num = 1 usually indicates primary diagnosis)
#         primary_diag = diag_df[diag_df['seq_num'] == 1]
        
#         # Merge Notes with Labels on admission ID (hadm_id)
#         merged = pd.merge(notes_df, primary_diag[['hadm_id', 'icd_code', 'icd_version']], on='hadm_id', how='inner')
        
#         # basic cleaning to remove PHI markers like [** ... **]
#         merged['text'] = merged['text'].apply(self._clean_text)
        
#         # Sample for robustness study to save compute time
#         sampled_data = merged.sample(n=min(len(merged), self.max_samples), random_state=42)
        
#         return sampled_data

#     def _clean_text(self, text):
#         # Remove MIMIC de-identification brackets
#         text = re.sub(r'\[\*\*.*?\*\*\]', ' ', text)
#         # Remove excessive whitespace
#         text = re.sub(r'\s+', ' ', text).strip()
#         return text

# # Usage Example
# if __name__ == "__main__":
#     # Ensure you point to your actual .csv.gz files
#     loader = MIMICDataLoader('discharge.csv.gz', 'diagnoses_icd.csv.gz')
#     df = loader.load_data()
#     print(df.head())