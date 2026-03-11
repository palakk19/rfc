import pandas as pd

# Configuration (Ensure these paths match your local setup)
NOTES_PATH = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz"
DIAGNOSES_PATH = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/diagnoses_icd.csv.gz"
DICT_PATH = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/d_icd_diagnoses.csv.gz"

# 1. Load the data
# We load 100 notes to find a good match, and all ICD codes/definitions
notes = pd.read_csv(NOTES_PATH, compression='gzip', nrows=100)
diags = pd.read_csv(DIAGNOSES_PATH, compression='gzip')
icd_dict = pd.read_csv(DICT_PATH, compression='gzip')

# Select the very first row (index 0) and print just the 'text' column
first_note = notes.iloc[30]['text']

print("--- FULL TEXT OF THE FIRST NOTE ---")
print(first_note)
# # 2. Preparation: Filter for Primary Diagnosis (seq_num 1)
# primary_diags = diags[diags['seq_num'] == 1]

# # 3. Merge files to get Text, Code, and Label in one place
# # Merge notes with their primary ICD code
# df = pd.merge(notes, primary_diags[['hadm_id', 'icd_code', 'icd_version']], on='hadm_id')

# # Merge with dictionary to get the 'long_title'
# df = pd.merge(df, icd_dict[['icd_code', 'icd_version', 'long_title']], on=['icd_code', 'icd_version'])

# # 4. Comparison Logic
# def compare_diagnoses(row):
#     print(f"\n{'='*60}")
#     print(f"HADM_ID: {row['hadm_id']}")
#     print(f"OFFICIAL ICD LABEL (Ground Truth): {row['long_title']}")
#     print(f"{'-'*60}")
    
#     text = row['text']
#     # Look for the specific header in the doctor's note
#     if "Discharge Diagnosis:" in text:
#         # Split at the header and take the text immediately following it
#         parts = text.split("Discharge Diagnosis:")
#         # We take the first 300 characters of the diagnosis section
#         doctor_written = parts[1].split("\n\n")[0].strip() 
#         print(f"DOCTOR'S WRITTEN DIAGNOSIS IN TEXT:\n{doctor_written}")
#     else:
#         # If the header isn't there, print a snippet of the end of the note
#         print("Header 'Discharge Diagnosis:' not found. Printing end of note:")
#         print(text[-500:])

# # Run the comparison for the first few records
# for i in range(min(3, len(df))):
#     compare_diagnoses(df.iloc[i])