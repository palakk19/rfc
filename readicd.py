import pandas as pd

# Pandas detects the compression automatically
df = pd.read_csv(r"C:\Users\palak\rfc\dataset\physionet.org\files\mimic-iv-note\2.2\note\d_icd_diagnoses.csv.gz")
df.to_csv('output_file.csv', index=False)


# 2. Get unique values and their counts for a specific column
# Replace 'Column_Name' with the actual name of your column
counts = df['icd_code'].value_counts()

print(counts)