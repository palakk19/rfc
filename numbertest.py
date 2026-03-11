import pandas as pd

# ── Paths — change these to match your machine ────────────────────────────
DIAGNOSES_PATH = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/diagnoses_icd.csv.gz"
NOTES_PATH     = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz"

# ── Load diagnoses ─────────────────────────────────────────────────────────
print("Loading diagnoses file...")
diags = pd.read_csv(DIAGNOSES_PATH, compression='gzip')

print("\n" + "="*50)
print("DIAGNOSES FILE — RAW OVERVIEW")
print("="*50)
print(f"Total rows (all diagnoses):       {len(diags):,}")
print(f"Unique patients (subject_id):     {diags['subject_id'].nunique():,}")
print(f"Unique admissions (hadm_id):      {diags['hadm_id'].nunique():,}")
print(f"Columns: {list(diags.columns)}")

# ── Primary diagnoses only (seq_num == 1) ─────────────────────────────────
primary = diags[diags['seq_num'] == 1]

print("\n" + "="*50)
print("PRIMARY DIAGNOSES ONLY (seq_num = 1)")
print("="*50)
print(f"Total primary diagnoses:          {len(primary):,}")
print(f"Unique patients:                  {primary['subject_id'].nunique():,}")
print(f"Unique admissions:                {primary['hadm_id'].nunique():,}")

# ── ICD version breakdown ──────────────────────────────────────────────────
version_counts = primary['icd_version'].value_counts().sort_index()

print("\n" + "="*50)
print("ICD VERSION BREAKDOWN (primary diagnoses)")
print("="*50)
for version, count in version_counts.items():
    pct = 100 * count / len(primary)
    print(f"  ICD-{version}: {count:,} admissions  ({pct:.1f}%)")

total = len(primary)
icd9_count  = version_counts.get(9, 0)
icd10_count = version_counts.get(10, 0)

print(f"\n  Total:  {total:,} admissions")
print(f"  ICD-9:  {icd9_count:,}  ({100*icd9_count/total:.1f}%)")
print(f"  ICD-10: {icd10_count:,}  ({100*icd10_count/total:.1f}%)")

# ── How many notes exist ───────────────────────────────────────────────────
print("\nLoading notes file (just counting rows, not full load)...")
notes = pd.read_csv(NOTES_PATH, compression='gzip',
                    usecols=['subject_id', 'hadm_id'])

print("\n" + "="*50)
print("NOTES FILE OVERVIEW")
print("="*50)
print(f"Total discharge notes:            {len(notes):,}")
print(f"Unique patients:                  {notes['subject_id'].nunique():,}")
print(f"Unique admissions:                {notes['hadm_id'].nunique():,}")

# ── How many notes have a matching primary ICD-10 diagnosis ───────────────
notes_with_icd10 = pd.merge(
    notes,
    primary[primary['icd_version'] == 10][['hadm_id']],
    on='hadm_id'
)
notes_with_icd9 = pd.merge(
    notes,
    primary[primary['icd_version'] == 9][['hadm_id']],
    on='hadm_id'
)

print("\n" + "="*50)
print("NOTES MATCHED WITH PRIMARY DIAGNOSIS")
print("="*50)
print(f"Notes with ICD-10 primary dx:     {len(notes_with_icd10):,}")
print(f"Notes with ICD-9  primary dx:     {len(notes_with_icd9):,}")
print(f"\nIf you filter to ICD-10 only:")
print(f"  Available for pipeline:         {len(notes_with_icd10):,} notes")
print(f"  Expected valid after ~76% pass: {int(len(notes_with_icd10)*0.76):,} notes")

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"  Total unique patients in MIMIC: {diags['subject_id'].nunique():,}")
print(f"  Total admissions with notes:    {notes['hadm_id'].nunique():,}")
print(f"  ICD-10 admissions (with notes): {len(notes_with_icd10):,}")
print(f"  ICD-9  admissions (with notes): {len(notes_with_icd9):,}")
print("="*50)