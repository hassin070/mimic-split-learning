import os
import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import LabelEncoder


def load_table(data_dir, filename):
    """
    Helper function to load compressed CSV safely.
    """
    path = os.path.join(data_dir, filename)
    print(f"Loading {filename}...")
    if os.path.exists(path):
        return pd.read_csv(path, compression='gzip')
    else:
        print(f"Warning: {filename} not found at {path}.")
        return None


def build_client_dataset(input_folder, output_file):
    """
    Loads raw MIMIC-IV tables from input_folder, performs preprocessing, feature engineering,
    and saves the resulting patient-level feature vectors to output_file.

    Features extracted (up to ~35 depending on available tables):
        Demographic  : age, gender_encoded
        Admission    : admission_type_encoded, icu_admitted, los
        Severity     : comorbidity_count, lab_test_count
        Laboratory   : mean / max / std for top-10 lab items (30 columns)
    Target: hospital_expire_flag
    """
    print(f"\n{'='*60}")
    print(f"Processing data from: {input_folder}")
    print(f"{'='*60}\n")

    # ── 1. Load core tables ──────────────────────────────────────────────────
    patients_df   = load_table(input_folder, 'patients.csv.gz')
    admissions_df = load_table(input_folder, 'admissions.csv.gz')
    icustays_df   = load_table(input_folder, 'icustays.csv.gz')

    # Diagnoses — for comorbidity count
    diagnoses_df = load_table(input_folder, 'diagnoses_icd.csv.gz')

    # Labevents — find the shard file (may have suffix like 'labevents.csv-003.gz')
    labevents_filename = None
    for f in os.listdir(input_folder):
        if f.startswith('labevents') and f.endswith('.gz'):
            labevents_filename = f
            break

    if patients_df is None or admissions_df is None:
        raise ValueError("Critical tables (patients, admissions) missing.")

    # ── 2. Merge Admissions and Patients ────────────────────────────────────
    print("\nMerging Admissions and Patients...")
    full_df = pd.merge(admissions_df, patients_df, on='subject_id', how='inner')

    # ── 3. Calculate Age ─────────────────────────────────────────────────────
    # age = anchor_age + (admittime_year - anchor_year)
    full_df['admittime']      = pd.to_datetime(full_df['admittime'])
    full_df['admission_year'] = full_df['admittime'].dt.year
    full_df['age']            = full_df['anchor_age'] + (full_df['admission_year'] - full_df['anchor_year'])

    # ── 4. Admission Type Encoding ───────────────────────────────────────────
    # FIX: admission_type was previously discarded.
    # Emergency vs elective is one of the strongest mortality predictors.
    print("Encoding admission type...")
    if 'admission_type' in full_df.columns:
        full_df['admission_type'] = full_df['admission_type'].fillna('UNKNOWN')
        le_admit = LabelEncoder()
        full_df['admission_type_encoded'] = le_admit.fit_transform(
            full_df['admission_type'].astype(str)
        )
        print(f"  Admission types found: {full_df['admission_type'].unique().tolist()}")
    else:
        print("  Warning: admission_type column not found. Defaulting to 0.")
        full_df['admission_type_encoded'] = 0

    # ── 5. ICU Stays — LOS + Admission Flag ─────────────────────────────────
    if icustays_df is not None:
        print("Merging ICU Stays...")
        agg_dict = {'los': 'sum'}
        if 'first_careunit' in icustays_df.columns:
            agg_dict['first_careunit'] = 'first'

        icu_features = icustays_df.groupby('hadm_id').agg(agg_dict).reset_index()
        full_df      = pd.merge(full_df, icu_features, on='hadm_id', how='left')
        full_df['los'] = full_df['los'].fillna(0)
    else:
        full_df['los'] = 0

    # FIX: Add binary ICU admission flag.
    # Complements los — distinguishes "never in ICU" from "very brief ICU stay".
    full_df['icu_admitted'] = (full_df['los'] > 0).astype(int)
    print(f"  ICU admitted: {full_df['icu_admitted'].sum():,} / {len(full_df):,} admissions")

    # ── 6. Comorbidity Count ─────────────────────────────────────────────────
    # FIX: Number of distinct ICD diagnosis codes per admission.
    # Disease burden is a well-validated mortality proxy in MIMIC literature.
    if diagnoses_df is not None:
        print("Computing comorbidity count from diagnoses_icd...")
        comorbidity = (
            diagnoses_df
            .groupby('hadm_id')['icd_code']
            .nunique()
            .reset_index()
        )
        comorbidity.columns = ['hadm_id', 'comorbidity_count']
        full_df = pd.merge(full_df, comorbidity, on='hadm_id', how='left')
        full_df['comorbidity_count'] = full_df['comorbidity_count'].fillna(0)
        print(f"  Comorbidity count — mean: {full_df['comorbidity_count'].mean():.1f}, "
              f"max: {full_df['comorbidity_count'].max():.0f}")
    else:
        print("  Warning: diagnoses_icd.csv.gz not found. comorbidity_count set to 0.")
        full_df['comorbidity_count'] = 0

    # ── 7. Lab Features ──────────────────────────────────────────────────────
    # FIX 1: Added mean / max / std per lab item (was mean only).
    #         max captures peak severity; std captures instability/volatility.
    # FIX 2: Added lab_test_count — care intensity proxy.
    if labevents_filename:
        print(f"\nProcessing Lab Events from {labevents_filename} in chunks...")
        lab_path   = os.path.join(input_folder, labevents_filename)
        chunk_size = 100_000

        # ── Pass 1: Identify top-10 most frequent lab items ─────────────────
        item_counts = pd.Series(dtype=int)
        try:
            for i, chunk in enumerate(
                pd.read_csv(lab_path, compression='gzip',
                            chunksize=chunk_size, usecols=['itemid'])
            ):
                if i > 5:
                    break   # sample first ~500k rows for speed
                item_counts = item_counts.add(
                    chunk['itemid'].value_counts(), fill_value=0
                )

            top_labs = item_counts.sort_values(ascending=False).head(10).index.tolist()
            print(f"  Top 10 Lab Items (from sample): {top_labs}")

            # ── Pass 2: Extract relevant items and compute aggregates ────────
            relevant_chunks = []
            for chunk in pd.read_csv(
                lab_path, compression='gzip', chunksize=chunk_size,
                usecols=['hadm_id', 'itemid', 'valuenum']
            ):
                filtered = chunk[chunk['itemid'].isin(top_labs)]
                if not filtered.empty:
                    relevant_chunks.append(filtered)

            if relevant_chunks:
                lab_subset = pd.concat(relevant_chunks, ignore_index=True)

                # FIX: mean + max + std instead of mean only
                # max  → peak severity (e.g. creatinine spike)
                # std  → volatility / instability signal
                lab_agg = (
                    lab_subset
                    .groupby(['hadm_id', 'itemid'])['valuenum']
                    .agg(['mean', 'max', 'std'])
                    .unstack()
                )
                lab_agg.columns = [
                    f'lab_{stat}_{item}'
                    for stat, item in lab_agg.columns
                ]
                lab_agg = lab_agg.fillna(0)
                full_df = pd.merge(full_df, lab_agg, on='hadm_id', how='left')
                print(f"  Lab feature columns added: {lab_agg.shape[1]} "
                      f"(mean/max/std × {len(top_labs)} items)")

                # FIX: Lab test count — number of distinct tests ordered
                # More distinct tests → more clinical concern → higher severity
                lab_test_count = (
                    lab_subset
                    .groupby('hadm_id')['itemid']
                    .nunique()
                    .reset_index()
                )
                lab_test_count.columns = ['hadm_id', 'lab_test_count']
                full_df = pd.merge(full_df, lab_test_count, on='hadm_id', how='left')
                full_df['lab_test_count'] = full_df['lab_test_count'].fillna(0)
                print(f"  lab_test_count added — mean: {full_df['lab_test_count'].mean():.1f}")

            else:
                print("  No matching lab events found.")
                full_df['lab_test_count'] = 0

        except Exception as e:
            print(f"  Error processing labevents: {e}. Skipping lab features.")
            full_df['lab_test_count'] = 0

    else:
        print("  No labevents file found. Skipping lab features.")
        full_df['lab_test_count'] = 0

    # ── 8. Handle Missing Values & Encode Gender ─────────────────────────────
    print("\nFinalizing features...")

    numeric_cols = full_df.select_dtypes(include=[np.number]).columns
    full_df[numeric_cols] = full_df[numeric_cols].fillna(0)

    if 'gender' in full_df.columns:
        full_df['gender'] = full_df['gender'].fillna('Unknown')
        le_gender = LabelEncoder()
        full_df['gender_encoded'] = le_gender.fit_transform(
            full_df['gender'].astype(str)
        )
    else:
        full_df['gender_encoded'] = 0

    # ── 9. Select Final Feature Columns ─────────────────────────────────────
    # Original 13 features:
    #   age, gender_encoded, los + 10 mean lab values
    #
    # Updated ~35 features:
    #   age, gender_encoded                        — demographic
    #   admission_type_encoded                     — NEW: emergency vs elective
    #   los, icu_admitted                          — ICU (icu_admitted is NEW)
    #   comorbidity_count                          — NEW: disease burden
    #   lab_test_count                             — NEW: care intensity
    #   lab_mean_*/lab_max_*/lab_std_* (30 cols)  — lab (max/std are NEW)

    feature_cols = (
        ['age', 'gender_encoded', 'admission_type_encoded',
         'los', 'icu_admitted', 'comorbidity_count', 'lab_test_count']
        + [c for c in full_df.columns if c.startswith('lab_')]
    )

    # Target
    if 'hospital_expire_flag' in full_df.columns:
        target_col = 'hospital_expire_flag'
    else:
        print("Warning: 'hospital_expire_flag' not found. Creating synthetic target.")
        full_df['hospital_expire_flag'] = np.random.randint(0, 2, size=len(full_df))
        target_col = 'hospital_expire_flag'

    cols_to_save = ['subject_id', 'hadm_id'] + feature_cols + [target_col]

    # Keep only columns that actually exist (guards against missing optional tables)
    cols_to_save = [c for c in cols_to_save if c in full_df.columns]
    final_df     = full_df[cols_to_save]

    # ── 10. Save ─────────────────────────────────────────────────────────────
    print(f"\nSaving processed data to {output_file}...")
    print(f"Final shape: {final_df.shape}")
    print(f"Features included ({len(feature_cols)}):")
    for col in feature_cols:
        if col in final_df.columns:
            print(f"  + {col}")

    mortality_rate = final_df[target_col].mean() * 100
    print(f"\nClass distribution:")
    print(f"  Survived (0): {(final_df[target_col]==0).sum():,}")
    print(f"  Expired  (1): {(final_df[target_col]==1).sum():,}  ({mortality_rate:.1f}%)")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if output_file.endswith('.parquet'):
        final_df.to_parquet(output_file, index=False)
    else:
        final_df.to_csv(output_file, index=False)

    print(f"\nDone. Saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess Client Data for Federated Learning"
    )
    parser.add_argument(
        "--input_folder", type=str, required=True,
        help="Path to folder containing raw CSV.GZ files"
    )
    parser.add_argument(
        "--output_file", type=str, required=True,
        help="Path to save the processed CSV or Parquet file"
    )

    args = parser.parse_args()
    build_client_dataset(args.input_folder, args.output_file)
