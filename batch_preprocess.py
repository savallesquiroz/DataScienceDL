# batch_preprocess.py
import os
import mne
import numpy as np

# -------------------------
# 0. Paths
# -------------------------
RAW_DIR = "data/raw"
PROC_DIR = "data/processed"
FEATURES_DIR = "data/features"

os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)

# -------------------------
# 1. Subjects
# -------------------------
subjects = [f"A0{i}T" for i in range(1, 10)]  # A01T … A09T

# -------------------------
# 2. Preprocessing Function
# -------------------------
def preprocess_subject(subj_id: str):
    print(f"\n=== Processing {subj_id} ===")
    raw_path = os.path.join(RAW_DIR, f"{subj_id}.gdf")

    if not os.path.exists(raw_path):
        print(f"Raw file for {subj_id} not found, skipping...")
        return False

    # --- Load raw ---
    raw = mne.io.read_raw_gdf(raw_path, preload=True)

    # --- Set channel types ---
    eeg_chs = raw.ch_names[:22]
    eog_chs = raw.ch_names[22:]
    raw.set_channel_types({ch: "eeg" for ch in eeg_chs})
    raw.set_channel_types({ch: "eog" for ch in eog_chs})

    # --- Montage ---
    montage = mne.channels.make_standard_montage("standard_1020")
    try:
        raw.set_montage(montage)
    except Exception as e:
        print("Montage issue:", e)
        # fallback: allow missing channels
        raw.set_montage(montage, on_missing="ignore")

    # --- Reference ---
    raw.set_eeg_reference("average", projection=True)

    # --- Filtering ---
    raw.filter(l_freq=1., h_freq=40.)

    # --- ICA for EOG artifacts ---
    ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter="auto")
    ica.fit(raw)
    eog_inds, _ = ica.find_bads_eog(raw)
    ica.exclude = eog_inds
    raw_clean = ica.apply(raw.copy())

    # Save cleaned raw
    clean_path = os.path.join(PROC_DIR, f"{subj_id}_clean.fif")
    raw_clean.save(clean_path, overwrite=True)

    # --- Events ---
    events, event_id_map = mne.events_from_annotations(raw_clean)
    motor_codes = ["769", "770", "771", "772"]
    event_ids = {
        name: event_id_map[code]
        for name, code in zip(["left_hand", "right_hand", "foot", "tongue"], motor_codes)
        if code in event_id_map
    }

    if not event_ids:
        print(f"No motor imagery events found for {subj_id}, skipping…")
        return False

    # --- Epoching ---
    tmin, tmax = 0.0, 4.0
    reject_criteria = dict(eeg=150e-6, eog=250e-6)

    epochs = mne.Epochs(
        raw_clean,
        events,
        event_id=event_ids,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        reject=reject_criteria,
        preload=True,
    )

    # --- Export features ---
    X = epochs.get_data()        # (trials, channels, timepoints)
    y_raw = epochs.events[:, -1] # labels
    label_map = {old: new for new, old in enumerate(event_ids.values())}
    y = np.array([label_map[val] for val in y_raw])

    # Balance classes
    rng = np.random.default_rng(seed=42)
    min_count = min([np.sum(y == cls) for cls in np.unique(y)])

    X_bal, y_bal = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        chosen = rng.choice(idx, size=min_count, replace=False)
        X_bal.append(X[chosen])
        y_bal.append(y[chosen])

    X_bal = np.vstack(X_bal)
    y_bal = np.hstack(y_bal)

    np.save(os.path.join(FEATURES_DIR, f"{subj_id}_X.npy"), X_bal)
    np.save(os.path.join(FEATURES_DIR, f"{subj_id}_y.npy"), y_bal)

    print(f"Saved {X_bal.shape[0]} balanced trials for {subj_id}")
    return True

# -------------------------
# 3. Run all subjects
# -------------------------
if __name__ == "__main__":
    for subj in subjects:
        preprocess_subject(subj)

    print("\nBatch preprocessing complete!")

