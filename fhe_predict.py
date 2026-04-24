import numpy as np
import pickle
from pathlib import Path
from concrete.ml.deployment import FHEModelClient, FHEModelServer
import time

SAVE_FOLDER = Path("saved_model")
CLIENT_ZIP  = SAVE_FOLDER / "client.zip"
SERVER_ZIP  = SAVE_FOLDER / "server.zip"
SYMPTOMS_PK = SAVE_FOLDER / "symptoms_list.pkl"

if not all(f.exists() for f in [CLIENT_ZIP, SERVER_ZIP, SYMPTOMS_PK]):
    print("ERROR: Model files not found. Run train_once.py first.")
    exit(1)

print("Loading model...")
client = FHEModelClient(path_dir=str(SAVE_FOLDER), key_dir=str(SAVE_FOLDER))
server = FHEModelServer(path_dir=str(SAVE_FOLDER))

with open(SYMPTOMS_PK, "rb") as f:
    all_symptoms = pickle.load(f)

# 15 selected symptoms (no sex-related)
KEY_SYMPTOMS = [
    "Heavy / Extreme menstrual bleeding",
    "Menstrual pain (Dysmenorrhea)",
    "Pelvic pain",
    "Cramping",
    "Abdominal pain / pressure",
    "Back pain",
    "Lower back pain",
    "Bloating",
    "Fatigue / Chronic fatigue",
    "Nausea",
    "Headaches",
    "Irregular / Missed periods",
    "Depression",
    "Anxiety",
    "Painful bowel movements"
]

symptom_to_idx = {sym: all_symptoms.index(sym) for sym in KEY_SYMPTOMS if sym in all_symptoms}

print(f"Model loaded – asking {len(symptom_to_idx)} symptoms")

while True:
    print("\n" + "=" * 70)
    print("   NEW RISK ASSESSMENT")
    print("=" * 70 + "\n")

    vector = np.zeros((1, len(all_symptoms)), dtype=float)
    yes_count = 0

    print("Answer: y/yes/n/no   (Enter = no)\n")

    for symptom in KEY_SYMPTOMS:
        if symptom not in symptom_to_idx:
            continue
        ans = input(f"{symptom} ? ").strip().lower()
        if ans in ['y', 'yes', 'ye']:
            vector[0, symptom_to_idx[symptom]] = 1.0
            yes_count += 1
            print("   → YES")
        else:
            print("   → NO")

    print(f"\nYES answers: {yes_count} / {len(symptom_to_idx)}")

    if yes_count == 0:
        print("\nRESULT: LOW RISK (0%)")
    else:
        print("\nComputing encrypted prediction...")

        t_start = time.time()

        serialized_input = client.quantize_encrypt_serialize(vector)
        serialized_eval_keys = client.get_serialized_evaluation_keys()

        enc_output = server.run(serialized_input, serialized_eval_keys)

        decrypted_array = client.deserialize_decrypt_dequantize(enc_output)

        # FIXED: take probability of class 1 (positive/high risk)
        proba = float(decrypted_array[:, 1].item())   # shape is usually (1,2)

        elapsed = time.time() - t_start

        percentage = round(proba * 100, 1)
        risk = "HIGH RISK" if proba >= 0.5 else "LOW RISK"

        print("\n" + "-" * 60)
        print(f"   RESULT:          {risk}")
        print(f"   Probability:     {percentage}%")
        print(f"   Raw value:       {proba:.4f}")
        print(f"   Time:            {elapsed:.2f} seconds")
        print("-" * 60)

    again = input("\nAnother assessment? (y/n): ").strip().lower()
    if again not in ['y', 'yes']:
        print("Done.")
        break