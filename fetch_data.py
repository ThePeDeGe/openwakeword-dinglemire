# Downloads the negative/background training data (ported from cell 3 of
# saintpete/openwakeword-colab-toolkit, verified 2026-08-01; FMA skipped
# for this CPU-budget run — see config/dangelmaier.yaml).
#
# Every guard is CONTENT-based (file counts / exact byte sizes), so
# re-running repairs a half-filled directory instead of skipping it.
import os
import numpy as np
import scipy.io.wavfile
import datasets
from pathlib import Path
from tqdm import tqdm


def wav_count(d):
    return len(list(Path(d).glob("*.wav"))) if os.path.isdir(d) else 0


# --- MIT room impulse responses -> ./mit_rirs (270 clips)
output_dir = "./mit_rirs"
if wav_count(output_dir) < 250:
    os.makedirs(output_dir, exist_ok=True)
    rir_dataset = datasets.load_dataset(
        "davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True)
    for row in tqdm(rir_dataset, desc="MIT RIRs"):
        name = row["audio"]["path"].split("/")[-1]
        scipy.io.wavfile.write(os.path.join(output_dir, name), 16000,
                               (row["audio"]["array"] * 32767).astype(np.int16))

# --- AudioSet ambient clips -> ./audioset_16k (2000 x 10 s ~= 5.5 h)
output_dir = "./audioset_16k"
N_AUDIOSET = 2000
if wav_count(output_dir) < N_AUDIOSET:
    os.makedirs(output_dir, exist_ok=True)
    aset = datasets.load_dataset("agkphysics/AudioSet", "balanced",
                                 split="train", streaming=True)
    aset = aset.cast_column("audio", datasets.Audio(sampling_rate=16000))
    done = 0
    for row in tqdm(aset, total=N_AUDIOSET, desc="AudioSet -> 16k"):
        stem = str(row.get("video_id") or f"audioset_{done:05d}")
        scipy.io.wavfile.write(os.path.join(output_dir, stem + ".wav"), 16000,
                               (row["audio"]["array"] * 32767).astype(np.int16))
        done += 1
        if done >= N_AUDIOSET:
            break

# --- Precomputed openWakeWord features (sizes verified via HTTP HEAD;
#     wget -c resumes a partial download instead of starting over)
FEATURES = {
    "openwakeword_features_ACAV100M_2000_hrs_16bit.npy": 17280000128,
    "validation_set_features.npy": 184836608,
}
for fname, fsize in FEATURES.items():
    if not (os.path.exists(fname) and os.path.getsize(fname) == fsize):
        url = ("https://huggingface.co/datasets/davidscripka/openwakeword_features"
               f"/resolve/main/{fname}")
        os.system(f"wget -c -q {url}")
        assert os.path.exists(fname) and os.path.getsize(fname) == fsize, \
            f"{fname}: incomplete download, rerun this step"

print("Data ready:",
      wav_count("mit_rirs"), "RIRs |",
      wav_count("audioset_16k"), "AudioSet clips |",
      "features OK")
