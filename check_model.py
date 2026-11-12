# Post-training sanity check: loads the freshly trained .onnx with
# openwakeword and scores the generated validation positive clips.
# Informational only — the real gate is the user's live testing on the
# Voice PE satellites.
import glob
import os
import sys
from pathlib import Path

import numpy as np
import scipy.io.wavfile

MODEL_NAME = "dangelmaier_v1"
THRESHOLD = 0.5


def find_model():
    cands = sorted(Path(".").glob("**/" + MODEL_NAME + "*.onnx"))
    assert cands, (
        "No " + MODEL_NAME + " .onnx found. If the training step failed, its "
        "error is in the previous step. Contents of the model dir: "
        + str(sorted(str(p) for p in Path("dangelmaier_model").glob("**/*"))[:80]))
    return cands[0]


def find_val_positives():
    # openwakeword train.py layout: <output_dir>/<model_name>/<...>_val_positive_clips/
    pats = [
        "dangelmaier_model/**/*val*positive*clips*/*.wav",
        "dangelmaier_model/**/*val*positive*.wav",
        "dangelmaier_model/**/*positive*/**/*.wav",
    ]
    for pat in pats:
        clips = sorted(glob.glob(pat, recursive=True))
        if clips:
            return clips
    return []


def main():
    onnx_path = find_model()
    print(f"Model: {onnx_path}  ({onnx_path.stat().st_size/1024:.0f} KiB)")

    from openwakeword.model import Model
    model = Model(wakeword_models=[str(onnx_path)])

    clips = find_val_positives()
    if not clips:
        print("WARN: no validation positive clips found — skipping scoring")
        return 0

    scores = []
    for wav in clips[:500]:
        sr, audio = scipy.io.wavfile.read(wav)
        model.reset()
        chunk = 1280
        peak = 0.0
        for i in range(0, len(audio) - chunk, chunk):
            frame = np.int16(audio[i:i + chunk])
            preds = model.predict(frame)
            peak = max(peak, max(preds.values()))
        scores.append(peak)

    scores = np.array(scores)
    recall = float((scores >= THRESHOLD).mean())
    print(f"Validation positives scored: {len(scores)}")
    print(f"  median peak score: {np.median(scores):.3f}")
    print(f"  p25/p75: {np.percentile(scores, 25):.3f} / {np.percentile(scores, 75):.3f}")
    print(f"  recall@{THRESHOLD}: {recall:.1%}")
    if recall < 0.5:
        print("WARNING: recall below 50% — model likely undertrained; "
              "consider more n_samples/steps before deploying.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
