# openwakeword-dinglemire

Training pipeline for a custom **openWakeWord** wake word: **"Dinglemire"**,
used with Home Assistant + openWakeWord add-on (Wyoming, port 10400).

Trained phrases (one binary model, fires on any):
- `dinglemire` — English reading (DAN-jil-MAY-er)
- `dangle myer` — Polish/German reading (DANG-gel-MAI-er, hard g)

Based on the verified 2026 recipe from
[saintpete/openwakeword-colab-toolkit](https://github.com/saintpete/openwakeword-colab-toolkit)
(oww_train_shim.py vendored unmodified), running **on CPU** in GitHub Actions —
no GPU needed, ~2–4 h per run.

## What the workflow does

1. Frees runner disk (~30 GB) for the 17.3 GB precomputed negative features.
2. Installs the proven dependency stack (openwakeword 0.6.0 `--no-deps`, modern
   torch, shim for removed torchaudio/scipy APIs).
3. Downloads: piper-sample-generator (dscripka fork) + libritts-high voice,
   MIT room impulse responses, AudioSet ambient clips (2000 × 10 s),
   precomputed ACAV100M negative features (17.3 GB) + 11 h FPR validation set.
4. Generates 6000 synthetic clips (2 phrasings), augments ×2,
   trains a DNN layer-32 model (≤30k steps, early stop at 0.2 FP/h).
5. Scores validation positives, uploads `.onnx` (+ `.tflite` if produced)
   as the `dinglemire-model` artifact.

Tuning lives in `config/dinglemire.yaml` — editing it re-runs the training.

## Deployment (Home Assistant)

Copy `dinglemire_v1.onnx` into the add-on's custom model dir
(`/share/openwakeword`), restart the openWakeWord add-on, the wake word
`dinglemire_v1` appears via the Wyoming integration.
