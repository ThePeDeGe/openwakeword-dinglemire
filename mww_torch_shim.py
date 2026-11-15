"""
Shim: torch>=2.6 domyślnie ładuje checkpointy z weights_only=True, co blokuje
checkpoint VITS piper-sample-generator (global SynthesizerTrn). Uruchom:
    python mww_torch_shim.py <script> [args...]
"""
import sys
import torch

_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python mww_torch_shim.py <script> [args...]")
        sys.exit(2)
    script = sys.argv[1]
    sys.argv = [script] + sys.argv[2:]
    code = open(script).read()
    g = {"__name__": "__main__", "__file__": script}
    exec(compile(code, script, "exec"), g)
