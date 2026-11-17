"""Exercise 1.1: check that your environment is ready for the lab.

Run:  python check_setup.py
Every line should end with OK. Fix anything marked PROBLEM before you continue.
"""
import importlib
import platform
import sys

from config import MODEL_ID

ok = True


def report(label: str, good: bool, detail: str = "") -> None:
    global ok
    ok = ok and good
    status = "OK" if good else "PROBLEM"
    print(f"{label:<28} {detail:<40} {status}")


# 1. Python version
v = sys.version_info
report("Python version", v >= (3, 11), f"{v.major}.{v.minor}.{v.micro} (need 3.11 or newer)")

# 2. Required packages
for name in ["torch", "transformers", "huggingface_hub", "smolagents", "pytest"]:
    try:
        module = importlib.import_module(name)
        report(f"package {name}", True, getattr(module, "__version__", "installed"))
    except ImportError:
        report(f"package {name}", False, "not installed: pip install -r requirements.txt")

# 3. Compute device
try:
    import torch

    if torch.cuda.is_available():
        device = f"cuda ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        device = "mps (Apple GPU)"
    else:
        device = f"cpu ({torch.get_num_threads()} threads)"
    report("compute device", True, device)
except ImportError:
    pass

# 4. Hugging Face login (optional, but avoids download limits)
try:
    from huggingface_hub import whoami

    user = whoami()["name"]
    report("Hugging Face login", True, f"logged in as {user}")
except Exception:
    print(f"{'Hugging Face login':<28} {'not logged in (optional: hf auth login)':<40} --")

# 5. Is the model already downloaded? (MODEL_ID can be a Hub id or a local folder)
from pathlib import Path

if Path(MODEL_ID).is_dir():
    report("model folder", (Path(MODEL_ID) / "config.json").exists(), MODEL_ID)
else:
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(MODEL_ID, "config.json")
        good = isinstance(cached, str)
        report("model in local cache", good, MODEL_ID if good else f"run: hf download {MODEL_ID}")
    except Exception as exc:  # e.g. an invalid model id
        report("model in local cache", False, f"{MODEL_ID}: {exc}")

print()
print(f"System: {platform.system()} {platform.release()}, machine {platform.machine()}")
print("All checks passed." if ok else "Some checks failed: fix them before the next exercise.")
