"""Settings shared by all the Lab 1 scripts.

You can override the model without editing this file, for example:
    Linux/macOS:  LAB_MODEL=Qwen/Qwen2.5-0.5B-Instruct python chat_demo.py
    Windows (PowerShell):  $env:LAB_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   (then run the script)
"""
import os
from pathlib import Path

# Qwen2.5-1.5B-Instruct: about 3 GB download, needs about 7 GB of free RAM on a CPU.
# Qwen2.5-0.5B-Instruct: about 1 GB download, runs on laptops with 8 GB of RAM.
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SMALL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_ID = os.environ.get("LAB_MODEL", DEFAULT_MODEL)

# Optional overrides. Empty = choose automatically.
DTYPE = os.environ.get("LAB_DTYPE", "")     # "float32", "bfloat16" or "float16"
DEVICE = os.environ.get("LAB_DEVICE", "")   # "cpu", "cuda" or "mps"

MAX_NEW_TOKENS = 256   # upper limit for one model reply
MAX_STEPS = 10         # upper limit for the Thought/Action/Observation cycles of the agent
                       # (the longest question needs 8: list the robots, look up all five, add, answer)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
