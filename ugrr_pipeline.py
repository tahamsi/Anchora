"""UGRR pipeline entrypoint.

Usage examples:
  python ugrr_pipeline.py sft --config config.json
  python ugrr_pipeline.py train_reward --config config.json
  python ugrr_pipeline.py regret_loop --config config.json
  python ugrr_pipeline.py evaluate --model_path models/ugrr_final --baseline_path models/mistral_sft
"""
import os
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from ugrr.cli import main


if __name__ == "__main__":
    main()
