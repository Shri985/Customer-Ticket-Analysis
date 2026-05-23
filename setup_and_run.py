"""
setup_and_run.py
One-click setup: installs dependencies, generates dataset, trains models, launches app.
"""

import subprocess
import sys
import os


def run(cmd, desc):
    print(f"\n{'='*55}")
    print(f"  {desc}")
    print(f"{'='*55}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[ERROR] Step failed: {desc}")
        sys.exit(1)


if __name__ == "__main__":
    print("\n🎫 Customer Ticket AI — Setup & Launch")

    run(f"{sys.executable} -m pip install -r requirements.txt -q", "Installing dependencies")
    run(f"{sys.executable} generate_dataset.py", "Generating dataset (8,500 tickets)")
    run(f"{sys.executable} train.py", "Training ML models")
    run("streamlit run app.py", "Launching Streamlit dashboard")
