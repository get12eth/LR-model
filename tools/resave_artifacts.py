"""Resave joblib artifacts with the runtime scikit-learn version.

This script loads each artifact, writes a .bak backup, then re-dumps the artifact
using the currently installed joblib/scikit-learn. Use this to remove
InconsistentVersionWarning caused by pickles created with an older sklearn.

Run:
    .\.venv\Scripts\python.exe tools\resave_artifacts.py

It will keep backups named <file>.bak so you can restore if needed.
"""
from pathlib import Path
import joblib
import sys

ARTIFACTS = [
    Path('models/logistic_regression_model.joblib'),
    Path('models/standard_scaler_lr.joblib'),
    Path('models/label_encoders_lr.joblib'),
]

def resave(path: Path):
    if not path.exists():
        print(f"Missing: {path}")
        return False
    try:
        print(f"Loading {path} ...")
        obj = joblib.load(path)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return False

    bak = path.with_suffix(path.suffix + '.bak')
    print(f"Backing up {path} -> {bak}")
    try:
        path.replace(bak)
    except Exception as e:
        print(f"Could not create backup: {e}")
        return False

    try:
        joblib.dump(obj, path)
        print(f"Re-saved {path}")
        return True
    except Exception as e:
        print(f"Failed to re-save {path}: {e}")
        # attempt to restore backup
        try:
            bak.replace(path)
            print(f"Restored original {path} from {bak}")
        except Exception:
            print(f"Could not restore original for {path}; manual recovery may be needed.")
        return False

def main():
    any_ok = False
    for p in ARTIFACTS:
        ok = resave(p)
        any_ok = any_ok or ok
    if not any_ok:
        print("No artifacts were resaved. Check messages above.")
        sys.exit(2)

if __name__ == '__main__':
    main()
