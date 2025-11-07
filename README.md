# LoanGuard (Streamlit) — Repo & Deployment Instructions

This repository contains a Streamlit app (`app.py`) for loan default risk prediction.

Quick checklist before publishing to GitHub / Streamlit Community Cloud

- Ensure `requirements.txt` is up to date (contains packages like `streamlit`, `scikit-learn`, `joblib`, `pandas`, `numpy`, `matplotlib`).
- Do NOT commit your virtual environment (`.venv`) or `site-packages` — `.gitignore` is included to prevent that.
- Remove any accidentally tracked `.venv` files from Git (command provided below).

Recommended GitHub / Streamlit publish steps (PowerShell)

1) Make sure files are saved and up to date locally.

2) If `.venv` was previously committed, untrack it and commit the `.gitignore`:

```powershell
# from project root
git add .gitignore README.md
git commit -m "Add .gitignore and README"

# If .venv was previously tracked, remove it from the index (safe: preserves files locally)
git rm -r --cached .venv
git commit -m "Stop tracking .venv"
```

3) Create a GitHub repository and push (two options):

- Option A — Using GitHub website
  - Create a new repo on github.com (choose a name, e.g., `loan-guard`)
  - Follow the provided instructions to add a remote and push:
```powershell
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

- Option B — Using GitHub CLI (if installed)
```powershell
gh repo create <your-username>/<repo-name> --public --confirm
git push -u origin main
```

4) Deploy on Streamlit Community Cloud

- Go to https://share.streamlit.io and log in with your GitHub account.
- Click "New app" → Select your repository, branch (main), and the main file path (`app.py`).
- Click "Deploy".

Notes about scikit-learn warnings

You may see runtime warnings like:

```
InconsistentVersionWarning: Trying to unpickle estimator ... from version 1.5.1 when using version 1.7.2.
```

This means the saved joblib artifacts were created with a different scikit-learn version. Recommended options:

- Re-save the artifacts using your current scikit-learn version (best): create a small script that loads each joblib file and re-dumps it.
- Or run the app in an environment with the older sklearn version (quick workaround).

If you want, I can add a `tools/resave_artifacts.py` script to re-save model artifacts (safe: keeps backups). Say "please add resave script" and I'll add it.

If you get stuck on any step, paste the git output here and I'll help debug the next action.
