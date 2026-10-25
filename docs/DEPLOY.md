# Publishing VIPER on Streamlit Community Cloud

Checked on 2026-10-24: `pytest -q` 32 passed; app loads Home, History, Settings with no errors; requirements pinned
to the tested versions; no absolute paths (data paths are relative to the repo root, which is where Streamlit Cloud
runs the app; the logo path is built from the code's own location).

## What gets published and what stays on your laptop
- Published: code (`snake/`, `ui/`, `app.py`, `scripts/`, `tests/`), `.streamlit/config.toml`, `assets/` (logo),
  `models/site_model.pkl` (numbers only), `data/labels/site_labels.json` (your labels, coordinates only), `docs/`.
- Never published (in `.gitignore`): `.venv/`, Kaggle images (`data/kaggle_forearm_nir/`, license "Unknown"),
  `data/viz/`, your photos in `data/problem_images/`, `data/site_log.json`, `data/feedback.json`.
- On the public site, each visitor's site history and Good/Bad choices are private to their browser tab and are
  gone when the tab closes (VIPER only saves them to files when `VIPER_LOCAL_FILES=1`, as on your laptop).

## 1. Install git (once)
Open PowerShell and run, then close and reopen PowerShell:
```bash
winget install --id Git.Git -e
```
Then tell git who you are (your name and the email of your GitHub account):
```bash
git config --global user.name "Your Name"
```
```bash
git config --global user.email "you@example.com"
```

## 2. Make the first commit
In the project folder (`E:\projects\snake\snake-cac\snake-cac`):
```bash
git init -b main
```
```bash
git add .
```
Check that no Kaggle images or personal photos are listed before committing:
```bash
git status
```
Commit. Keep the Co-Authored-By line: it records that Claude Code wrote code in this project (project rule in CLAUDE.md):
```bash
git commit -m "VIPER prototype: vein finding, site ranking, rotation, mobile UI" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

## 3. Put it on GitHub
1. Sign in at https://github.com and click **New repository**. Name it e.g. `viper`. Leave "Add a README" unchecked.
2. Copy the two commands GitHub shows under "push an existing repository", e.g.:
```bash
git remote add origin https://github.com/YOUR-USERNAME/viper.git
```
```bash
git push -u origin main
```

## 4. Deploy
1. Go to https://share.streamlit.io and sign in with GitHub.
2. **Create app** -> **Deploy a public app from GitHub**.
3. Repository: `YOUR-USERNAME/viper`, branch `main`, main file path `app.py`.
4. **Advanced settings** -> Python version **3.12** (the version tested). No secrets are needed.
5. Click **Deploy**. The first start installs the libraries and takes a few minutes.
6. Open the URL on your phone. Tip: in the phone's browser menu choose "Add to Home Screen" so it opens like an app
   with the VIPER icon.

## 5. After every change
```bash
git add .
```
```bash
git commit -m "Describe the change" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```
```bash
git push
```
Streamlit Cloud redeploys automatically.

## Known limits of the public version
- No NIR camera yet: "Scan" shows a placeholder message; visitors need their own NIR photo to upload.
- The developer pages (Dev, Label) appear only with `?dev=1`; the Label page needs the Kaggle images, which are
  only on your laptop, so it shows "No Kaggle images found" online.
- Educational research prototype; no clinical accuracy is claimed (measured results: docs/EVAL_REAL_final.txt).
