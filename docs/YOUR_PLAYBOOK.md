# Aradhya's playbook (vibe-coding workflow)

## Where we develop
- **Claude Code on your own computer** is the main workshop. It edits files, runs tests, and runs the app next to you so you can see the UI and give feedback in seconds. Install: follow the Claude Code setup at claude.com (needs a terminal; macOS Terminal, or Windows via WSL). Open it inside the repo folder with `claude`.
- This chat (planning, logic, review) stays separate. Nothing here needs to connect to your terminal; the repo is the shared artifact. If you want me to build a chunk here first, I can run and test it in my sandbox and hand you the zip.
- Python in a virtual environment: `python3 -m venv .venv`, activate it, `pip install -r requirements.txt`.

## Setup (45 min tonight)
1. Register for CAC (personal email, home and school ZIP+4, district, parent contact).
2. Unzip `snake-cac.zip`, set up the venv, run `pytest -q` and `streamlit run app.py`.
3. GitHub: new Public repo `viper`; `git init`, `git add .`, `git commit -m "Initial scaffold"`, `git remote add origin ...`, `git push -u origin main`. You commit and push; Claude Code's Co-Authored-By trailer stays on AI-assisted commits.
4. In Claude Code, paste: "Read CLAUDE.md, docs/CLAUDE_CODE_MASTER_BRIEF.md and docs/ROLES_AND_ATTRIBUTION.md, then implement Steps 1-4 of the pipeline with tests, one step at a time, showing me results in the app after each."

## Build loop (your job: logic and testing)
For each step Claude Code implements -> you open the app -> you judge the output ("veins look too noisy", "sites cluster") -> you change requirements -> repeat. Keep a short list of your decisions (thresholds, weights, what counts as a good site) in `docs/DECISIONS.md`; that file is evidence of your contribution.

## Track B for CAC (do not skip if you will submit)
Sunday morning, after the reference works, write 3-4 functions yourself in `student/` with Claude Code as tutor (not author), wire the toggle, and describe it exactly in the video and disclosure. See `docs/ROLES_AND_ATTRIBUTION.md`.

## UI design: when and how
- Sat evening: functional UI (current Streamlit tabs) just to see outputs.
- Sun 10 am-1 pm: real UI pass. Give Claude Code the UI spec (brief section 10) and iterate on screenshots. Priorities: first-10-seconds visual (veins + numbered sites), "why this site" text, accept/reject, Rules/Learned/Blend switch, visible disclaimer.

## Putting it online
- **Code**: GitHub public repo (CAC asks for a code link).
- **Live app**: Streamlit Community Cloud (free): sign in with GitHub, choose the repo, main file `app.py`, deploy. It reads `requirements.txt`, so keep `opencv-python-headless`. Test the live URL logged out. Deploy by Sunday 2 pm so there is time to fix problems.
- **Video**: record the screen, upload to YouTube/Vimeo as public.
- **Later for S.N.A.K.E.**: keep `snake/` as a pure-Python library with no Streamlit imports, so the same code can run behind a FastAPI service, on a Raspberry Pi with a NIR camera, or in a mobile/web front end. Streamlit stays the demo shell.

## Cut order if behind
Real-image demo, UI polish, blend mode, learning layer. Never cut: registration, tests, honest disclosure, video, submission.

## Sleep by 11 pm.
