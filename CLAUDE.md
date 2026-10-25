# Project rules for Claude Code (read first every session)

VIPER: Aradhya's vein-imaging / puncture-site ranking prototype for S.N.A.K.E. Aradhya owns the product logic, requirements, testing and decisions. **Claude Code writes the code and tests** (builder mode). Read `docs/CLAUDE_CODE_MASTER_BRIEF.md` and `docs/ROLES_AND_ATTRIBUTION.md`.

## What you do
- Implement everything in `snake/`, `app.py`, `scripts/`, and `tests/`, including the functions marked `STUDENT:` in stubs (those markers now mean "reference implementation: Claude writes it; Aradhya may later re-implement it himself in `student/`").
- Explain every algorithm choice in plain language when you implement it, so Aradhya can defend it. Put the "why" in docstrings.
- Run `pytest -q` and the sweep script before saying anything works. Report only measured numbers and name the data.
- Keep parameters in `snake/config.py`. Python 3.11+, type hints, short functions.

## Honesty rules (non-negotiable)
- Do not claim clinical accuracy or real-patient results. (The on-screen "not a medical device" banner was removed from the app by Aradhya's decision on 2026-10-24; see docs/DECISIONS.md. Do not re-add it unless he asks.)
- Record the truth about authorship: append an entry to `docs/AI_LOG.md` after every session stating what Claude wrote and what Aradhya decided/changed. Never describe AI-written code as written by Aradhya, in docs, commit messages, README, or the Congressional App Challenge materials.
- Leave git commit attribution as the tooling produces it (Co-Authored-By trailer). Do not strip or hide it.
- Do not download or train on any image without recording its license in `docs/DATA_SOURCES.md`.

## Contest guard
The Congressional App Challenge requires the student to do the coding and to disclose AI use. If a CAC submission is being prepared from this repo, remind Aradhya that only the code he personally wrote may be described as his, and help him build `student/` so there is real, honest, personal work to submit.
