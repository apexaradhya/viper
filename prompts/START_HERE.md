Read CLAUDE.md, docs/CLAUDE_CODE_MASTER_BRIEF.md, docs/ROLES_AND_ATTRIBUTION.md and docs/DECISIONS.md. You are in BUILDER mode: you write the code and tests, I (Aradhya) direct the logic and test the app. The reference pipeline in snake/ already works (run `pytest -q`, expect 20 passed, and `streamlit run app.py`).

Today's goals, in order, one at a time, showing me the result in the app after each:
1. Verify the repo runs on my machine; fix anything environment-related.
2. UI polish pass using section 10 of the brief: strong first impression (veins plus numbered sites visible immediately), clear one-sentence explainer per tab, "why this site" text, accept/reject, Rules/Learned/Blend, calm high-contrast palette (no red/green-only meaning), visible disclaimer, friendly errors. Show me screenshots or describe changes and wait for my feedback.
3. Add an "About VIPER" expander with a placeholder for my own mission sentence (I will write it).
4. Make it deployable on Streamlit Community Cloud: check requirements.txt, Python version, no absolute paths, data/ writes work, then give me the exact deploy steps.
5. Keep docs honest: after each session append to docs/AI_LOG.md what you wrote and what I decided. Never describe your code as mine. Keep Co-Authored-By trailers. No clinical claims; only report measured numbers.

Rules: explain each design choice in plain language so I can defend it; ask me before changing scoring weights or thresholds (record my decisions in docs/DECISIONS.md); run tests before saying anything works. Start with step 1 now.
