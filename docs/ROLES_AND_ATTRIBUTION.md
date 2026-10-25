# Roles and attribution (the honest version)

## Two tracks
**Track A: VIPER feasibility prototype (this repo's `snake/`).** Claude Code writes the code and tests. Aradhya writes the requirements, scoring logic, thresholds, UI decisions, and test acceptance criteria, runs the app, tests it, and decides what changes. Purpose: prove feasibility fast and seed S.N.A.K.E. software. Honest description: *"Designed and directed by Aradhya Pasikanti; implemented with Claude Code (AI)."*

**Track B: Aradhya's own build (`student/`).** For the Congressional App Challenge, the code in the entry must be his. He re-implements the core pipeline himself, using Track A as a reference and Claude as a tutor, and the CAC video, answers and disclosure describe only what he actually did.

## Attribution rules
- A contribution list in the repo names the contributor (Aradhya = design, requirements, logic, testing, decisions; Claude = implementation). Listing it without names or committing so the history shows only Aradhya would misstate who wrote the code; it is not allowed in anything submitted to CAC.
- Git: commit normally; keep Claude's Co-Authored-By trailer on AI-assisted commits.
- For a future S.N.A.K.E. repo (not a contest), Aradhya decides how to credit; recommended: same CONTRIBUTIONS table.

## CONTRIBUTIONS.md template (in repo root, fill as you go)
| Area | Aradhya | Claude Code |
| --- | --- | --- |
| Problem, requirements, user flow | yes | drafting help |
| Scoring criteria and weights | decides | proposes, implements |
| Image pipeline code | reviews, tunes | writes |
| Learning layer | decides features/labels | writes |
| Tests | sets acceptance criteria | writes |
| UI design | decides layout | writes |
| `student/` re-implementation | writes | tutor/reviewer only |

## What CAC needs from Track B (be realistic by Monday noon)
Minimum credible student contribution in ~24 hours: write 3-4 functions yourself in `student/` (enhance, vein_mask, skeleton or arm coordinates), plug them into the app through a toggle ("Reference" vs "Student"), and be able to explain them on camera. Then say plainly in the disclosure what Claude did. If that is not achievable, the honest alternatives are to skip CAC this year and keep VIPER for S.N.A.K.E., ISEF and applications (you can describe it as AI-assisted, directed by you), or submit with maximum disclosure and accept the risk of disqualification or low scoring.
