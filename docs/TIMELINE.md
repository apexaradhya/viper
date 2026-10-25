# 48-hour timeline (Sat 12:00 pm -> Mon 12:00 pm EDT). Submit by Sunday 8 pm.

Rule of thumb: ship the smallest complete version, then improve. Sleep. A tired mistake in the submission form costs more than any feature.

## Saturday
| Time | Task | Done when |
| --- | --- | --- |
| 12:00-12:45 | **Register for CAC** (see checklist A). Create GitHub repo, push scaffold. Install deps, run `pytest -q` and `streamlit run app.py`. | Registration email received; app shows "not implemented yet" notes |
| 12:45-2:30 | **Step 1+2a:** write `enhance()` and `vesselness()`. | Enhance and vesselness tests pass; Veins tab shows a vesselness image |
| 2:30-3:00 | Break and food. | |
| 3:00-4:30 | **Step 2b+2c:** `vein_mask()` and `skeleton()`; tune threshold. | Skeleton test passes (85% within 4 px); accuracy metric shows in app |
| 4:30-7:30 | **Step 3:** `score_candidates()` then `pick_top()`. | Sites tab shows numbered sites on veins; tests pass |
| 7:30-8:30 | Dinner. Update `AI_LOG.md`. Commit. | |
| 8:30-10:30 | **Step 4a:** `to_arm_coords()`; start `rank_with_rotation()`. | Arm-coordinate test passes |
| 10:30 | Stop. Commit. Sleep. | |

## Sunday
| Time | Task | Done when |
| --- | --- | --- |
| 9:00-10:30 | Finish `rank_with_rotation()`; run full app flow | All tests pass, no skips; rotation demo works |
| 10:30-12:00 | Try 1-2 real public NIR images (license permitting); tune; note honest results in README | One real-image screenshot, honestly described |
| 12:00-12:45 | Lunch | |
| 12:45-2:30 | Polish UI text and disclaimer; README; `AI_USE_DISCLOSURE.md`; `DATA_SOURCES.md` | Fresh-clone run works |
| 2:30-4:30 | Rehearse and **record the video** (max 3:00); upload to YouTube/Vimeo as Public | Link works logged out |
| 4:30-6:00 | Write the written answers (Q3-Q5 in your own voice) | `SUBMISSION_ANSWERS.md` final |
| 6:00-7:30 | Fill the portal form; run the final checklist (F) | Everything pasted and verified |
| **by 8:00 pm** | **SUBMIT.** Save confirmation. | Confirmation received |
| 8:00-10:00 | Optional stretch only if rested. Do not change the submission. | |

## Monday
- Buffer only. If anything failed to submit, fix it before school. Hard deadline 12:00 pm EDT.

## Cut list if you fall behind (cut in this order)
1. Real-image testing (keep synthetic + metric)
2. Stretch features
3. Pretty UI
4. Rotation sophistication (use plain pixel distance, say so)
Never cut: registration, tests, AI disclosure, video, submission.
