# Demo video script (target 2:30, hard limit 3:00)

Required content (from the rules): participant name(s), app name, purpose in ONE clear sentence, target audience, tools and languages used, and a demonstration of the functionality. Judging criteria: quality of the idea, implementation (UX and design), demonstrated coding skill.

Record the screen with voiceover (QuickTime, OBS or Loom). Keep it plain; this is not a video-editing contest.

| Time | On screen | Say (edit into your own words) |
| --- | --- | --- |
| 0:00-0:15 | Title slide or app header | "Hi, I'm Aradhya Pasikanti. This is VIPER. It maps veins in near-infrared arm images, ranks the best access sites, and tracks which sites were used so they can be rotated." |
| 0:15-0:35 | Short bullets or a photo-free diagram | "It's for nursing and phlebotomy students and, as a concept, for care teams treating patients who need repeated IV access, where hard-to-find veins and reusing the same spot cause harm. This is an educational research prototype, not a medical device." |
| 0:35-0:50 | Code editor / terminal | "I built it in Python with Streamlit, NumPy, SciPy, scikit-image and OpenCV. I designed VIPER and directed the build. [Say exactly which parts you wrote yourself in student/ and which Claude wrote.] All AI use is disclosed in the repo." |
| 0:50-1:20 | Enhance tab, then Veins tab | "First I enhance contrast with CLAHE. Then a multi-scale Frangi ridge filter finds dark tube-like structures, I threshold and remove specks, and skeletonize to center lines. On synthetic arms with known veins, X percent of my skeleton lies within 4 pixels of a true vein." (use YOUR real number) |
| 1:20-1:55 | Sites tab | "Each point on a vein gets a score from width, straightness, and distance from branch points and edges. I pick the top sites with spacing so they aren't all in one spot, and the app explains each one." |
| 1:55-2:25 | Rotation tab: log a site, show ranking change | "To rotate sites, I convert pixels to arm coordinates between the wrist and elbow. When I log a used site, nearby candidates drop in the ranking, with a note saying why." |
| 2:25-2:50 | README / GitHub page or tests passing | "Tests check accuracy against ground truth. Next, I'd test on real near-infrared images and connect this to a camera and the S.N.A.K.E. robotic platform I'm building." |

## Rules for the video
- 1:00-3:00 long. Aim for 2:30.
- Say your name and the app name out loud and show them on screen.
- Say the purpose as one sentence.
- Show a working flow, not slides.
- Do NOT claim clinical accuracy or real-patient results.
- Upload to YouTube or Vimeo and set visibility to **Public** (not unlisted if the portal requires public). Test the link logged out.
