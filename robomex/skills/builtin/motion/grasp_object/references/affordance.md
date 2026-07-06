# Act-Owned Affordance Notes

This reference replaces the old affordance-delegation pattern.

Act owns grasp candidate generation. After loading the relevant perception and
affordance skills, Act may:

- read local `EVIDENCE` masks, points, and geometry summaries;
- generate top-down, PCA/side, open-bowl, or GraspNet candidates;
- check candidate feasibility with `solve_ik`;
- save `affordance_candidates.png` or a 3D review artifact under `ARTIFACTS_DIR`;
- execute only one bounded motion block after selecting a candidate.

Use the Verifier SubAgent only to check a concrete visual claim, such as whether the
current fingertips straddle the rim or whether the lifted object is visibly held. The
Verifier should not generate or return execution poses.
