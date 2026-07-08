# HANDOFF: MPPI full-length fixes (run on the GPU workstation)

**For the agent picking this up** (Claude session on NvidiaWorkstation, repo
cloned from AntoineRichard/newton): you are resuming a paused campaign. This
file is your complete brief. Work on branch `antoiner/wheeled-vehicle-design`
(fetch + reset to the fork tip first). Push to the fork when done.

## Context in 30 seconds

Read, in order:
1. `docs/superpowers/notes/2026-07-07-mppi-task1-baseline-and-braking.md` —
   the entire experimental record (protocol, esc brake mode, spline knots,
   rejected knobs, and the two 2026-07-08 sections at the end).
2. `docs/superpowers/plans/2026-07-07-dial-mpc-annealing-for-mppi.md` — the
   plan with per-task checkboxes and decisions.

Headline state: the accepted config (`--brake-mode esc --n-knots 12`) wins
every 240-frame acceptance metric but LOSES a 1300-frame run: it goes OOB at
the hairpin-1 exit on the tuning track (hull s4) and stalls terminally
(13.4 m vs the old config's 104.1 m). Prime suspect: esc mode removed reverse
from the action space, so an OOB nose-in pose is unrecoverable.

## The work, in order

### 1. ESC reverse (transmitter-faithful)
In `newton/examples/vehicles/example_vehicle_mppi_track.py`'s esc mapping:
above a small speed threshold, negative drive = brake (current behavior);
at/below the threshold, sustained negative drive = REVERSE (negative
drive command, brake released) — like a real ESC's second-pull reverse.
Validate on the exact failure: 1300-frame hull s4 run (same seed as the
notes' video runs) must recover from the hairpin-1-exit OOB and keep racing.

### 2. Full-length re-validation (the real acceptance)
1300-frame paired runs, all 5 tracks (hull s4 tuning; bezier s0; bezier s9
rad=0.25 min_np=12 max_np=15; checkpoint s5 cp_count=18; repulsive s3
rep_growmin=3.0 rep_growmax=3.5). Arms: old config (`--brake-mode none`,
per-step) vs `esc + knots12 + reverse`. Every 240-frame verdict is suspect;
if the accepted arm loses anywhere, decompose (esc-only arm) and report
which ingredient regresses. Metrics per the notes (lap distance, ΔRMS
drive/steer, reversals, OOB fraction). Smoothness ≥ lap distance
(Decision 2), but a terminal stall is an automatic fail.

### 3. Task 3c acceptance at full length (if 1+2 succeed)
The anti-stall hardening implementation is in WIP commit `5ca8924a5`
(`_mppi_bench.py` + example changes). Finish its acceptance: 5-track
`_tsallis_q=1.2` table vs q=1.0, at 1300 frames. The q=1.2 smoothness win
(−9% drive / −18% steer at equal lap on 3 tracks) is the prize; the bezier
s0 crawl was its blocker.

## Conventions (binding)
- Paired same-session runs; one run at a time on the GPU; foreground only.
- Planner changes: private knobs, default-off bit-identical, CUDA-graph safe
  (see `newton/_src/vehicles/mppi.py` `_n_knots`/`_tsallis_q` precedent).
- unittest only; `uvx pre-commit run -a` before commits; imperative subjects;
  end commit messages with `Co-Authored-By:` + the agent's model name;
  `--no-gpg-sign`.
- Update the notes file tables + plan checkboxes as part of the commits.
- Append results/outcome to THIS file's Results section, then mark it done.

## Results (filled by the workstation agent)

_(pending)_
