# Mission Mars

This mission requires the Dog walkaround a certain area of a Mars surface and return to the base/spaceship; record each route attempt and save it as part of the main dataset, the Robotic Dog can learn from the previous attempts and improve itself

## Mars Dog Explorer — the robot dog that gets better

A 3D simulator of a robotic dog exploring a Mars surface: walk to four beacons and return to the lander.
Every route attempt is saved as JSON. Between attempts the dog turns each failure into a **versioned skill change**,
tests it on **five held-out terrains** the mission never uses, and **adopts it only if the held-out score improves —
otherwise it rolls back** (the rejected version stays on disk for audit).

## Run the demo (3 commands)
```bash
python3 mars.py reset        # clean slate
python3 mars.py run 20       # 20 mission attempts, learning between each (~6 s)
python3 mars.py serve        # opens http://localhost:8000/viewer.html  (3D replay)
```
In the viewer click **▶ Replay all attempts** and watch the dog fall on steep slopes early on, then complete the whole mission once the adopted skill versions take over.

## What gets written
| Path | Content |
|---|---|
| `attempts/attempt_NNN.json` | full route (x, y, height), events (fall, stuck in sand, target reached), outcome, score, skill used |
| `attempts/index.json` | summary of all attempts |
| `skills/vN.json` | every proposed skill version with its `decision` (ADOPTED / ROLLED BACK) |
| `skills/current.json` | the skill version in use now |
| `skills/log.json` | the learning log: change, held-out score before/after, decision |
| `memory.json` | hazard memory (fall spots, sand pits) the dog carries across attempts |
| `terrain.json` | the mission heightmap for the 3D viewer |

## How the self-improvement loop works
1. **Attempt** the mission on terrain seed 42 with the current skill (A* path planning over the heightmap, with parameters `max_slope`, `slope_weight`, `energy_reserve`, `replan_every`, `hazard_penalty`).
2. **Diagnose** the failure reason (fell, battery dead, no route, came home early) and **propose** one targeted parameter change → new version file.
3. **Evaluate** current vs. candidate on 5 held-out terrains × 2 runs (paired random seeds).
4. **Adopt** if the candidate is better, else **roll back**. Hazard memory (context) is kept either way.
