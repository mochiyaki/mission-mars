#!/usr/bin/env python3
"""
Mars Rover-Dog: a self-improving autonomous exploration simulator.

  python3 mars.py run 12     -> run 12 mission attempts, learning between them
  python3 mars.py reset      -> wipe attempts / skill versions / hazard memory
  python3 mars.py serve      -> serve the 3D viewer at http://localhost:8000

Every attempt is saved to attempts/attempt_NNN.json.
Every skill change is a versioned file in skills/vN.json and is adopted ONLY if
it beats the current version on held-out terrains; otherwise it is rolled back.
"""
import json, math, os, random, sys, heapq, time

ROOT = os.path.dirname(os.path.abspath(__file__))
ATT_DIR = os.path.join(ROOT, "attempts")
SK_DIR = os.path.join(ROOT, "skills")
N = 60                      # terrain grid size
MISSION_SEED = 42           # the real mission terrain
HELD_OUT_SEEDS = [7, 13, 21, 33, 55]   # never used for the mission; used to judge changes
FALL_POINT = 0.5                        # physics: the dog gets clumsy above this slope (unknown to the brain)

# ------------------------------------------------------------------ terrain
def make_terrain(seed):
    rnd = random.Random(seed)
    waves = [(rnd.uniform(0.08, 0.25), rnd.uniform(0.08, 0.25), rnd.uniform(0, 6.28), rnd.uniform(0.4, 1.2)) for _ in range(5)]
    craters = [(rnd.uniform(8, N - 8), rnd.uniform(8, N - 8), rnd.uniform(4, 8), rnd.uniform(0.8, 2.0)) for _ in range(6)]
    pits = set()
    while len(pits) < 70:    # hidden soft-sand pits (not visible to the dog until stepped on)
        pits.add((rnd.randrange(N), rnd.randrange(N)))
    h = [[0.0] * N for _ in range(N)]
    for y in range(N):
        for x in range(N):
            v = 0.0
            for fx, fy, ph, amp in waves:
                v += amp * math.sin(fx * x + ph) * math.cos(fy * y + ph * 0.7)
            for cx, cy, r, d in craters:
                dist = math.hypot(x - cx, y - cy)
                if dist < r:
                    v -= d * (1 - (dist / r) ** 2)
                elif dist < r * 1.4:
                    v += d * 0.35 * (1 - (dist - r) / (r * 0.4))
            h[y][x] = v
    base = (N // 2, N // 2)
    # four exploration targets in a ring around the base
    targets = [(int(base[0] + 18 * math.cos(a)), int(base[1] + 18 * math.sin(a))) for a in (0.6, 2.2, 3.8, 5.4)]
    # keep base and targets clear of pits
    for p in [base] + targets:
        pits.discard(p)
    return {"seed": seed, "h": h, "base": base, "targets": targets, "pits": pits}

def slope(t, a, b):
    return abs(t["h"][b[1]][b[0]] - t["h"][a[1]][a[0]]) / math.hypot(b[0] - a[0], b[1] - a[1])

NBRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]

# ------------------------------------------------------------------ dog brain
def astar(t, start, goal, skill, known_hazards):
    """Grid A*; cost = distance + slope_weight*slope; refuses slopes > max_slope and known hazards."""
    ms, sw = skill["max_slope"], skill["slope_weight"]
    openq = [(0, start)]
    g = {start: 0.0}
    came = {}
    while openq:
        _, cur = heapq.heappop(openq)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        for dx, dy in NBRS:
            nb = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nb[0] < N and 0 <= nb[1] < N):
                continue
            s = slope(t, cur, nb)
            if s > ms:
                continue
            step = math.hypot(dx, dy) * (1 + sw * s)
            if nb in known_hazards:
                step += skill["hazard_penalty"]      # remembered falls / sand pits are avoided, not forbidden
            ng = g[cur] + step
            if ng < g.get(nb, 1e18):
                g[nb] = ng
                came[nb] = cur
                heapq.heappush(openq, (ng + math.hypot(goal[0] - nb[0], goal[1] - nb[1]), nb))
    return None

def run_mission(t, skill, memory, rng, record=True):
    """Walk to every target then home. Returns an attempt record."""
    pos = t["base"]
    battery = 100.0
    known = set(tuple(p) for p in memory.get(str(t["seed"]), []))
    path_log = [list(pos)]
    events = []
    visited = []
    todo = list(t["targets"])
    goal_queue = []
    outcome, reason = "fail", "unknown"
    steps = 0
    while True:
        # pick next goal: nearest unvisited target, or home if reserve reached / all done
        if not todo or battery < skill["energy_reserve"]:
            if pos == t["base"]:
                skipped = sum(1 for e in events if e["type"] == "unreachable")
                if todo:
                    outcome, reason = "fail", "returned early with %d targets left" % len(todo)
                elif skipped:
                    outcome, reason = "partial", "home, but %d target unreachable" % skipped
                else:
                    outcome, reason = "success", "mission complete"
                break
            goal = t["base"]
        else:
            goal = min(todo, key=lambda p: math.hypot(p[0] - pos[0], p[1] - pos[1]))
        path = astar(t, pos, goal, skill, known)
        if path is None:
            if goal in todo:          # can't reach that target: skip it
                todo.remove(goal)
                events.append({"step": steps, "pos": list(pos), "type": "unreachable", "target": list(goal)})
                continue
            reason = "no path home"
            break
        # follow the path, replanning when something is learned
        for nxt in path[1:skill["replan_every"] + 1]:
            s = slope(t, pos, nxt)
            battery -= 0.35 + 2.2 * s
            steps += 1
            # fall risk grows with slope; the dog is clumsy above ~0.35
            p_fall = 0.3 / (1 + math.exp(-(s - FALL_POINT) * 40))
            if rng.random() < p_fall:
                known.add(nxt)
                events.append({"step": steps, "pos": list(nxt), "type": "fall", "slope": round(s, 3)})
                reason = "fell on slope %.2f" % s
                pos = nxt
                path_log.append(list(pos))
                outcome = "fail"
                return finish(t, skill, path_log, events, visited, battery, outcome, reason, known, memory)
            pos = nxt
            path_log.append(list(pos))
            if pos in t["pits"]:
                battery -= 6.0
                known.add(pos)
                events.append({"step": steps, "pos": list(pos), "type": "stuck_in_sand"})
                break                    # replan immediately
            if battery <= 0:
                reason = "battery dead"
                return finish(t, skill, path_log, events, visited, battery, "fail", reason, known, memory)
            if pos in todo:
                todo.remove(pos)
                visited.append(list(pos))
                events.append({"step": steps, "pos": list(pos), "type": "target_reached"})
                break
            if pos == goal:
                break
        if steps > 1500:
            reason = "timeout"
            break
    return finish(t, skill, path_log, events, visited, battery, outcome, reason, known, memory)

def finish(t, skill, path_log, events, visited, battery, outcome, reason, known, memory):
    # update hazard memory (context that persists across attempts)
    memory[str(t["seed"])] = sorted(list(p) for p in known)
    score = 25 * len(visited) + (40 if outcome == "success" else 20 if outcome == "partial" else 0) + max(0.0, battery) * 0.3
    return {"outcome": outcome, "reason": reason, "score": round(score, 1), "targets_visited": len(visited),
            "battery_left": round(max(0.0, battery), 1), "steps": len(path_log) - 1,
            "path": [[x, y, round(t["h"][y][x], 3)] for x, y in path_log], "events": events}

# ------------------------------------------------------------------ learner
DEFAULT_SKILL = {"version": 1, "max_slope": 0.9, "slope_weight": 1.0, "energy_reserve": 20.0,
                 "replan_every": 6, "hazard_penalty": 8.0, "note": "initial bold explorer"}

def propose(skill, reason, rng, next_version):
    """Turn a failure into a concrete, named change of the skill parameters."""
    s = dict(skill)
    if reason.startswith("fell"):
        pick = rng.choice(["max_slope", "slope_weight"])
        if pick == "max_slope":
            s["max_slope"] = round(max(0.15, skill["max_slope"] * 0.75), 3)
            change = "lower max_slope %.2f -> %.2f (fall)" % (skill["max_slope"], s["max_slope"])
        else:
            s["slope_weight"] = round(skill["slope_weight"] * 1.8, 2)
            change = "raise slope_weight %.2f -> %.2f (fall)" % (skill["slope_weight"], s["slope_weight"])
    elif reason.startswith("battery") or "returned early" in reason:
        if "returned early" in reason and rng.random() < 0.5:
            s["energy_reserve"] = round(max(5.0, skill["energy_reserve"] - 5), 1)
            change = "lower energy_reserve %.0f -> %.0f (came home too early)" % (skill["energy_reserve"], s["energy_reserve"])
        else:
            s["slope_weight"] = round(skill["slope_weight"] * 0.7, 2)
            change = "lower slope_weight %.2f -> %.2f (shorter routes to save battery)" % (skill["slope_weight"], s["slope_weight"])
    elif reason.startswith("no path") or reason == "timeout" or "unreachable" in reason:
        s["max_slope"] = round(min(1.2, skill["max_slope"] * 1.15), 3)
        change = "raise max_slope %.2f -> %.2f (could not find a route)" % (skill["max_slope"], s["max_slope"])
    else:   # success -> small exploratory tweak
        if rng.random() < 0.5:
            s["replan_every"] = max(2, min(12, skill["replan_every"] + rng.choice([-2, 2])))
            change = "tune replan_every %d -> %d" % (skill["replan_every"], s["replan_every"])
        else:
            s["hazard_penalty"] = round(skill["hazard_penalty"] * rng.choice([0.6, 1.6]), 1)
            change = "tune hazard_penalty %.1f -> %.1f" % (skill["hazard_penalty"], s["hazard_penalty"])
    s["version"] = next_version
    s["note"] = change
    return s

def evaluate(skill, memory, seed_base):
    """Held-out evaluation: average score over terrains the mission never sees."""
    total = 0.0
    mem_copy = json.loads(json.dumps(memory))
    for i, sd in enumerate(HELD_OUT_SEEDS):
        t = make_terrain(sd)
        for rep in range(2):
            r = run_mission(t, skill, mem_copy, random.Random(seed_base * 1000 + sd * 10 + rep))
            total += r["score"]
    return round(total / (len(HELD_OUT_SEEDS) * 2), 2)

# ------------------------------------------------------------------ io
def load(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default

def save(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)

def export_terrain(t):
    save(os.path.join(ROOT, "terrain.json"), {"n": N, "h": [[round(v, 3) for v in row] for row in t["h"]],
                                               "base": list(t["base"]), "targets": [list(p) for p in t["targets"]]})

def cmd_run(count):
    os.makedirs(ATT_DIR, exist_ok=True)
    os.makedirs(SK_DIR, exist_ok=True)
    t = make_terrain(MISSION_SEED)
    export_terrain(t)
    memory = load(os.path.join(ROOT, "memory.json"), {})
    skill = load(os.path.join(SK_DIR, "current.json"), DEFAULT_SKILL)
    if not os.path.exists(os.path.join(SK_DIR, "v1.json")):
        save(os.path.join(SK_DIR, "v1.json"), DEFAULT_SKILL)
    log = load(os.path.join(SK_DIR, "log.json"), [])
    index = load(os.path.join(ATT_DIR, "index.json"), [])
    print("%-4s %-4s %-8s %-6s %-4s %-7s %s" % ("att", "ver", "outcome", "score", "tgt", "battery", "reason"))
    for i in range(count):
        n = len(index) + 1
        rng = random.Random(n * 7919)
        r = run_mission(t, skill, memory, rng)
        r.update({"attempt": n, "skill_version": skill["version"], "skill": skill, "time": time.strftime("%H:%M:%S")})
        save(os.path.join(ATT_DIR, "attempt_%03d.json" % n), r)
        save(os.path.join(ROOT, "memory.json"), memory)
        print("%-4d v%-3d %-8s %-6.1f %-4d %-7.1f %s" % (n, skill["version"], r["outcome"], r["score"], r["targets_visited"], r["battery_left"], r["reason"]))

        # ---- self-improvement: propose, test on held-out terrains, adopt or roll back
        cand = propose(skill, r["reason"], rng, len(log) + 2)
        base_score = evaluate(skill, memory, n)
        cand_score = evaluate(cand, memory, n)
        adopted = cand_score > base_score + 0.5
        entry = {"after_attempt": n, "from_version": skill["version"], "candidate_version": cand["version"],
                 "change": cand["note"], "heldout_current": base_score, "heldout_candidate": cand_score,
                 "decision": "ADOPTED" if adopted else "ROLLED BACK"}
        log.append(entry)
        print("      learn: %-58s held-out %6.1f -> %6.1f  %s" % (cand["note"], base_score, cand_score, entry["decision"]))
        cand["decision"] = entry["decision"]
        save(os.path.join(SK_DIR, "v%d.json" % cand["version"]), cand)
        if adopted:
            skill = cand                      # new version becomes current
        # else: roll back = keep current skill untouched; rejected version stays on disk for audit
        save(os.path.join(SK_DIR, "current.json"), skill)
        save(os.path.join(SK_DIR, "log.json"), log)
        index.append({"attempt": n, "file": "attempts/attempt_%03d.json" % n, "outcome": r["outcome"], "score": r["score"],
                      "skill_version": r["skill_version"], "reason": r["reason"], "targets_visited": r["targets_visited"]})
        save(os.path.join(ATT_DIR, "index.json"), index)

def cmd_reset():
    import shutil
    for d in (ATT_DIR, SK_DIR):
        shutil.rmtree(d, ignore_errors=True)
    for f in ("memory.json", "terrain.json"):
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            os.remove(p)
    print("reset done")

def cmd_serve():
    import http.server, socketserver, webbrowser
    os.chdir(ROOT)
    handler = http.server.SimpleHTTPRequestHandler
    handler.extensions_map.update({".json": "application/json"})
    with socketserver.TCPServer(("", 8000), handler) as httpd:
        print("viewer: http://localhost:8000/viewer.html   (Ctrl+C to stop)")
        try:
            webbrowser.open("http://localhost:8000/viewer.html")
        except Exception:
            pass
        httpd.serve_forever()

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        cmd_run(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    elif cmd == "reset":
        cmd_reset()
    elif cmd == "serve":
        cmd_serve()
    else:
        print(__doc__)
