"""Build history/versions.json: per-video math% and stability for each full
algorithm snapshot, so the dashboard Overview can compare versions. Historical
snapshots are read straight from git; stability is recomputed from x_smooth so
it is comparable even for versions that predate the stored grade."""
import json, os, subprocess, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from cropper import stability_grade

# (label, git ref).  "WORKTREE" = current files on disk.
SNAPSHOTS = [("v4", "2a655a1"), ("v5", "WORKTREE")]


def load(ref, fname):
    if ref == "WORKTREE":
        p = os.path.join(ROOT, "metadata", fname)
        return json.load(open(p)) if os.path.exists(p) else None
    r = subprocess.run(["git", "-C", ROOT, "show", f"{ref}:metadata/{fname}"],
                       capture_output=True, text=True)
    return json.loads(r.stdout) if r.returncode == 0 else None


def metrics(m):
    xs = [f["x_smooth"] for f in m["frames"]]
    math = (m.get("quality", {}).get("avg_score")
            or (sum(m.get("frame_scores", [])) / len(m["frame_scores"])
                if m.get("frame_scores") else None))
    stab = stability_grade(xs, m["width"], m["fps"])
    ai = m.get("ai_score")
    return {"math": round(math * 100, 1) if math is not None else None,
            "stability": stab["score"], "grade": stab["grade"],
            "ai": ai if isinstance(ai, (int, float)) else None}


videos = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(ROOT, "metadata"))
                if f.endswith(".json"))
data = {}
for label, ref in SNAPSHOTS:
    d = {}
    for v in videos:
        m = load(ref, v + ".json")
        if m:
            d[v] = metrics(m)
    data[label] = d
    print(f"{label}: {len(d)} videos", flush=True)

out = {"versions": [s[0] for s in SNAPSHOTS], "videos": videos, "data": data}
os.makedirs(os.path.join(ROOT, "history"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "history", "versions.json"), "w"), indent=0)
print("wrote history/versions.json", flush=True)
