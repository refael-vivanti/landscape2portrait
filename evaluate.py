"""
evaluate.py — AI (VLM) crop-quality evaluation via Google Gemini.

Builds a single 2x3 storyboard grid image (6 frames with their green crop-box
overlays) and asks Gemini to rate the crop quality. Results (`ai_score`,
`ai_reasoning`) are merged into the video's JSON metadata.

Requires:
  pip install google-genai
  export GEMINI_API_KEY=...        (or GOOGLE_API_KEY)

Everything degrades gracefully: if the SDK or key is missing, or the API call
fails, the metadata gets ai_score=None and an explanatory ai_reasoning instead
of crashing the batch.
"""

import json
import os
import re

import cv2
import numpy as np

CROP_COLOR = (0, 255, 0)      # green (BGR)
FACE_COLOR = (255, 255, 0)    # cyan
PERSON_COLOR = (0, 255, 255)  # yellow

PROMPT = """You are an expert cinematic director assessing a smart-cropping algorithm (16:9 to 9:16).
Analyze the attached 6-frame storyboard. The green boxes represent the algorithm's crop window.
Rate the overall crop quality on a scale from 1.0 to 5.0. Focus on:
1. Subject Retention: Are the key actors, faces, or objects of interest fully preserved inside the crop?
2. Temporal Flow: Based on the green box positions, does the camera panning feel natural or abrupt?

Return your response STRICTLY as a JSON object with keys:
{
  "ai_score": float,
  "ai_reasoning": "string (brief 2-3 sentence explanation)"
}"""


def build_storyboard_grid(meta, out_root, save_path, cols=3, tile=(480, 270),
                          draw_boxes=True):
    """Compose the 6 storyboard frames (with green crop overlays) into one grid."""
    frames_by_i = {f["i"]: f for f in meta["frames"]}
    tiles = []
    for s in meta.get("storyboard", []):
        img = cv2.imread(os.path.join(out_root, s["frame"]))
        if img is None:
            continue
        fm = frames_by_i.get(s["i"], {})
        if draw_boxes:
            for b in fm.get("faces", []):
                x, y, w, h = b[:4]
                cv2.rectangle(img, (x, y), (x + w, y + h), FACE_COLOR, 2)
            for b in fm.get("people", []):
                x, y, w, h = b[:4]
                cv2.rectangle(img, (x, y), (x + w, y + h), PERSON_COLOR, 2)
        if "crop" in fm:
            x0, y0, x1, y1 = fm["crop"]
            cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), CROP_COLOR, 5)
        cv2.putText(img, f't={s["t"]}s', (12, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
        tiles.append(cv2.resize(img, tile))

    if not tiles:
        raise RuntimeError("no storyboard frames to build grid")
    while len(tiles) % cols != 0:
        tiles.append(np.zeros((tile[1], tile[0], 3), np.uint8))
    rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
    grid = np.vstack(rows)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, grid)
    return save_path


def _parse_json(text):
    """Pull the JSON object out of the model's reply (tolerates code fences)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in response: {text[:200]}")
    obj = json.loads(m.group(0))
    return float(obj["ai_score"]), str(obj.get("ai_reasoning", "")).strip()


def ai_evaluate(grid_path, model="gemini-2.5-flash"):
    """Return (ai_score, ai_reasoning). ai_score is None when evaluation is skipped."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return None, "skipped: set GEMINI_API_KEY (or GOOGLE_API_KEY) to enable AI eval"
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return None, "skipped: run `pip install google-genai`"
    try:
        client = genai.Client(api_key=key)
        img = open(grid_path, "rb").read()
        resp = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=img, mime_type="image/png"),
                PROMPT,
            ],
        )
        return _parse_json(resp.text)
    except Exception as e:  # network / quota / parse — never break the batch
        return None, f"error: {type(e).__name__}: {str(e)[:180]}"


def evaluate_and_update(meta_path, out_root, model="gemini-2.5-flash"):
    """Load a metadata JSON, build its grid, run Gemini, write ai_* fields back."""
    with open(meta_path) as fh:
        meta = json.load(fh)
    name = os.path.splitext(os.path.basename(meta_path))[0]
    grid_path = os.path.join(out_root, "frames", name, "_storyboard_grid.png")
    try:
        build_storyboard_grid(meta, out_root, grid_path)
    except Exception as e:
        meta["ai_score"] = None
        meta["ai_reasoning"] = f"grid error: {e}"
    else:
        score, reasoning = ai_evaluate(grid_path, model=model)
        meta["ai_score"] = score
        meta["ai_reasoning"] = reasoning
        meta["ai_grid"] = os.path.relpath(grid_path, out_root)
    with open(meta_path, "w") as fh:
        json.dump(meta, fh)
    return meta.get("ai_score"), meta.get("ai_reasoning")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="AI (Gemini) crop-quality evaluation")
    ap.add_argument("--meta", required=True, help="path to a metadata JSON")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--model", default="gemini-2.5-flash")
    a = ap.parse_args()
    s, r = evaluate_and_update(a.meta, a.out, model=a.model)
    print(f"ai_score={s}\nai_reasoning={r}")
