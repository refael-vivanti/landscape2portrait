"""
evaluate.py — AI (VLM) crop-quality evaluation.

Builds a single 2x3 storyboard grid image (6 frames with their green crop-box
overlays) and asks a vision LLM to rate the crop quality. Results (`ai_score`,
`ai_reasoning`) are merged into the video's JSON metadata.

Two providers, auto-selected by whichever key is present (override with
`--provider`):

  * llama   — OpenAI-compatible endpoint, e.g. Meta's Llama API (recommended for
              Meta FTEs: high rate limits, keeps data on approved infra).
                export LLAMA_API_KEY=...
                # optional: LLAMA_BASE_URL (default https://api.llama.com/compat/v1)
  * openai  — any OpenAI-compatible endpoint.   export OPENAI_API_KEY=...
  * gemini  — Google Gemini.                    export GEMINI_API_KEY=...  (or GOOGLE_API_KEY)

Everything degrades gracefully: if no key/SDK is available or the call fails, the
metadata gets ai_score=None and an explanatory ai_reasoning instead of crashing.
"""

import base64
import json
import os
import re
import time

import cv2
import numpy as np

# Default vision-capable model per provider (override with --ai-model).
DEFAULT_MODELS = {
    "llama": "Llama-4-Maverick-17B-128E-Instruct-FP8",
    "openai": "gpt-4o",
    "gemini": "gemini-flash-latest",
}
LLAMA_DEFAULT_BASE_URL = "https://api.llama.com/compat/v1"


def _key(*names):
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


def resolve_provider(explicit=None):
    """Pick a provider: explicit if given, else the first with a key present."""
    if explicit and explicit != "auto":
        return explicit
    if _key("LLAMA_API_KEY"):
        return "llama"
    if _key("OPENAI_API_KEY"):
        return "openai"
    if _key("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        return "gemini"
    return "gemini"  # will report the missing-key reason

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


def _retryable(call, retries):
    """Run call() with exponential backoff on transient errors; fail fast on fatal."""
    last = "unknown error"
    for attempt in range(retries):
        try:
            return call()
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:180]}"
            msg = str(e) + type(e).__name__
            fatal = any(c in msg for c in ("401", "403", "404", "API key",
                                           "PERMISSION_DENIED", "NOT_FOUND",
                                           "INVALID_ARGUMENT", "API_KEY_INVALID",
                                           "AuthenticationError"))
            if not fatal and attempt < retries - 1:
                time.sleep(min(30, 4 * (2 ** attempt)))   # 4,8,16,30,30s
                continue
            break
    return None, f"error: {last}"


def _eval_gemini(grid_path, model, retries):
    key = _key("GEMINI_API_KEY", "GOOGLE_API_KEY")
    if not key:
        return None, "skipped: set GEMINI_API_KEY (or GOOGLE_API_KEY)"
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return None, "skipped: run `pip install google-genai`"
    client = genai.Client(api_key=key)
    img = open(grid_path, "rb").read()

    def call():
        resp = client.models.generate_content(
            model=model,
            contents=[types.Part.from_bytes(data=img, mime_type="image/png"), PROMPT])
        return _parse_json(resp.text)
    return _retryable(call, retries)


def _eval_openai_compatible(grid_path, model, base_url, api_key, retries):
    """Works for the Meta Llama API and any OpenAI-compatible vision endpoint."""
    try:
        from openai import OpenAI
    except ImportError:
        return None, "skipped: run `pip install openai`"
    client = OpenAI(base_url=base_url, api_key=api_key)
    b64 = base64.b64encode(open(grid_path, "rb").read()).decode()

    def call():
        resp = client.chat.completions.create(
            model=model, max_tokens=400,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}])
        return _parse_json(resp.choices[0].message.content)
    return _retryable(call, retries)


def ai_evaluate(grid_path, provider=None, model=None, retries=5):
    """Return (ai_score, ai_reasoning). ai_score is None when skipped/failed."""
    provider = resolve_provider(provider)
    model = model or DEFAULT_MODELS.get(provider)
    if provider == "gemini":
        return _eval_gemini(grid_path, model, retries)
    if provider == "llama":
        key = _key("LLAMA_API_KEY")
        if not key:
            return None, "skipped: set LLAMA_API_KEY to enable Llama API eval"
        base = os.environ.get("LLAMA_BASE_URL", LLAMA_DEFAULT_BASE_URL)
        return _eval_openai_compatible(grid_path, model, base, key, retries)
    if provider == "openai":
        key = _key("OPENAI_API_KEY")
        if not key:
            return None, "skipped: set OPENAI_API_KEY"
        return _eval_openai_compatible(grid_path, model,
                                       os.environ.get("OPENAI_BASE_URL"), key, retries)
    return None, f"skipped: unknown provider '{provider}'"


def evaluate_and_update(meta_path, out_root, provider=None, model=None):
    """Load a metadata JSON, build its grid, run the VLM, write ai_* fields back."""
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
        score, reasoning = ai_evaluate(grid_path, provider=provider, model=model)
        meta["ai_score"] = score
        meta["ai_reasoning"] = reasoning
        meta["ai_provider"] = resolve_provider(provider)
        meta["ai_grid"] = os.path.relpath(grid_path, out_root)
    with open(meta_path, "w") as fh:
        json.dump(meta, fh)
    return meta.get("ai_score"), meta.get("ai_reasoning")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="AI (VLM) crop-quality evaluation")
    ap.add_argument("--meta", required=True, help="path to a metadata JSON")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--provider", default="auto",
                    choices=["auto", "llama", "openai", "gemini"])
    ap.add_argument("--model", default=None, help="override the per-provider default")
    a = ap.parse_args()
    s, r = evaluate_and_update(a.meta, a.out, provider=a.provider, model=a.model)
    print(f"provider={resolve_provider(a.provider)}\nai_score={s}\nai_reasoning={r}")
