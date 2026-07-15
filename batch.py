"""
batch.py — run the smart cropper over a folder, with progress logging.

Examples:
    python batch.py --folder /path/to/videos --limit 5          # first 5
    python batch.py --folder /path/to/videos --skip 5           # the rest
    nohup python batch.py --folder /path/to/videos > logs/batch.log 2>&1 &

Skips videos whose metadata JSON already exists so it is safe to resume.
"""

import argparse
import os
import time

from cropper import DEFAULT_ALPHA, DEFAULT_PROC_WIDTH, DEFAULT_FLOW_STRIDE, \
    DEFAULT_ANCHOR, TERMINAL_WINDOW, list_videos, process_video

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--out", default=ROOT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--proc-width", type=int, default=DEFAULT_PROC_WIDTH)
    ap.add_argument("--flow-stride", type=int, default=DEFAULT_FLOW_STRIDE)
    ap.add_argument("--detect-window", type=int, default=TERMINAL_WINDOW)
    ap.add_argument("--anchor", type=float, default=DEFAULT_ANCHOR)
    ap.add_argument("--full", action="store_true", help="detect every frame (slow)")
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "yolov8n.pt"))
    ap.add_argument("--force", action="store_true", help="reprocess even if JSON exists")
    ap.add_argument("--ai-eval", action="store_true",
                    help="run Gemini VLM evaluation after each video (needs GEMINI_API_KEY)")
    ap.add_argument("--ai-model", default="gemini-2.5-flash")
    args = ap.parse_args()

    names = list_videos(args.folder)[args.skip:]
    if args.limit:
        names = names[:args.limit]

    meta_dir = os.path.join(args.out, "metadata")
    os.makedirs(meta_dir, exist_ok=True)

    total = len(names)
    print(f"[batch] {total} videos from {args.folder}", flush=True)
    t0 = time.time()
    done = skipped = failed = 0
    for k, name in enumerate(names, 1):
        stem = os.path.splitext(name)[0]
        if not args.force and os.path.exists(os.path.join(meta_dir, stem + ".json")):
            print(f"[batch {k}/{total}] skip (exists) {name}", flush=True)
            skipped += 1
            continue
        print(f"[batch {k}/{total}] {name}", flush=True)
        try:
            process_video(os.path.join(args.folder, name), args.out,
                          alpha=args.alpha, proc_width=args.proc_width,
                          flow_stride=args.flow_stride,
                          detect_window=args.detect_window, anchor=args.anchor,
                          full=args.full, model_path=args.model)
            if args.ai_eval:
                from evaluate import evaluate_and_update
                meta_path = os.path.join(meta_dir, stem + ".json")
                s, r = evaluate_and_update(meta_path, args.out, model=args.ai_model)
                print(f"[batch {k}/{total}] AI score={s}", flush=True)
            done += 1
        except Exception as e:
            print(f"[batch {k}/{total}] ERROR {name}: {e}", flush=True)
            failed += 1

    dt = time.time() - t0
    print(f"[batch] complete: {done} done, {skipped} skipped, {failed} failed "
          f"in {dt/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
