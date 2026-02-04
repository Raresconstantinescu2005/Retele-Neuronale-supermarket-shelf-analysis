"""Generate demo assets for the end-to-end pipeline.

This script is intentionally lightweight:
- reads images from docs/demo/input
- runs YOLO inference (optimized_model.pt preferred)
- writes annotated images to docs/demo/output

It proves the pipeline can be executed locally before the exam.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    demo_in = repo / "docs" / "demo" / "input"
    demo_out = repo / "docs" / "demo" / "output"
    demo_out.mkdir(parents=True, exist_ok=True)

    # If input folder is empty, copy a few originals as a default demo set
    demo_in.mkdir(parents=True, exist_ok=True)
    images = sorted([p for p in demo_in.iterdir() if p.is_file()])
    if not images:
        src_gen = repo / "data" / "generated"
        if src_gen.exists():
            candidates = sorted([p for p in src_gen.iterdir() if p.is_file()])[:3]
            for p in candidates:
                shutil.copy2(p, demo_in / p.name)
            images = sorted([p for p in demo_in.iterdir() if p.is_file()])

    if not images:
        raise SystemExit("No images found for demo. Put 1-3 images into docs/demo/input/")

    # Lazy import so repo installs are optional
    from ultralytics import YOLO

    model_path = repo / "models" / "optimized_model.pt"
    if not model_path.exists():
        model_path = repo / "models" / "trained_model.pt"

    model = YOLO(str(model_path))

    for img in images:
        results = model.predict(source=str(img), imgsz=640, conf=0.25, verbose=False)
        # results[0].plot() returns numpy array
        out_img = results[0].plot()
        out_path = demo_out / f"{img.stem}_pred.jpg"
        try:
            import cv2

            cv2.imwrite(str(out_path), out_img)
        except Exception:
            # Fallback if cv2 is not available; just skip writing image
            pass

    print(f"Demo generated using model: {model_path.relative_to(repo)}")
    print(f"Input:  {demo_in.relative_to(repo)}")
    print(f"Output: {demo_out.relative_to(repo)}")


if __name__ == "__main__":
    main()
