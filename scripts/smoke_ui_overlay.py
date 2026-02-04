from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.web_interface.app import model, sorteaza_si_analizeaza, deseneaza_rezultat_custom


def main() -> int:
    uploads = REPO / "src" / "web_interface" / "static" / "uploads"
    if not uploads.exists():
        print(f"uploads folder not found: {uploads}")
        return 2

    imgs = list(uploads.glob("test_*.jpg"))
    if not imgs:
        imgs = [p for p in uploads.glob("*.jpg") if not p.name.startswith("res_") and not p.name.startswith("heat_")]

    if not imgs:
        print("No sample image found in uploads")
        return 3

    img = imgs[0]
    results = model(str(img))
    rap = sorteaza_si_analizeaza(results, str(img))
    out = deseneaza_rezultat_custom(str(img), results, rap)

    out_path = uploads / out
    print(f"input={img.name}")
    print(f"output={out}")
    print(f"exists={out_path.exists()}")
    print(f"shelves={len(rap)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
