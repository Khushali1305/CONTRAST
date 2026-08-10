"""Check locally acquired upstream assets without downloading or redistributing them."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="JSON copied from assets.example.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    failures = []
    for name, item in config["assets"].items():
        path = Path(item["path"])
        if not path.exists():
            failures.append(f"{name}: missing {path}")
            continue
        expected = item.get("sha256")
        if expected and path.is_file() and sha256(path) != expected:
            failures.append(f"{name}: SHA-256 mismatch")
    if failures:
        raise SystemExit("\n".join(failures))
    print("All configured external assets are present; no asset was downloaded or copied.")


if __name__ == "__main__":
    main()

