"""Checksum manifest for the reproduction artifacts.

``make manifest`` records SHA-256 for the generated caches; ``make verify`` checks them.
The point is to catch a silently changed artifact -- a re-downloaded corpus that is not
the same corpus, or a model file that is not the one the figure was rendered from.

A caveat worth stating plainly: these checksums record what *this* build produced. The
pipeline is fully seeded, so re-running it on the same machine reproduces the bytes -- but
a different platform or BLAS can yield different last-bit floating-point results and hence
a different digest, without anything being wrong. So treat a mismatch as "investigate",
not "corrupt", and take ``make test-repro`` (tolerance-based metric assertions) as the
authoritative reproduction check. This manifest's real job is catching a *swapped or
truncated* artifact.

Only artifacts that are genuinely byte-reproducible belong here. The corpus, the trained
encoder and the trained network all qualify: the encoder's dictionary learning and the
network's simulation are both fully seeded (see ``tests/test_determinism.py``). Rendered
figures do **not** -- matplotlib embeds timestamps and font state, so a PNG can differ
byte-wise while being pixel-identical.

Usage::

    python -m spiking_ven.manifest write  --manifest MANIFEST.sha256 --root .
    python -m spiking_ven.manifest verify --manifest MANIFEST.sha256 --root .
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

__all__ = ["sha256_file", "DEFAULT_ARTIFACTS", "write_manifest", "verify_manifest"]

# Relative to the repo root. Order is the pipeline order.
DEFAULT_ARTIFACTS = (
    "outputs/motifs.npz",
    "outputs/of_encoder.npz",
    "outputs/of_ven_model_k4max.npz",
)

_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(manifest: Path, root: Path, artifacts=DEFAULT_ARTIFACTS) -> int:
    lines, missing = [], []
    for rel in artifacts:
        p = root / rel
        if p.exists():
            lines.append(f"{sha256_file(p)}  {rel}")
            print(f"  {rel}: {lines[-1][:16]}...")
        else:
            missing.append(rel)
    if missing:
        print("  not present, so not recorded: " + ", ".join(missing))
    if not lines:
        print("Nothing to record. Run the pipeline first (`make figure`).")
        return 1
    manifest.write_text("\n".join(lines) + "\n")
    print(f"Wrote {manifest} ({len(lines)} artifact(s))")
    return 0


def verify_manifest(manifest: Path, root: Path) -> int:
    if not manifest.exists():
        print(f"{manifest} does not exist -- run `make manifest` to record checksums.")
        return 1
    ok = bad = absent = 0
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, rel = line.split(None, 1)
        p = root / rel.strip()
        if not p.exists():
            print(f"  MISSING  {rel.strip()}")
            absent += 1
            continue
        actual = sha256_file(p)
        if actual == digest:
            print(f"  ok       {rel.strip()}")
            ok += 1
        else:
            print(f"  CHANGED  {rel.strip()}\n             recorded {digest}\n"
                  f"             actual   {actual}")
            bad += 1
    print(f"\n{ok} ok, {bad} changed, {absent} missing")
    return 0 if (bad == 0 and absent == 0) else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["write", "verify"])
    p.add_argument("--manifest", default="MANIFEST.sha256")
    p.add_argument("--root", default=".")
    args = p.parse_args(argv)
    manifest, root = Path(args.manifest), Path(args.root)
    return (write_manifest(manifest, root) if args.action == "write"
            else verify_manifest(manifest, root))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
