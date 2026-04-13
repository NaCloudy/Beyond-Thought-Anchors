#!/usr/bin/env python3
"""
Download a rollout subdirectory from a Hugging Face dataset repo into a local target folder.

Default behavior is tailored for:
  source: deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/correct_base_solution
  target: math-rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/correct_base_solution

Example:
  python misc-scripts/download_rollouts_from_hf.py --repo-id uzaymacar/math-rollouts

If the repo is private, set HF_TOKEN env var or pass --hf-token.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Tuple


def _import_hf() -> Tuple[object, object]:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
        from huggingface_hub import login as hf_login  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit("Missing dependency: huggingface_hub. Install with: pip install huggingface_hub") from exc
    return snapshot_download, hf_login


def sync_tree(src: Path, dst: Path) -> Tuple[int, int]:
    """Copy files from src to dst, skipping files with the same size.

    Returns:
      copied_count, skipped_count
    """
    copied = 0
    skipped = 0
    for root, _, files in os.walk(src):
        root_path = Path(root)
        rel = root_path.relative_to(src)
        dst_root = dst / rel
        dst_root.mkdir(parents=True, exist_ok=True)

        for name in files:
            src_file = root_path / name
            dst_file = dst_root / name

            if dst_file.exists() and dst_file.stat().st_size == src_file.stat().st_size:
                skipped += 1
                continue

            shutil.copy2(src_file, dst_file)
            copied += 1

    return copied, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a rollout subdir from HF dataset repo to local target dir")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="uzaymacar/math-rollouts",
        required=True,
        help="HF dataset repo id, e.g. uzaymacar/math-rollouts",
    )
    parser.add_argument(
        "--source-subdir",
        type=str,
        default="deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/correct_base_solution",
        help="Subdirectory path inside HF dataset repo",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=r"math-rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/correct_base_solution",
        help="Local destination directory",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="HF repo revision (branch/tag/commit)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        help="HF token (optional for public repo)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional custom HF cache dir",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot_download, hf_login = _import_hf()

    if args.hf_token:
        hf_login(token=args.hf_token)

    print(f"[1/3] Downloading from repo: {args.repo_id}")
    allow_pattern = f"{args.source_subdir}/**"

    snapshot_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        token=args.hf_token,
        allow_patterns=[allow_pattern],
        resume_download=True,
        cache_dir=args.cache_dir,
    )

    src_dir = Path(snapshot_path) / args.source_subdir
    if not src_dir.exists():
        print(f"ERROR: source subdir not found in snapshot: {src_dir}")
        return 2

    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[2/3] Syncing files to: {target_dir}")
    copied, skipped = sync_tree(src_dir, target_dir)

    print("[3/3] Done")
    print(f"Copied: {copied}, Skipped(existing same size): {skipped}")
    print(f"Target: {target_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
