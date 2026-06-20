#!/usr/bin/env python3
"""Inspect a MOve-style ReID split without importing torch.

Examples:
    python tools/inspect_move_dataset.py data.zip
    python tools/inspect_move_dataset.py data/move_eval_cam
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import tarfile
import zipfile

from safe_outputs import unique_path


PATTERN = re.compile(r"^([-\d]+)C(\d+)T(\d+)F(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)
SPLITS = ("train", "query", "gallery")
SPLIT_ALIASES = {
    "bounding_box_train": "train",
    "train": "train",
    "query": "query",
    "bounding_box_test": "gallery",
    "gallery": "gallery",
    "test": "gallery",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect MOve ReID query/gallery/train splits.")
    parser.add_argument("path", help="path to data.zip, move_eval_cam directory, or its parent data directory")
    parser.add_argument(
        "--dataset-dir",
        default="move_eval_cam",
        help="dataset folder name when inspecting a parent directory or zip",
    )
    parser.add_argument("--save-json", default="", help="optional path to save inspection details")
    return parser.parse_args()


def parse_image_name(name: str) -> tuple[int, int, int, int]:
    match = PATTERN.match(Path(name).name)
    if match is None:
        raise ValueError(f"Unexpected MOve image name: {name}")
    pid, camid, trackid, frameid = map(int, match.groups())
    return pid, camid, trackid, frameid


def split_from_parts(parts: tuple[str, ...], dataset_dir: str) -> str | None:
    if dataset_dir in parts:
        start = parts.index(dataset_dir) + 1
        if start < len(parts) and parts[start] in SPLIT_ALIASES:
            return SPLIT_ALIASES[parts[start]]
    for part in parts:
        if part in SPLIT_ALIASES:
            return SPLIT_ALIASES[part]
    return None


def collect_from_archive(names: list[str], dataset_dir: str) -> dict[str, list[str]]:
    items = {split: [] for split in SPLITS}
    for name in names:
        normalized = name.lstrip("./")
        if normalized.startswith("__MACOSX/"):
            continue
        suffix = Path(normalized).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            continue
        parts = tuple(Path(normalized).parts)
        split = split_from_parts(parts, dataset_dir)
        if split is not None:
            items[split].append(normalized)
    return items


def collect_from_zip(path: Path, dataset_dir: str) -> dict[str, list[str]]:
    with zipfile.ZipFile(path) as zf:
        return collect_from_archive(zf.namelist(), dataset_dir)


def collect_from_tar(path: Path, dataset_dir: str) -> dict[str, list[str]]:
    with tarfile.open(path) as tf:
        return collect_from_archive(tf.getnames(), dataset_dir)


def collect_from_dir(path: Path, dataset_dir: str) -> dict[str, list[str]]:
    root = path
    for candidate in (dataset_dir, "move_eval_cam", "MOVE", "move"):
        candidate_path = path / candidate
        if candidate_path.is_dir():
            root = candidate_path
            break
    items = {split: [] for split in SPLITS}
    for split in SPLITS:
        split_dir = root / split
        if split == "gallery" and not split_dir.is_dir():
            split_dir = root / "test"
        if not split_dir.is_dir():
            continue
        for img_path in split_dir.rglob("*"):
            if img_path.suffix.lower() in IMAGE_SUFFIXES:
                items[split].append(str(img_path))
    return items


def summarize_split(paths: list[str]) -> dict:
    pids = []
    cams = []
    tracks = []
    bad_names = []
    for path in sorted(paths):
        try:
            pid, camid, trackid, _ = parse_image_name(path)
        except ValueError:
            bad_names.append(path)
            continue
        if pid == -1:
            continue
        pids.append(pid)
        cams.append(camid)
        tracks.append(trackid)
    return {
        "num_images": len(paths),
        "num_ids": len(set(pids)),
        "num_cameras": len(set(cams)),
        "num_tracks": len(set(tracks)),
        "camera_counts": dict(sorted(Counter(cams).items())),
        "first_images": sorted(paths)[:5],
        "bad_names": bad_names[:20],
    }


def main() -> None:
    args = parse_args()
    path = Path(args.path)
    if path.suffix.lower() == ".zip":
        items = collect_from_zip(path, args.dataset_dir)
    elif path.name.endswith((".tar.gz", ".tgz", ".tar")):
        items = collect_from_tar(path, args.dataset_dir)
    else:
        items = collect_from_dir(path, args.dataset_dir)

    summary = {split: summarize_split(paths) for split, paths in items.items()}
    query_ids = {parse_image_name(path)[0] for path in items["query"] if PATTERN.match(Path(path).name)}
    gallery_ids = {parse_image_name(path)[0] for path in items["gallery"] if PATTERN.match(Path(path).name)}
    summary["query_gallery"] = {
        "shared_ids": len(query_ids & gallery_ids),
        "query_only_ids": len(query_ids - gallery_ids),
        "gallery_only_ids": len(gallery_ids - query_ids),
    }

    for split in SPLITS:
        row = summary[split]
        print(
            f"{split:<7} images={row['num_images']:<6} "
            f"ids={row['num_ids']:<5} cameras={row['num_cameras']:<3} "
            f"tracks={row['num_tracks']:<4} camera_counts={row['camera_counts']}"
        )
        if row["bad_names"]:
            print(f"  bad names: {row['bad_names'][:3]}")

    qg = summary["query_gallery"]
    print(
        "query/gallery ids: "
        f"shared={qg['shared_ids']} query_only={qg['query_only_ids']} gallery_only={qg['gallery_only_ids']}"
    )

    if args.save_json:
        out = unique_path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved inspection JSON to {out}")


if __name__ == "__main__":
    main()
