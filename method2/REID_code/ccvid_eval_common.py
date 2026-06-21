import re
from pathlib import Path

import numpy as np


TRACKLET_LINE_RE = re.compile(r"^(session\d+)/(\d{3})_(\d{2})$")


def parse_ccvid_list(list_path):
    items = []
    with open(list_path, "r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            tracklet = parts[0]
            pid = int(parts[1])
            clothes = parts[2] if len(parts) > 2 else ""
            match = TRACKLET_LINE_RE.fullmatch(tracklet)
            if not match:
                raise ValueError(f"Unexpected CCVID tracklet format: {tracklet}")
            session, pid_str, tracklet_id = match.groups()
            prefix = f"{session}_{pid_str}_{tracklet_id}"
            items.append(
                {
                    "tracklet": tracklet,
                    "prefix": prefix,
                    "session": session,
                    "pid": pid,
                    "tracklet_id": int(tracklet_id),
                    "clothes": clothes,
                }
            )
    return items


def build_tracklet_frames(data_root, split_name, items, max_frames=4):
    split_dir = Path(data_root) / split_name
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split dir: {split_dir}")

    prefix_index = {}
    for path in split_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        prefix = path.stem.rsplit("_", 1)[0]
        prefix_index.setdefault(prefix, []).append(path)

    for file_list in prefix_index.values():
        file_list.sort()

    tracklets = []
    missing = []

    for item in items:
        files = prefix_index.get(item["prefix"], [])
        if not files:
            missing.append(item["tracklet"])
            continue

        if max_frames and len(files) > max_frames:
            indices = np.linspace(0, len(files) - 1, max_frames, dtype=int)
            files = [files[idx] for idx in indices]

        tracklets.append(
            {
                "tracklet": item["tracklet"],
                "prefix": item["prefix"],
                "img_paths": [str(path) for path in files],
                "pid": item["pid"],
                "camid": session_tracklet_to_camid(item["session"], item["tracklet_id"]),
                "clothes": item["clothes"],
            }
        )

    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(f"Missing frames for {len(missing)} tracklets, first few: {preview}")

    return tracklets


def session_tracklet_to_camid(session, tracklet_id):
    if session == "session3":
        return tracklet_id + 11
    if session in {"session1", "session2"}:
        return tracklet_id - 1
    if session not in {"session1", "session2", "session3"}:
        raise ValueError(f"Unknown session: {session}")
    return tracklet_id - 1
