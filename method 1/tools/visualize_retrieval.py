#!/usr/bin/env python3
"""Visualize top-k ReID retrieval results from saved feature files.

The script can read images either from an extracted MOve directory or directly
from MOVE.tar.gz. It writes a new output directory and never overwrites an old
visualization run.
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
import random
import tarfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ensemble_reid import assert_same_eval_set, euclidean_distance, load_npz, normalize_dist, resolve_path
from postprocess_search import camera_pair_calibrate
from rerank_experiments import all_distance, re_ranking_from_all_dist
from safe_outputs import unique_path


METHOD_LABELS = {
    "a": "TransReID",
    "b": "CLIP-ReID",
    "fixed": "Fixed fusion",
    "rerank_b": "CLIP-ReID rerank",
    "camera_minmax": "Camera minmax",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create top-k retrieval visualization grids.")
    parser.add_argument("--a", required=True, help="first .npz feature file, usually TransReID")
    parser.add_argument("--b", required=True, help="second .npz feature file, usually CLIP-ReID")
    parser.add_argument("--name-a", default="TransReID")
    parser.add_argument("--name-b", default="CLIP-ReID")
    parser.add_argument("--image-root", default="", help="extracted image root, e.g. MOVE or .")
    parser.add_argument("--image-archive", default="", help="tar/tar.gz archive, e.g. MOVE.tar.gz")
    parser.add_argument(
        "--methods",
        default="a,b,fixed,rerank_b",
        help="comma-separated methods: a,b,fixed,rerank_b,camera_minmax",
    )
    parser.add_argument("--fixed-alpha", type=float, default=0.4)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument(
        "--select",
        choices=(
            "clip_success_trans_fail",
            "fusion_hurts_clip",
            "rerank_improves_clip",
            "camera_calib_improves_clip",
            "clip_fail",
            "first",
            "random",
        ),
        default="clip_success_trans_fail",
        help="which query examples to visualize",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="output/vis_0603", help="output directory")
    parser.add_argument("--k1", type=int, default=20)
    parser.add_argument("--k2", type=int, default=6)
    parser.add_argument("--lambda-value", type=float, default=0.3)
    return parser.parse_args()


class ImageStore:
    def __init__(self, image_root: str = "", image_archive: str = ""):
        self.root = resolve_path(image_root) if image_root else None
        self.archive_path = resolve_path(image_archive) if image_archive else None
        self.path_index: dict[str, Path] = {}
        self.tar_index: dict[str, str] = {}
        self.tar: tarfile.TarFile | None = None

        if self.root:
            for suffix in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                for path in self.root.rglob(suffix):
                    self.path_index.setdefault(path.name, path)

        if self.archive_path:
            self.tar = tarfile.open(self.archive_path)
            for member in self.tar.getmembers():
                if member.isfile() and Path(member.name).suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.tar_index.setdefault(Path(member.name).name, member.name)

    def close(self) -> None:
        if self.tar is not None:
            self.tar.close()

    def load(self, basename: str) -> Image.Image:
        if basename in self.path_index:
            return Image.open(self.path_index[basename]).convert("RGB")
        if self.tar is not None and basename in self.tar_index:
            extracted = self.tar.extractfile(self.tar_index[basename])
            if extracted is None:
                raise FileNotFoundError(basename)
            return Image.open(BytesIO(extracted.read())).convert("RGB")
        raise FileNotFoundError(f"Could not locate image {basename}; pass --image-root or --image-archive.")


def filtered_order(dist: np.ndarray, q_idx: int, meta: dict[str, np.ndarray]) -> np.ndarray:
    order = np.argsort(dist[q_idx])
    remove = (meta["g_pids"][order] == meta["q_pids"][q_idx]) & (
        meta["g_camids"][order] == meta["q_camids"][q_idx]
    )
    return order[~remove]


def top1_correct(dist: np.ndarray, q_idx: int, meta: dict[str, np.ndarray]) -> bool:
    order = filtered_order(dist, q_idx, meta)
    return bool(meta["g_pids"][order[0]] == meta["q_pids"][q_idx])


def choose_queries(
    dists: dict[str, np.ndarray],
    meta: dict[str, np.ndarray],
    mode: str,
    num_queries: int,
    seed: int,
) -> list[int]:
    all_indices = list(range(len(meta["q_pids"])))
    if mode == "first":
        candidates = all_indices
    elif mode == "random":
        candidates = all_indices[:]
        random.Random(seed).shuffle(candidates)
    elif mode == "clip_success_trans_fail":
        candidates = [
            i for i in all_indices if top1_correct(dists["b"], i, meta) and not top1_correct(dists["a"], i, meta)
        ]
    elif mode == "fusion_hurts_clip":
        candidates = [
            i
            for i in all_indices
            if top1_correct(dists["b"], i, meta) and not top1_correct(dists["fixed"], i, meta)
        ]
    elif mode == "rerank_improves_clip":
        candidates = [
            i
            for i in all_indices
            if top1_correct(dists["rerank_b"], i, meta) and not top1_correct(dists["b"], i, meta)
        ]
    elif mode == "camera_calib_improves_clip":
        candidates = [
            i
            for i in all_indices
            if top1_correct(dists["camera_minmax"], i, meta) and not top1_correct(dists["b"], i, meta)
        ]
    elif mode == "clip_fail":
        candidates = [i for i in all_indices if not top1_correct(dists["b"], i, meta)]
    else:
        raise ValueError(mode)

    if len(candidates) < num_queries:
        seen = set(candidates)
        candidates.extend(i for i in all_indices if i not in seen)
    return candidates[:num_queries]


def make_thumb(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    img = img.copy()
    img.thumbnail((size[0] - 8, size[1] - 28))
    x = (size[0] - img.width) // 2
    y = 22 + (size[1] - 28 - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def draw_cell(
    canvas: Image.Image,
    image: Image.Image,
    xy: tuple[int, int],
    size: tuple[int, int],
    border: str,
    title: str,
    subtitle: str,
) -> None:
    draw = ImageDraw.Draw(canvas)
    x, y = xy
    thumb = make_thumb(image, size)
    canvas.paste(thumb, xy)
    draw.rectangle([x, y, x + size[0] - 1, y + size[1] - 1], outline=border, width=4)
    draw.text((x + 6, y + 4), title, fill="black")
    draw.text((x + 6, y + size[1] - 20), subtitle, fill="black")


def render_query_grid(
    q_idx: int,
    methods: list[str],
    dists: dict[str, np.ndarray],
    meta: dict[str, np.ndarray],
    store: ImageStore,
    topk: int,
    out_path: Path,
    labels: dict[str, str],
) -> None:
    cell = (118, 228)
    pad = 12
    label_w = 170
    title_h = 44
    rows = len(methods)
    cols = topk + 1
    width = label_w + cols * cell[0] + (cols + 1) * pad
    height = title_h + rows * cell[1] + (rows + 1) * pad
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    q_pid = int(meta["q_pids"][q_idx])
    q_cam = int(meta["q_camids"][q_idx]) + 1
    q_path = str(meta["q_paths"][q_idx])
    draw.text((pad, 12), f"Query {q_idx} | pid={q_pid} cam=C{q_cam} | {q_path}", fill="black")

    for row, method in enumerate(methods):
        y = title_h + pad + row * (cell[1] + pad)
        draw.text((pad, y + cell[1] // 2 - 10), labels[method], fill="black")
        x0 = label_w + pad
        q_img = store.load(q_path)
        draw_cell(canvas, q_img, (x0, y), cell, "#2563eb", "Query", f"pid={q_pid} C{q_cam}")

        order = filtered_order(dists[method], q_idx, meta)[:topk]
        for rank, g_idx in enumerate(order, start=1):
            gx = x0 + rank * (cell[0] + pad)
            g_path = str(meta["g_paths"][g_idx])
            g_pid = int(meta["g_pids"][g_idx])
            g_cam = int(meta["g_camids"][g_idx]) + 1
            ok = g_pid == q_pid
            border = "#16a34a" if ok else "#dc2626"
            title = f"Top-{rank} {'OK' if ok else 'NO'}"
            subtitle = f"pid={g_pid} C{g_cam}"
            draw_cell(canvas, store.load(g_path), (gx, y), cell, border, title, subtitle)

    canvas.save(out_path)


def write_index(out_dir: Path, images: list[Path], title: str) -> None:
    html = ["<!doctype html><meta charset='utf-8'>", f"<title>{title}</title>", f"<h1>{title}</h1>"]
    for image in images:
        html.append(f"<h2>{image.name}</h2><img src='{image.name}' style='max-width:100%;height:auto'>")
    (out_dir / "index.html").write_text("\n".join(html), encoding="utf-8")


def main() -> None:
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = sorted(set(methods) - set(METHOD_LABELS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")

    a = load_npz(args.a)
    b = load_npz(args.b)
    assert_same_eval_set(a, b)

    dist_a = euclidean_distance(a["qf"], a["gf"])
    dist_b = euclidean_distance(b["qf"], b["gf"])
    fixed = args.fixed_alpha * normalize_dist(dist_a) + (1.0 - args.fixed_alpha) * normalize_dist(dist_b)

    dists = {"a": dist_a, "b": dist_b, "fixed": fixed}
    if "camera_minmax" in methods or args.select == "camera_calib_improves_clip":
        dists["camera_minmax"] = camera_pair_calibrate(dist_b, b["q_camids"], b["g_camids"], "minmax")
    if "rerank_b" in methods or args.select == "rerank_improves_clip":
        all_b = all_distance(b["qf"], b["gf"])
        dists["rerank_b"] = re_ranking_from_all_dist(
            all_b,
            len(b["qf"]),
            args.k1,
            args.k2,
            args.lambda_value,
        )

    labels = dict(METHOD_LABELS)
    labels["a"] = args.name_a
    labels["b"] = args.name_b
    labels["fixed"] = f"Fixed alpha={args.fixed_alpha:g}"
    labels["rerank_b"] = f"{args.name_b} rerank"
    labels["camera_minmax"] = f"{args.name_b} camera minmax"

    out_dir = unique_path(resolve_path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = choose_queries(dists, a, args.select, args.num_queries, args.seed)
    store = ImageStore(args.image_root, args.image_archive)
    written: list[Path] = []
    try:
        for q_idx in selected:
            q_pid = int(a["q_pids"][q_idx])
            q_path = Path(str(a["q_paths"][q_idx])).stem
            out_path = out_dir / f"query_{q_idx:04d}_pid_{q_pid}_{q_path}.png"
            render_query_grid(q_idx, methods, dists, a, store, args.topk, out_path, labels)
            written.append(out_path)
    finally:
        store.close()

    write_index(out_dir, written, f"Retrieval visualization: {args.select}")
    print(f"Saved {len(written)} visualizations to {out_dir}")
    print(f"Open {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
