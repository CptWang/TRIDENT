#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import random
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

EMPTY_TOKENS = {"", "na", "n/a", "none", "null", "nan", "empty"}
COLOR_PALETTE = [
    "#ff2d55",
    "#1e90ff",
    "#00c853",
    "#ffb300",
    "#9c27b0",
    "#00acc1",
    "#fb8c00",
    "#8bc34a",
]


def _batch_mod():
    import run_batch_of_slides as batch_mod

    return batch_mod


def parse_bool_like(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def clean_path(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in EMPTY_TOKENS:
        return None
    return Path(text)


def default_coords_dir(coords_dir: str | None, mag: float, patch_size: int, overlap: int) -> str:
    if coords_dir is not None and str(coords_dir).strip():
        return str(coords_dir)
    return f"{mag}x_{patch_size}px_{overlap}px_overlap"


def ensure_supported_task(modality: str, task: str) -> None:
    if modality == "odo" and task != "feat":
        raise ValueError(
            "ODO modality only supports --task feat in v1, because ODO patches are derived from "
            "NODO geometry using paired coordinate mapping."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = _batch_mod().build_parser()
    parser.description = "Run TRIDENT multimodal extraction (NODO, ODO, or both) using a combined manifest."

    parser.add_argument(
        "--modality",
        type=str,
        choices=("nodo", "odo", "both"),
        default="both",
        help="Which modality flow to run.",
    )
    parser.add_argument(
        "--nodo-job-dir",
        type=str,
        default=None,
        help=(
            "Optional directory containing NODO outputs. "
            "Useful for --modality odo to reuse precomputed NODO coords from another run. "
            "Defaults to <job_dir>/NODO."
        ),
    )
    parser.add_argument("--wsi_nodo_column", type=str, default="wsi_nodo")
    parser.add_argument("--wsi_odo_column", type=str, default="wsi_odo")
    parser.add_argument("--sample_id_column", type=str, default="sample_id")

    parser.add_argument("--global_reg_column", type=str, default="global_registration")
    parser.add_argument("--elastic_reg_column", type=str, default="elastic_registration")
    parser.add_argument("--nodo_mpp", type=float, default=0.4049)
    parser.add_argument("--odo_mpp", type=float, default=2.2727)

    parser.add_argument(
        "--odo_space",
        type=str,
        choices=("auto", "native", "global", "registered"),
        default="auto",
        help="Target ODO coordinate space passed to paired mapping.",
    )
    parser.add_argument(
        "--global_direction",
        type=str,
        choices=("auto", "forward", "inverse"),
        default="auto",
        help="Global affine direction passed to paired mapping.",
    )
    parser.add_argument(
        "--f_ud_policy",
        type=str,
        choices=("auto", "apply", "ignore"),
        default="auto",
        help="f_ud handling policy passed to paired mapping.",
    )
    parser.add_argument(
        "--elastic_direction",
        type=str,
        choices=("auto", "forward", "inverse", "off"),
        default="auto",
        help="Elastic transform direction passed to paired mapping.",
    )
    parser.add_argument(
        "--disable_elastic",
        action="store_true",
        help="Disable elastic mapping when constructing paired ODO coordinates.",
    )
    parser.add_argument(
        "--pairing_allow_missing",
        action="store_true",
        help="Allow paired-mapping script to continue on sample errors.",
    )

    parser.add_argument("--qc_num_slides", type=int, default=5)
    parser.add_argument("--qc_total_patches", type=int, default=30)
    parser.add_argument("--qc_seed", type=int, default=7)
    parser.add_argument("--qc_thumb_max_side", type=int, default=1200)

    return parser


def require_manifest_columns(df: pd.DataFrame, required: list[str], csv_path: Path) -> None:
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")


def validate_sample_ids(df: pd.DataFrame, sample_col: str, csv_path: Path) -> pd.Series:
    if sample_col not in df.columns:
        raise ValueError(f"Missing sample column '{sample_col}' in {csv_path}")
    sample_ids = df[sample_col].fillna("").astype(str).str.strip()
    empty_mask = sample_ids == ""
    if bool(empty_mask.any()):
        bad_rows = (df.index[empty_mask] + 2).tolist()
        raise ValueError(f"Column '{sample_col}' has empty values at rows {bad_rows[:10]} in {csv_path}")
    dup_mask = sample_ids.duplicated(keep=False)
    if bool(dup_mask.any()):
        duplicates = sorted(set(sample_ids[dup_mask].tolist()))
        raise ValueError(
            f"Column '{sample_col}' must be unique for multimodal processing. "
            f"Duplicate values: {duplicates[:10]}"
        )
    return sample_ids


def validate_wsi_paths(df: pd.DataFrame, wsi_col: str, csv_path: Path) -> None:
    if wsi_col not in df.columns:
        raise ValueError(f"Missing WSI column '{wsi_col}' in {csv_path}")
    wsi_vals = df[wsi_col].fillna("").astype(str).str.strip()
    empty_mask = wsi_vals == ""
    if bool(empty_mask.any()):
        bad_rows = (df.index[empty_mask] + 2).tolist()
        raise ValueError(f"Column '{wsi_col}' has empty values at rows {bad_rows[:10]} in {csv_path}")


def create_modality_manifest(
    combined_df: pd.DataFrame,
    sample_ids: pd.Series,
    wsi_column: str,
    mpp: float,
    out_csv: Path,
) -> None:
    out_df = combined_df.copy()
    out_df["wsi"] = out_df[wsi_column].astype(str).str.strip()
    out_df["mpp"] = float(mpp)
    out_df["wsi_name"] = sample_ids.values
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)


def run_trident_job(args: argparse.Namespace) -> None:
    batch_mod = _batch_mod()
    processor = batch_mod.initialize_processor(args)
    original_task = args.task
    tasks = ["seg", "coords", "feat"] if args.task == "all" else [args.task]
    try:
        for task_name in tasks:
            args.task = task_name
            batch_mod.run_task(processor, args)
    finally:
        args.task = original_task
        try:
            processor.close()
        except Exception:
            pass


def read_coords_h5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if "coords" not in handle:
            raise ValueError(f"Missing 'coords' dataset in {path}")
        coords = np.asarray(handle["coords"][:], dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Coords must have shape (N,2), got {coords.shape} in {path}")
    return coords


def build_nodo_patch_manifest_from_coords(
    combined_df: pd.DataFrame,
    sample_col: str,
    wsi_nodo_col: str,
    nodo_patches_dir: Path,
    out_csv: Path,
) -> dict[str, int]:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    fieldnames = list(combined_df.columns)
    for col in ["sample_id", "x", "y"]:
        if col not in fieldnames:
            fieldnames.append(col)

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for _, row in combined_df.iterrows():
            sample_id = str(row[sample_col]).strip()
            wsi_nodo = str(row.get(wsi_nodo_col, "")).strip()
            wsi_nodo_name = Path(wsi_nodo).name if wsi_nodo else ""
            wsi_nodo_stem = wsi_nodo_name
            if wsi_nodo_stem.lower().endswith(".ome.zarr"):
                wsi_nodo_stem = wsi_nodo_stem[: -len(".ome.zarr")]
            elif wsi_nodo_stem.lower().endswith(".zarr"):
                wsi_nodo_stem = wsi_nodo_stem[: -len(".zarr")]
            else:
                wsi_nodo_stem = Path(wsi_nodo_stem).stem if wsi_nodo_stem else ""

            candidate_ids: list[str] = []
            for candidate in [sample_id, wsi_nodo_stem]:
                text = str(candidate).strip()
                if text and text not in candidate_ids:
                    candidate_ids.append(text)

            coords_h5 = None
            for candidate in candidate_ids:
                candidate_path = nodo_patches_dir / f"{candidate}_patches.h5"
                if candidate_path.exists():
                    coords_h5 = candidate_path
                    break
            if coords_h5 is None:
                tried = [str(nodo_patches_dir / f"{candidate}_patches.h5") for candidate in candidate_ids]
                raise FileNotFoundError(
                    f"Missing NODO coords for sample '{sample_id}'. Tried: {tried}"
                )
            coords = read_coords_h5(coords_h5)
            counts[sample_id] = int(coords.shape[0])

            base = {col: row[col] for col in combined_df.columns}
            base["sample_id"] = sample_id
            for x, y in coords.tolist():
                rec = dict(base)
                rec["x"] = int(x)
                rec["y"] = int(y)
                writer.writerow(rec)

    return counts


def run_paired_mapping(
    patch_manifest_csv: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "data" / "tissue" / "pairs" / "build_paired_coords_from_patch_manifest.py"
    if not script.exists():
        raise FileNotFoundError(f"Paired-mapping script not found: {script}")

    cmd = [
        sys.executable,
        str(script),
        "--manifest-csv",
        str(patch_manifest_csv),
        "--output-root",
        str(output_root),
        "--pair-target",
        "odo",
        "--sample-col",
        "sample_id",
        "--x-col",
        "x",
        "--y-col",
        "y",
        "--nodo-image-col",
        args.wsi_nodo_column,
        "--odo-image-col",
        args.wsi_odo_column,
        "--global-reg-col",
        args.global_reg_column,
        "--elastic-reg-col",
        args.elastic_reg_column,
        "--source-patch-size",
        str(args.patch_size),
        "--nodo-mpp",
        str(args.nodo_mpp),
        "--odo-mpp",
        str(args.odo_mpp),
        "--odo-space",
        args.odo_space,
        "--global-direction",
        args.global_direction,
        "--f-ud-policy",
        args.f_ud_policy,
        "--elastic-direction",
        args.elastic_direction,
        "--target-mag",
        str(int(round(args.mag))),
        "--overlap",
        str(int(args.overlap)),
    ]
    if args.disable_elastic:
        cmd.append("--disable-elastic")
    if args.pairing_allow_missing:
        cmd.append("--allow-missing")

    subprocess.run(cmd, check=True)


def collect_patch_counts(coords_patches_dir: Path, sample_ids: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample_id in sample_ids:
        coords_h5 = coords_patches_dir / f"{sample_id}_patches.h5"
        if not coords_h5.exists():
            raise FileNotFoundError(
                f"Missing mapped ODO coords for sample '{sample_id}': expected {coords_h5}"
            )
        counts[sample_id] = int(read_coords_h5(coords_h5).shape[0])
    return counts


def compare_patch_counts(
    nodo_counts: dict[str, int],
    odo_counts: dict[str, int],
    report_csv: Path,
) -> int:
    sample_ids = sorted(set(nodo_counts.keys()) | set(odo_counts.keys()))
    rows: list[dict[str, Any]] = []
    mismatches = 0

    for sample_id in sample_ids:
        n_count = int(nodo_counts.get(sample_id, -1))
        o_count = int(odo_counts.get(sample_id, -1))
        is_match = n_count == o_count
        if not is_match:
            mismatches += 1
        rows.append(
            {
                "sample_id": sample_id,
                "nodo_patch_count": n_count,
                "odo_patch_count": o_count,
                "delta_odo_minus_nodo": o_count - n_count,
                "is_match": bool(is_match),
            }
        )

    report_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(report_csv, index=False)
    return mismatches


def open_zarr_array(path: Path):
    try:
        import zarr
    except Exception as exc:
        raise RuntimeError(f"zarr is required to read .zarr inputs: {path}") from exc

    text = str(path)
    if ".zarr" not in text:
        raise ValueError(f"Path does not look like a zarr path: {path}")

    idx = text.index(".zarr")
    root_path = text[: idx + len(".zarr")]
    suffix = text[idx + len(".zarr") :].strip("/")

    obj = zarr.open(root_path, mode="r")
    if suffix:
        for token in suffix.split("/"):
            obj = obj[token]

    if hasattr(obj, "shape"):
        return obj

    for key in ["0", "s0", "data", "image"]:
        if key in obj:
            return obj[key]

    keys = list(getattr(obj, "keys", lambda: [])())
    if keys:
        numeric = [key for key in keys if str(key).isdigit()]
        if numeric:
            best = str(min(int(key) for key in numeric))
            return obj[best]
        return obj[sorted(keys)[0]]

    raise RuntimeError(f"Could not resolve zarr array from {path}")


def to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr)

    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
    elif x.ndim == 3:
        if x.shape[0] <= 4 and x.shape[1] > 4 and x.shape[2] > 4:  # CYX
            x = np.moveaxis(x, 0, 2)
        if x.shape[-1] == 1:
            x = np.repeat(x, 3, axis=2)
        elif x.shape[-1] >= 4:
            x = x[..., :3]
    else:
        raise ValueError(f"Unsupported image array shape: {x.shape}")

    if x.dtype == np.uint8:
        return x
    if x.dtype == np.bool_:
        return x.astype(np.uint8) * 255

    x = x.astype(np.float32)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=np.uint8)

    vals = x[finite]
    lo = float(np.percentile(vals, 1))
    hi = float(np.percentile(vals, 99))
    if hi <= lo:
        hi = lo + 1.0
    x = (np.clip(x, lo, hi) - lo) / (hi - lo)
    x = np.clip(x * 255.0, 0, 255)
    return x.astype(np.uint8)


def infer_spatial_size(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) == 2:
        return int(shape[1]), int(shape[0])
    if len(shape) == 3:
        if shape[0] <= 4 and shape[1] > 4 and shape[2] > 4:
            return int(shape[2]), int(shape[1])
        if shape[2] <= 4 and shape[0] > 4 and shape[1] > 4:
            return int(shape[1]), int(shape[0])
        return int(shape[-1]), int(shape[-2])
    raise ValueError(f"Unsupported shape for spatial size inference: {shape}")


def load_thumbnail(path: Path, max_side: int) -> tuple[np.ndarray, tuple[int, int]]:
    if max_side <= 0:
        raise ValueError("max_side must be > 0")

    if ".zarr" in str(path):
        arr = open_zarr_array(path)
        shape = tuple(int(v) for v in arr.shape)
        orig_w, orig_h = infer_spatial_size(shape)

        step = max(1, int(math.ceil(max(orig_w, orig_h) / float(max_side))))
        if len(shape) == 2:
            thumb = np.asarray(arr[::step, ::step])
        elif len(shape) == 3 and shape[0] <= 4 and shape[1] > 4 and shape[2] > 4:
            thumb = np.asarray(arr[:, ::step, ::step])
        elif len(shape) == 3:
            thumb = np.asarray(arr[::step, ::step, :])
        else:
            raise ValueError(f"Unsupported zarr shape for thumbnail: {shape}")
        return to_rgb_uint8(thumb), (orig_w, orig_h)

    with Image.open(path) as img:
        img = img.convert("RGB")
        orig_w, orig_h = img.size
        scale = min(1.0, max_side / float(max(orig_w, orig_h)))
        if scale < 1.0:
            tw = max(1, int(round(orig_w * scale)))
            th = max(1, int(round(orig_h * scale)))
            img = img.resize((tw, th), resample=Image.Resampling.BILINEAR)
        return np.asarray(img), (orig_w, orig_h)


def extract_patch(path: Path, x: int, y: int, patch_size: int) -> np.ndarray:
    if patch_size <= 0:
        raise ValueError(f"patch_size must be > 0, got {patch_size}")

    if ".zarr" in str(path):
        arr = open_zarr_array(path)
        shape = tuple(int(v) for v in arr.shape)
        if len(shape) == 2:
            patch = np.asarray(arr[y : y + patch_size, x : x + patch_size])
        elif len(shape) == 3 and shape[0] <= 4 and shape[1] > 4 and shape[2] > 4:
            patch = np.asarray(arr[:, y : y + patch_size, x : x + patch_size])
        elif len(shape) == 3:
            patch = np.asarray(arr[y : y + patch_size, x : x + patch_size, :])
        else:
            raise ValueError(f"Unsupported zarr shape for patch extraction: {shape}")
        patch_rgb = to_rgb_uint8(patch)
    else:
        with Image.open(path) as img:
            img = img.convert("RGB")
            patch_rgb = np.asarray(img.crop((x, y, x + patch_size, y + patch_size)))

    h, w = patch_rgb.shape[:2]
    if h == patch_size and w == patch_size:
        return patch_rgb

    padded = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    padded[:h, :w] = patch_rgb
    return padded


def draw_indexed_boxes(
    thumb_rgb: np.ndarray,
    orig_size: tuple[int, int],
    boxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    image = Image.fromarray(thumb_rgb)
    draw = ImageDraw.Draw(image)

    orig_w, orig_h = orig_size
    thumb_w, thumb_h = image.size
    sx = float(thumb_w) / float(max(orig_w, 1))
    sy = float(thumb_h) / float(max(orig_h, 1))
    thickness = max(2, int(round(min(thumb_w, thumb_h) * 0.004)))

    for local_idx, (x, y, patch_size, global_idx) in enumerate(boxes):
        color = COLOR_PALETTE[local_idx % len(COLOR_PALETTE)]
        x0 = int(round(x * sx))
        y0 = int(round(y * sy))
        x1 = int(round((x + patch_size) * sx))
        y1 = int(round((y + patch_size) * sy))

        for offset in range(thickness):
            draw.rectangle([x0 - offset, y0 - offset, x1 + offset, y1 + offset], outline=color)

        label = str(global_idx)
        text_pad = 3
        if hasattr(draw, "textbbox"):
            text_box = draw.textbbox((0, 0), label)
            tw = int(text_box[2] - text_box[0])
            th = int(text_box[3] - text_box[1])
        else:  # pragma: no cover - Pillow < 8 fallback
            tw, th = draw.textsize(label)
        lx0 = x0
        ly0 = max(0, y0 - (th + 2 * text_pad))
        lx1 = lx0 + tw + 2 * text_pad
        ly1 = ly0 + th + 2 * text_pad
        draw.rectangle([lx0, ly0, lx1, ly1], fill=color)
        draw.text((lx0 + text_pad, ly0 + text_pad), label, fill="white")

    return np.asarray(image)


def to_int_or_none(value: Any) -> int | None:
    try:
        v = float(value)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return int(round(v))


def draw_outline_boxes(
    thumb_rgb: np.ndarray,
    orig_size: tuple[int, int],
    boxes: list[tuple[int, int, int]],
    color: str,
) -> np.ndarray:
    if not boxes:
        return np.asarray(thumb_rgb)

    image = Image.fromarray(np.asarray(thumb_rgb))
    draw = ImageDraw.Draw(image)

    orig_w, orig_h = orig_size
    thumb_w, thumb_h = image.size
    sx = float(thumb_w) / float(max(orig_w, 1))
    sy = float(thumb_h) / float(max(orig_h, 1))
    thickness = max(2, int(round(min(thumb_w, thumb_h) * 0.004)))

    for x, y, patch_size in boxes:
        x0 = int(round(x * sx))
        y0 = int(round(y * sy))
        x1 = int(round((x + patch_size) * sx))
        y1 = int(round((y + patch_size) * sy))
        for offset in range(thickness):
            draw.rectangle([x0 - offset, y0 - offset, x1 + offset, y1 + offset], outline=color)

    return np.asarray(image)


def allocate_balanced(total: int, availability: dict[str, int]) -> dict[str, int]:
    keys = list(availability.keys())
    alloc = {k: 0 for k in keys}
    if total <= 0 or not keys:
        return alloc

    target_total = min(int(total), int(sum(max(0, v) for v in availability.values())))
    base = target_total // len(keys)
    rem = target_total % len(keys)

    for i, key in enumerate(keys):
        alloc[key] = min(max(0, availability[key]), base + (1 if i < rem else 0))

    assigned = sum(alloc.values())
    while assigned < target_total:
        progressed = False
        for key in keys:
            if alloc[key] < max(0, availability[key]):
                alloc[key] += 1
                assigned += 1
                progressed = True
                if assigned >= target_total:
                    break
        if not progressed:
            break

    return alloc


def build_patch_pair_tile(source_patch: np.ndarray, target_patch: np.ndarray, idx: int) -> Image.Image:
    size = 96
    src = Image.fromarray(source_patch).resize((size, size), resample=Image.Resampling.BILINEAR)
    tgt = Image.fromarray(target_patch).resize((size, size), resample=Image.Resampling.BILINEAR)
    tile = Image.new("RGB", (size * 2 + 8, size + 24), (245, 245, 245))
    tile.paste(src, (0, 24))
    tile.paste(tgt, (size + 8, 24))

    draw = ImageDraw.Draw(tile)
    draw.text((4, 4), f"#{idx}", fill="black")
    draw.text((4, 12), "NODO", fill="black")
    draw.text((size + 12, 12), "ODO", fill="black")
    return tile


def compose_tile_grid(tiles: list[Image.Image], cols: int = 5, pad: int = 8) -> Image.Image:
    if not tiles:
        return Image.new("RGB", (1, 1), (255, 255, 255))
    cols = max(1, min(cols, len(tiles)))
    rows = int(math.ceil(len(tiles) / float(cols)))

    tw, th = tiles[0].size
    grid_w = cols * tw + (cols + 1) * pad
    grid_h = rows * th + (rows + 1) * pad
    canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))

    for i, tile in enumerate(tiles):
        r = i // cols
        c = i % cols
        x = pad + c * (tw + pad)
        y = pad + r * (th + pad)
        canvas.paste(tile, (x, y))
    return canvas


def create_visual_confirmation(
    mapped_csv: Path,
    output_dir: Path,
    num_slides: int = 5,
    total_patches: int = 30,
    seed: int = 7,
    thumb_max_side: int = 1200,
) -> int:
    if num_slides <= 0:
        raise ValueError("num_slides must be > 0")
    if total_patches <= 0:
        raise ValueError("total_patches must be > 0")
    if not mapped_csv.exists():
        raise FileNotFoundError(f"Mapped CSV not found: {mapped_csv}")

    df = pd.read_csv(mapped_csv)
    required = [
        "sample_id",
        "source_x",
        "source_y",
        "mapped_x",
        "mapped_y",
        "source_image_path",
        "target_image_path",
        "source_patch_size",
        "target_patch_size",
        "is_valid",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Mapped CSV missing columns for visual confirmation: {missing}")

    usable = df[df["is_valid"].map(parse_bool_like)].copy()
    if usable.empty:
        raise ValueError("No valid rows in mapped CSV for visual confirmation.")

    by_sample = {sid: g.reset_index(drop=True) for sid, g in usable.groupby("sample_id")}
    sample_ids = list(by_sample.keys())
    rng = random.Random(seed)
    rng.shuffle(sample_ids)
    chosen_samples = sample_ids[: min(num_slides, len(sample_ids))]

    availability = {sid: int(len(by_sample[sid])) for sid in chosen_samples}
    allocations = allocate_balanced(total_patches, availability)
    total_selected = int(sum(allocations.values()))

    output_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = output_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    selected_records: list[dict[str, Any]] = []
    panel_records: list[dict[str, Any]] = []

    for sample_rank, sample_id in enumerate(chosen_samples, start=1):
        n_take = int(allocations.get(sample_id, 0))
        if n_take <= 0:
            continue
        sample_df = by_sample[sample_id]
        sampled = sample_df.sample(n=n_take, random_state=seed + sample_rank).reset_index(drop=True)

        source_image = clean_path(sampled.iloc[0]["source_image_path"])
        target_image = clean_path(sampled.iloc[0]["target_image_path"])
        if source_image is None or target_image is None:
            raise ValueError(f"Sample {sample_id}: missing source/target image paths in mapped CSV.")
        if not source_image.exists():
            raise FileNotFoundError(f"Sample {sample_id}: source image not found: {source_image}")
        if not target_image.exists():
            raise FileNotFoundError(f"Sample {sample_id}: target image not found: {target_image}")

        source_thumb, source_orig = load_thumbnail(source_image, max_side=thumb_max_side)
        target_thumb, target_orig = load_thumbnail(target_image, max_side=thumb_max_side)

        src_boxes: list[tuple[int, int, int, int]] = []
        tgt_boxes: list[tuple[int, int, int, int]] = []
        tiles: list[Image.Image] = []

        for i, row in sampled.iterrows():
            global_idx = i + 1
            source_x = int(round(float(row["source_x"])))
            source_y = int(round(float(row["source_y"])))
            target_x = int(round(float(row["mapped_x"])))
            target_y = int(round(float(row["mapped_y"])))
            source_patch_size = int(round(float(row["source_patch_size"])))
            target_patch_size = int(round(float(row["target_patch_size"])))

            src_boxes.append((source_x, source_y, source_patch_size, global_idx))
            tgt_boxes.append((target_x, target_y, target_patch_size, global_idx))

            source_patch = extract_patch(source_image, source_x, source_y, source_patch_size)
            target_patch = extract_patch(target_image, target_x, target_y, target_patch_size)
            tiles.append(build_patch_pair_tile(source_patch, target_patch, global_idx))

            selected_records.append(
                {
                    "sample_id": str(sample_id),
                    "index_in_sample": int(global_idx),
                    "source_image_path": str(source_image),
                    "target_image_path": str(target_image),
                    "source_x": source_x,
                    "source_y": source_y,
                    "target_x": target_x,
                    "target_y": target_y,
                    "source_patch_size": source_patch_size,
                    "target_patch_size": target_patch_size,
                }
            )

        source_overlay = draw_indexed_boxes(source_thumb, source_orig, src_boxes)
        target_overlay = draw_indexed_boxes(target_thumb, target_orig, tgt_boxes)
        source_img = Image.fromarray(source_overlay)
        target_img = Image.fromarray(target_overlay)

        top_h = max(source_img.height, target_img.height)
        if source_img.height != top_h:
            new_w = max(1, int(round(source_img.width * (top_h / float(source_img.height)))))
            source_img = source_img.resize((new_w, top_h), resample=Image.Resampling.BILINEAR)
        if target_img.height != top_h:
            new_w = max(1, int(round(target_img.width * (top_h / float(target_img.height)))))
            target_img = target_img.resize((new_w, top_h), resample=Image.Resampling.BILINEAR)

        top = Image.new("RGB", (source_img.width + target_img.width + 12, top_h), (255, 255, 255))
        top.paste(source_img, (0, 0))
        top.paste(target_img, (source_img.width + 12, 0))

        grid = compose_tile_grid(tiles, cols=5, pad=8)
        header_h = 32
        panel_w = max(top.width, grid.width)
        panel_h = header_h + top.height + 12 + grid.height
        panel = Image.new("RGB", (panel_w, panel_h), (255, 255, 255))
        draw = ImageDraw.Draw(panel)
        draw.text(
            (8, 8),
            f"sample={sample_id} | selected_patches={len(tiles)}",
            fill="black",
        )
        panel.paste(top, (0, header_h))
        panel.paste(grid, (0, header_h + top.height + 12))

        panel_path = panels_dir / f"{sample_rank:03d}_{sample_id}.png"
        panel.save(panel_path)
        panel_records.append(
            {
                "sample_id": str(sample_id),
                "panel_path": str(panel_path),
                "num_patches": int(len(tiles)),
            }
        )

    selected_csv = output_dir / "selected_patches.csv"
    panels_csv = output_dir / "panels_index.csv"
    pd.DataFrame(selected_records).to_csv(selected_csv, index=False)
    pd.DataFrame(panel_records).to_csv(panels_csv, index=False)
    return total_selected


def create_mismatch_visual_confirmation(
    mapped_csv: Path,
    count_report_csv: Path,
    output_dir: Path,
    thumb_max_side: int = 1200,
) -> int:
    if not mapped_csv.exists():
        raise FileNotFoundError(f"Mapped CSV not found: {mapped_csv}")
    if not count_report_csv.exists():
        raise FileNotFoundError(f"Count report CSV not found: {count_report_csv}")

    mapped_df = pd.read_csv(mapped_csv)
    count_df = pd.read_csv(count_report_csv)

    required_mapped = [
        "sample_id",
        "source_x",
        "source_y",
        "mapped_x",
        "mapped_y",
        "source_image_path",
        "target_image_path",
        "source_patch_size",
        "target_patch_size",
        "is_valid",
    ]
    missing_mapped = [c for c in required_mapped if c not in mapped_df.columns]
    if missing_mapped:
        raise ValueError(f"Mapped CSV missing columns for mismatch visualization: {missing_mapped}")

    required_count = ["sample_id", "nodo_patch_count", "odo_patch_count", "delta_odo_minus_nodo", "is_match"]
    missing_count = [c for c in required_count if c not in count_df.columns]
    if missing_count:
        raise ValueError(f"Count report CSV missing columns for mismatch visualization: {missing_count}")

    mismatch_df = count_df[~count_df["is_match"].map(parse_bool_like)].copy()
    if mismatch_df.empty:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "panels").mkdir(parents=True, exist_ok=True)
        pd.DataFrame([]).to_csv(output_dir / "mismatch_panels_index.csv", index=False)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = output_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    panel_records: list[dict[str, Any]] = []
    mismatch_samples = mismatch_df["sample_id"].astype(str).tolist()

    for sample_rank, sample_id in enumerate(mismatch_samples, start=1):
        sample_rows = mapped_df[mapped_df["sample_id"].astype(str) == str(sample_id)].copy()
        if sample_rows.empty:
            continue

        source_image = clean_path(sample_rows.iloc[0]["source_image_path"])
        target_image = clean_path(sample_rows.iloc[0]["target_image_path"])
        if source_image is None or target_image is None:
            continue
        if not source_image.exists() or not target_image.exists():
            continue

        source_thumb, source_orig = load_thumbnail(source_image, max_side=thumb_max_side)
        target_thumb, target_orig = load_thumbnail(target_image, max_side=thumb_max_side)

        src_boxes_indexed: list[tuple[int, int, int, int]] = []
        tgt_boxes_valid_indexed: list[tuple[int, int, int, int]] = []
        tgt_boxes_invalid_plain: list[tuple[int, int, int]] = []

        for local_idx, row in sample_rows.reset_index(drop=True).iterrows():
            idx = local_idx + 1
            source_x = to_int_or_none(row["source_x"])
            source_y = to_int_or_none(row["source_y"])
            mapped_x = to_int_or_none(row["mapped_x"])
            mapped_y = to_int_or_none(row["mapped_y"])
            source_patch_size = to_int_or_none(row["source_patch_size"])
            target_patch_size = to_int_or_none(row["target_patch_size"])
            is_valid = parse_bool_like(row["is_valid"])

            if source_x is not None and source_y is not None and source_patch_size is not None:
                src_boxes_indexed.append((source_x, source_y, max(1, source_patch_size), idx))

            if mapped_x is None or mapped_y is None or target_patch_size is None:
                continue
            if mapped_x < 0 or mapped_y < 0:
                continue

            if is_valid:
                tgt_boxes_valid_indexed.append((mapped_x, mapped_y, max(1, target_patch_size), idx))
            else:
                tgt_boxes_invalid_plain.append((mapped_x, mapped_y, max(1, target_patch_size)))

        source_overlay = draw_indexed_boxes(source_thumb, source_orig, src_boxes_indexed)
        target_overlay = draw_indexed_boxes(target_thumb, target_orig, tgt_boxes_valid_indexed)
        target_overlay = draw_outline_boxes(
            target_overlay,
            target_orig,
            tgt_boxes_invalid_plain,
            color="#ff3b30",
        )

        source_img = Image.fromarray(source_overlay)
        target_img = Image.fromarray(target_overlay)

        top_h = max(source_img.height, target_img.height)
        if source_img.height != top_h:
            new_w = max(1, int(round(source_img.width * (top_h / float(source_img.height)))))
            source_img = source_img.resize((new_w, top_h), resample=Image.Resampling.BILINEAR)
        if target_img.height != top_h:
            new_w = max(1, int(round(target_img.width * (top_h / float(target_img.height)))))
            target_img = target_img.resize((new_w, top_h), resample=Image.Resampling.BILINEAR)

        top = Image.new("RGB", (source_img.width + target_img.width + 12, top_h), (255, 255, 255))
        top.paste(source_img, (0, 0))
        top.paste(target_img, (source_img.width + 12, 0))

        count_row = mismatch_df[mismatch_df["sample_id"].astype(str) == str(sample_id)].iloc[0]
        nodo_count = int(count_row["nodo_patch_count"])
        odo_count = int(count_row["odo_patch_count"])
        delta = int(count_row["delta_odo_minus_nodo"])
        valid_rows = int(sample_rows["is_valid"].map(parse_bool_like).sum())
        total_rows = int(len(sample_rows))

        header_h = 44
        panel = Image.new("RGB", (top.width, header_h + top.height), (255, 255, 255))
        draw = ImageDraw.Draw(panel)
        draw.text(
            (8, 8),
            f"sample={sample_id} | nodo_count={nodo_count} odo_count={odo_count} delta={delta}",
            fill="black",
        )
        draw.text(
            (8, 24),
            f"rows_total={total_rows} mapped_valid={valid_rows} | red boxes on ODO = invalid mapped coords in report",
            fill="black",
        )
        panel.paste(top, (0, header_h))

        panel_path = panels_dir / f"{sample_rank:03d}_{sample_id}.png"
        panel.save(panel_path)

        panel_records.append(
            {
                "sample_id": str(sample_id),
                "panel_path": str(panel_path),
                "nodo_patch_count": nodo_count,
                "odo_patch_count": odo_count,
                "delta_odo_minus_nodo": delta,
                "rows_total": total_rows,
                "rows_valid": valid_rows,
            }
        )

    panels_csv = output_dir / "mismatch_panels_index.csv"
    pd.DataFrame(panel_records).to_csv(panels_csv, index=False)
    return len(panel_records)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_supported_task(args.modality, args.task)

    combined_manifest_csv = Path(args.custom_list_of_wsis) if args.custom_list_of_wsis is not None else None
    if combined_manifest_csv is None:
        raise ValueError("--custom_list_of_wsis is required and must point to a combined multimodal manifest CSV.")
    if not combined_manifest_csv.exists():
        raise FileNotFoundError(f"Combined manifest CSV not found: {combined_manifest_csv}")

    combined_df = pd.read_csv(combined_manifest_csv)
    if combined_df.empty:
        raise ValueError(f"Combined manifest CSV has no rows: {combined_manifest_csv}")

    sample_ids = validate_sample_ids(combined_df, args.sample_id_column, combined_manifest_csv)
    validate_wsi_paths(combined_df, args.wsi_nodo_column, combined_manifest_csv)
    if args.modality in ("odo", "both"):
        require_manifest_columns(
            combined_df,
            [args.wsi_odo_column, args.global_reg_column, args.elastic_reg_column],
            combined_manifest_csv,
        )
        validate_wsi_paths(combined_df, args.wsi_odo_column, combined_manifest_csv)

    job_root = Path(args.job_dir)
    nodo_job_dir = (
        Path(str(args.nodo_job_dir).strip())
        if args.nodo_job_dir is not None and str(args.nodo_job_dir).strip()
        else (job_root / "NODO")
    )
    odo_job_dir = job_root / "ODO"
    visual_confirmation_dir = job_root / "visual_confirmation"
    if args.modality in ("nodo", "both"):
        nodo_job_dir.mkdir(parents=True, exist_ok=True)
    if args.modality in ("odo", "both"):
        odo_job_dir.mkdir(parents=True, exist_ok=True)

    if args.modality in ("nodo", "both"):
        nodo_manifest_csv = nodo_job_dir / "manifest_trident_nodo_multimodal.csv"
        create_modality_manifest(
            combined_df=combined_df,
            sample_ids=sample_ids,
            wsi_column=args.wsi_nodo_column,
            mpp=float(args.nodo_mpp),
            out_csv=nodo_manifest_csv,
        )
        nodo_args = deepcopy(args)
        nodo_args.job_dir = str(nodo_job_dir)
        nodo_args.custom_list_of_wsis = str(nodo_manifest_csv)
        nodo_args.wsi_name_column = "wsi_name"
        run_trident_job(nodo_args)

    if args.modality in ("odo", "both"):
        coords_profile = default_coords_dir(args.coords_dir, args.mag, args.patch_size, args.overlap)
        nodo_coords_patches_dir = nodo_job_dir / coords_profile / "patches"
        if not nodo_coords_patches_dir.exists():
            raise FileNotFoundError(
                "NODO coords were not found for paired mapping. Expected directory: "
                f"{nodo_coords_patches_dir}. Run NODO coords first, set --nodo-job-dir to an existing "
                "NODO output directory, or run --modality both."
            )

        odo_profile_root = odo_job_dir / coords_profile
        patch_manifest_csv = odo_profile_root / "manifest_patch_level_nodo_for_pairing.csv"
        nodo_counts = build_nodo_patch_manifest_from_coords(
            combined_df=combined_df,
            sample_col=args.sample_id_column,
            wsi_nodo_col=args.wsi_nodo_column,
            nodo_patches_dir=nodo_coords_patches_dir,
            out_csv=patch_manifest_csv,
        )

        run_paired_mapping(
            patch_manifest_csv=patch_manifest_csv,
            output_root=odo_profile_root,
            args=args,
        )

        paired_report_csv = odo_profile_root / "paired_coord_mapping_odo.csv"
        mapped_manifest_csv = odo_profile_root / "manifest_trident_odo.csv"
        mapped_coords_patches_dir = odo_profile_root / "coords" / "patches"
        if not paired_report_csv.exists():
            raise FileNotFoundError(f"Expected paired mapping report not found: {paired_report_csv}")
        if not mapped_manifest_csv.exists():
            raise FileNotFoundError(f"Expected mapped ODO manifest not found: {mapped_manifest_csv}")

        odo_counts = collect_patch_counts(mapped_coords_patches_dir, list(nodo_counts.keys()))
        count_report_csv = odo_profile_root / "count_check_report.csv"
        mismatch_count = compare_patch_counts(
            nodo_counts=nodo_counts,
            odo_counts=odo_counts,
            report_csv=count_report_csv,
        )
        if mismatch_count > 0:
            mismatch_visual_dir = visual_confirmation_dir / "mismatch_cases"
            try:
                mismatch_panels = create_mismatch_visual_confirmation(
                    mapped_csv=paired_report_csv,
                    count_report_csv=count_report_csv,
                    output_dir=mismatch_visual_dir,
                    thumb_max_side=int(args.qc_thumb_max_side),
                )
                print(
                    "[MULTIMODAL] Wrote mismatch visualizations: "
                    f"{mismatch_panels} panel(s) at {mismatch_visual_dir}"
                )
            except Exception as mismatch_vis_exc:
                print(
                    "[MULTIMODAL][WARN] Failed to generate mismatch visualizations before raising: "
                    f"{type(mismatch_vis_exc).__name__}: {mismatch_vis_exc}",
                    file=sys.stderr,
                )
            raise ValueError(
                f"NODO/ODO patch count mismatch detected in {mismatch_count} samples. "
                f"See report: {count_report_csv}"
            )

        selected_qc_count = create_visual_confirmation(
            mapped_csv=paired_report_csv,
            output_dir=visual_confirmation_dir,
            num_slides=int(args.qc_num_slides),
            total_patches=int(args.qc_total_patches),
            seed=int(args.qc_seed),
            thumb_max_side=int(args.qc_thumb_max_side),
        )
        print(f"[MULTIMODAL] Wrote visual confirmation samples: {selected_qc_count}")

        odo_args = deepcopy(args)
        odo_args.task = "feat"
        odo_args.job_dir = str(odo_job_dir)
        odo_args.custom_list_of_wsis = str(mapped_manifest_csv)
        odo_args.coords_dir = f"{coords_profile}/coords"
        odo_args.wsi_name_column = "wsi_name"
        run_trident_job(odo_args)

    print(
        "[MULTIMODAL] Done. "
        f"modality={args.modality} job_dir={job_root} "
        f"nodo_dir={nodo_job_dir} odo_dir={odo_job_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
