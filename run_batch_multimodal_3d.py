#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TextIO


EMPTY_TOKENS = {"", "na", "n/a", "none", "null", "nan", "empty"}
DEPTH_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Adapt a 2.5D/3D image-level manifest for run_batch_multimodal.py by "
            "creating unique per-slice sample IDs. By default, NODO is run once per "
            "original sample and reused for all depth slices."
        ),
        add_help=True,
    )
    parser.add_argument("--custom_list_of_wsis", type=Path, required=True)
    parser.add_argument("--job_dir", type=Path, required=True)
    parser.add_argument("--sample_id_column", type=str, default="sample_id")
    parser.add_argument("--slice_depth_column", type=str, default="slice_depth")
    parser.add_argument("--trident_sample_id_column", type=str, default="sample_id_3d")
    parser.add_argument("--source_sample_id_column", type=str, default="source_sample_id")
    parser.add_argument(
        "--adapted_manifest_csv",
        type=Path,
        default=None,
        help=(
            "Optional path for the generated TRIDENT-safe manifest. "
            "Default: <job_dir>/manifest_trident_3D_adapted.csv"
        ),
    )
    parser.add_argument(
        "--source_manifest_csv",
        type=Path,
        default=None,
        help=(
            "Optional path for the generated source-only NODO manifest. "
            "Default: <job_dir>/manifest_trident_3D_nodo_source.csv"
        ),
    )
    parser.add_argument(
        "--adapt-only",
        action="store_true",
        help="Only write adapted manifests and print the commands that would be run.",
    )
    parser.add_argument(
        "--no-reuse-nodo",
        action="store_true",
        help=(
            "Disable optimized NODO reuse. Without this flag, --modality both runs "
            "NODO once per original sample, aliases NODO outputs per depth slice, "
            "then runs ODO on the full per-slice manifest."
        ),
    )
    parser.add_argument(
        "--alias-mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How to materialize per-slice NODO aliases. Default: symlink.",
    )
    parser.add_argument(
        "--strict-pair-counts",
        action="store_true",
        help=(
            "Keep the original multimodal NODO/ODO patch-count equality failure. "
            "By default, 2.5D runs continue to ODO feature extraction when depth slices "
            "produce fewer valid mapped ODO patches than their reused NODO source."
        ),
    )
    parser.add_argument(
        "--elastic_only_samples",
        action="append",
        default=[],
        help=(
            "Same meaning as run_batch_multimodal.py. Original sample IDs are expanded "
            "to all matching per-slice IDs in the adapted manifest."
        ),
    )
    parser.add_argument("--elastic_only_samples_file", type=Path, default=None)
    return parser


def is_empty(value: str | None) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in EMPTY_TOKENS


def safe_depth_token(depth: str) -> str:
    text = str(depth).strip()
    text = text.replace("-", "minus_").replace(".", "p")
    text = DEPTH_SAFE_RE.sub("_", text)
    return text.strip("_") or "unknown"


def make_slice_sample_id(sample_id: str, depth: str) -> str:
    if str(depth).strip() == "-99":
        return sample_id
    return f"{sample_id}__depth_{safe_depth_token(depth)}"


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def adapt_manifest(
    manifest_csv: Path,
    output_csv: Path,
    sample_col: str,
    slice_depth_col: str,
    trident_sample_col: str,
    source_sample_col: str,
) -> dict[str, list[str]]:
    fieldnames, rows = read_csv(manifest_csv)
    if sample_col not in fieldnames:
        raise ValueError(f"Missing sample column '{sample_col}' in {manifest_csv}")
    if slice_depth_col not in fieldnames:
        raise ValueError(f"Missing slice depth column '{slice_depth_col}' in {manifest_csv}")
    if not rows:
        raise ValueError(f"Manifest has no rows: {manifest_csv}")

    output_fieldnames = list(fieldnames)
    for column in (source_sample_col, trident_sample_col):
        if column not in output_fieldnames:
            output_fieldnames.append(column)

    by_source_sample: dict[str, list[str]] = {}
    seen_trident_ids: set[str] = set()
    adapted_rows: list[dict[str, str]] = []
    for row_index, row in enumerate(rows, start=2):
        sample_id = str(row.get(sample_col, "")).strip()
        depth = str(row.get(slice_depth_col, "")).strip()
        if not sample_id:
            raise ValueError(f"Empty sample ID at row {row_index} in {manifest_csv}")
        if is_empty(depth):
            raise ValueError(f"Empty slice depth at row {row_index} in {manifest_csv}")

        trident_sample_id = make_slice_sample_id(sample_id, depth)
        if trident_sample_id in seen_trident_ids:
            raise ValueError(
                f"Duplicate adapted sample ID '{trident_sample_id}' from row {row_index}. "
                "Check sample_id/slice_depth uniqueness."
            )
        seen_trident_ids.add(trident_sample_id)
        by_source_sample.setdefault(sample_id, []).append(trident_sample_id)

        out = dict(row)
        out[source_sample_col] = sample_id
        out[trident_sample_col] = trident_sample_id
        adapted_rows.append(out)

    write_csv(output_csv, output_fieldnames, adapted_rows)
    return by_source_sample


def write_source_manifest(
    manifest_csv: Path,
    output_csv: Path,
    sample_col: str,
    slice_depth_col: str,
    trident_sample_col: str,
    source_sample_col: str,
) -> None:
    fieldnames, rows = read_csv(manifest_csv)
    if sample_col not in fieldnames:
        raise ValueError(f"Missing sample column '{sample_col}' in {manifest_csv}")
    if slice_depth_col not in fieldnames:
        raise ValueError(f"Missing slice depth column '{slice_depth_col}' in {manifest_csv}")

    output_fieldnames = list(fieldnames)
    for column in (source_sample_col, trident_sample_col):
        if column not in output_fieldnames:
            output_fieldnames.append(column)

    best_by_sample: dict[str, tuple[int, int, dict[str, str]]] = {}
    for row_index, row in enumerate(rows):
        sample_id = str(row.get(sample_col, "")).strip()
        depth = str(row.get(slice_depth_col, "")).strip()
        if not sample_id:
            continue
        priority = 0 if depth == "-99" else 1
        existing = best_by_sample.get(sample_id)
        if existing is None or (priority, row_index) < (existing[0], existing[1]):
            out = dict(row)
            out[source_sample_col] = sample_id
            out[trident_sample_col] = sample_id
            best_by_sample[sample_id] = (priority, row_index, out)

    source_rows = [item[2] for item in sorted(best_by_sample.values(), key=lambda item: item[2][trident_sample_col])]
    if not source_rows:
        raise ValueError(f"No source rows could be selected from {manifest_csv}")
    write_csv(output_csv, output_fieldnames, source_rows)


def read_elastic_only_samples(values: list[str], path: Path | None) -> list[str]:
    samples: list[str] = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if token:
                samples.append(token)
    if path is not None:
        for line in path.read_text().splitlines():
            token = line.strip()
            if token and not token.startswith("#"):
                samples.append(token)
    return samples


def expand_elastic_only_samples(
    requested: list[str],
    by_source_sample: dict[str, list[str]],
) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()
    all_adapted = {sample_id for ids in by_source_sample.values() for sample_id in ids}
    for token in requested:
        candidates = by_source_sample.get(token)
        if candidates is None and token in all_adapted:
            candidates = [token]
        if candidates is None:
            candidates = [token]
        for candidate in candidates:
            if candidate not in seen:
                expanded.append(candidate)
                seen.add(candidate)
    return expanded


def get_repeated_option(args: list[str], option: str) -> list[str]:
    values: list[str] = []
    idx = 0
    while idx < len(args):
        token = args[idx]
        if token == option and idx + 1 < len(args):
            values.append(args[idx + 1])
            idx += 2
            continue
        if token.startswith(f"{option}="):
            values.append(token.split("=", 1)[1])
        idx += 1
    return values


def get_option(args: list[str], option: str, default: str | None = None) -> str | None:
    values = get_repeated_option(args, option)
    return values[-1] if values else default


def normalized_modality(unknown_args: list[str]) -> str:
    return str(get_option(unknown_args, "--modality", "both")).strip().lower()


def normalized_task(unknown_args: list[str]) -> str:
    return str(get_option(unknown_args, "--task", "seg")).strip().lower()


def default_coords_profile(unknown_args: list[str]) -> str:
    coords_dir = get_option(unknown_args, "--coords_dir")
    if coords_dir:
        return coords_dir
    mag = float(get_option(unknown_args, "--mag", "20") or "20")
    patch_size = int(float(get_option(unknown_args, "--patch_size", "512") or "512"))
    overlap = int(float(get_option(unknown_args, "--overlap", "0") or "0"))
    return f"{mag}x_{patch_size}px_{overlap}px_overlap"


def patch_encoder_name(unknown_args: list[str]) -> str:
    return str(get_option(unknown_args, "--patch_encoder", "conch_v15") or "conch_v15")


def has_flag(args: list[str], option: str) -> bool:
    return option in args or any(token.startswith(f"{option}=") for token in args)


def append_option_if_present(cmd: list[str], args: list[str], option: str) -> None:
    value = get_option(args, option)
    if value is not None:
        cmd.extend([option, value])


def parse_cuda_visible_devices(raw_value: str | None = None) -> list[str]:
    value = raw_value if raw_value is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return []
    tokens = [token.strip() for token in str(value).split(",")]
    return [token for token in tokens if token]


def round_robin_row_partitions(rows: list[dict[str, str]], num_partitions: int) -> list[list[dict[str, str]]]:
    if num_partitions <= 0:
        raise ValueError("num_partitions must be > 0")
    partitions: list[list[dict[str, str]]] = [[] for _ in range(num_partitions)]
    for row_idx, row in enumerate(rows):
        partitions[row_idx % num_partitions].append(row)
    return partitions


def mapped_odo_paths(job_dir: Path, coords_profile: str) -> tuple[Path, Path, Path]:
    odo_profile_root = job_dir / "ODO" / coords_profile
    return (
        odo_profile_root / "manifest_trident_odo.csv",
        odo_profile_root / "paired_coord_mapping_odo.csv",
        odo_profile_root / "coords" / "patches",
    )


def build_direct_feature_command(
    unknown_args: list[str],
    *,
    job_dir: Path,
    manifest_csv: Path,
    coords_dir: str,
) -> list[str]:
    runner = Path(__file__).with_name("run_batch_of_slides.py")
    cmd = [
        sys.executable,
        str(runner),
        "--task",
        "feat",
        "--job_dir",
        str(job_dir),
        "--wsi_dir",
        str(get_option(unknown_args, "--wsi_dir", "/") or "/"),
        "--custom_list_of_wsis",
        str(manifest_csv),
        "--coords_dir",
        coords_dir,
        "--wsi_name_column",
        "wsi_name",
        "--segmentation_source",
        "model",
    ]

    for option in [
        "--reader_type",
        "--mag",
        "--patch_size",
        "--overlap",
        "--patch_encoder",
        "--patch_encoder_ckpt_path",
        "--batch_size",
        "--feat_batch_size",
        "--max_workers",
        "--wsi_cache",
        "--cache_batch_size",
    ]:
        append_option_if_present(cmd, unknown_args, option)

    if has_flag(unknown_args, "--skip_errors"):
        cmd.append("--skip_errors")
    if has_flag(unknown_args, "--search_nested"):
        cmd.append("--search_nested")

    return cmd


def run_sharded_feature_extraction(
    *,
    stage_name: str,
    manifest_csv: Path,
    job_dir: Path,
    coords_dir: str,
    unknown_args: list[str],
    gpu_tokens: list[str],
) -> None:
    if len(gpu_tokens) <= 1:
        raise ValueError("run_sharded_feature_extraction requires at least 2 GPU tokens")

    fieldnames, rows = read_csv(manifest_csv)
    if not rows:
        raise ValueError(f"Cannot shard empty feature manifest: {manifest_csv}")

    shard_root = job_dir / "_multigpu_shards" / stage_name
    shard_root.mkdir(parents=True, exist_ok=True)
    partitions = round_robin_row_partitions(rows, len(gpu_tokens))

    workers: list[tuple[subprocess.Popen, list[str], Path, TextIO]] = []
    for shard_idx, (gpu_token, shard_rows) in enumerate(zip(gpu_tokens, partitions)):
        if not shard_rows:
            continue
        shard_csv = shard_root / f"manifest_shard_{shard_idx:02d}.csv"
        write_csv(shard_csv, fieldnames, shard_rows)

        worker_cmd = build_direct_feature_command(
            unknown_args=unknown_args,
            job_dir=job_dir,
            manifest_csv=shard_csv,
            coords_dir=coords_dir,
        )
        worker_cmd.extend(["--gpu", "0"])

        worker_env = os.environ.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = gpu_token

        log_path = shard_root / f"worker_{shard_idx:02d}_gpu_{gpu_token}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        log_handle.write(f"# cmd: {' '.join(worker_cmd)}\n")
        log_handle.flush()

        print(
            f"[3D] Launching shard stage={stage_name} shard={shard_idx} gpu={gpu_token} "
            f"rows={len(shard_rows)} manifest={shard_csv}"
        )
        proc = subprocess.Popen(
            worker_cmd,
            env=worker_env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        workers.append((proc, worker_cmd, log_path, log_handle))

    failures: list[tuple[int, list[str], Path]] = []
    for proc, worker_cmd, log_path, log_handle in workers:
        try:
            return_code = proc.wait()
        finally:
            log_handle.close()
        if return_code != 0:
            failures.append((return_code, worker_cmd, log_path))

    if failures:
        return_code, failed_cmd, failed_log = failures[0]
        print(
            f"[3D][ERR] Sharded feature worker failed rc={return_code}. "
            f"See log: {failed_log}",
            file=sys.stderr,
        )
        raise subprocess.CalledProcessError(return_code, failed_cmd)


def run_odo_with_2d_count_mismatch_tolerance(
    *,
    odo_cmd: list[str],
    unknown_args: list[str],
    args: argparse.Namespace,
    run_direct_feat: bool = True,
) -> Path:
    coords_profile = default_coords_profile(unknown_args)
    mapped_manifest_csv, paired_report_csv, mapped_coords_patches_dir = mapped_odo_paths(
        args.job_dir,
        coords_profile,
    )

    if mapped_manifest_csv.exists() and mapped_coords_patches_dir.is_dir():
        print(
            "[3D] Reusing existing mapped ODO coords after a previous mapping run: "
            f"{mapped_manifest_csv}"
        )
    else:
        result = subprocess.run(odo_cmd, check=False)
        if result.returncode == 0:
            return mapped_manifest_csv
        if args.strict_pair_counts:
            raise subprocess.CalledProcessError(result.returncode, odo_cmd)
        if not mapped_manifest_csv.exists() or not paired_report_csv.exists() or not mapped_coords_patches_dir.is_dir():
            raise subprocess.CalledProcessError(result.returncode, odo_cmd)
        print(
            "[3D][WARN] ODO paired mapping command exited non-zero after writing mapped "
            "coords/manifest. This is expected for 2.5D when depth slices have fewer valid "
            "ODO patches than the reused NODO source. Continuing with ODO feature extraction."
        )

    if not run_direct_feat:
        return mapped_manifest_csv

    feat_cmd = build_direct_feature_command(
        unknown_args=unknown_args,
        job_dir=args.job_dir / "ODO",
        manifest_csv=mapped_manifest_csv,
        coords_dir=f"{coords_profile}/coords",
    )
    append_option_if_present(feat_cmd, unknown_args, "--gpu")
    print("[3D] ODO direct feature command:")
    print(" ".join(feat_cmd))
    subprocess.run(feat_cmd, check=True)
    return mapped_manifest_csv


def materialize_alias(src: Path, dst: Path, mode: str) -> bool:
    if not src.exists():
        return False
    if dst.exists() or dst.is_symlink():
        try:
            if dst.resolve() == src.resolve():
                return False
        except FileNotFoundError:
            pass
        raise FileExistsError(f"Alias destination already exists and points elsewhere: {dst}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        relative_src = os.path.relpath(src, start=dst.parent)
        dst.symlink_to(relative_src)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported alias mode: {mode}")
    return True


def materialize_nodo_aliases(
    *,
    job_dir: Path,
    coords_profile: str,
    patch_encoder: str,
    by_source_sample: dict[str, list[str]],
    alias_mode: str,
) -> dict[str, int]:
    nodo_profile_root = job_dir / "NODO" / coords_profile
    alias_counts = {"patches": 0, "features": 0, "slide_features": 0}
    feature_dirs = [
        nodo_profile_root / f"features_{patch_encoder}",
        nodo_profile_root / f"slide_features_{patch_encoder}",
    ]

    for source_sample_id, adapted_ids in by_source_sample.items():
        for adapted_id in adapted_ids:
            if adapted_id == source_sample_id:
                continue

            if materialize_alias(
                nodo_profile_root / "patches" / f"{source_sample_id}_patches.h5",
                nodo_profile_root / "patches" / f"{adapted_id}_patches.h5",
                alias_mode,
            ):
                alias_counts["patches"] += 1

            if materialize_alias(
                feature_dirs[0] / f"{source_sample_id}.h5",
                feature_dirs[0] / f"{adapted_id}.h5",
                alias_mode,
            ):
                alias_counts["features"] += 1

            if materialize_alias(
                feature_dirs[1] / f"{source_sample_id}.h5",
                feature_dirs[1] / f"{adapted_id}.h5",
                alias_mode,
            ):
                alias_counts["slide_features"] += 1

    return alias_counts


def build_run_command(
    unknown_args: list[str],
    args: argparse.Namespace,
    manifest_csv: Path,
    sample_id_column: str,
    expanded_elastic_only_samples: list[str],
    extra_overrides: list[str] | None = None,
) -> list[str]:
    runner = Path(__file__).with_name("run_batch_multimodal.py")
    cmd = [
        sys.executable,
        str(runner),
        *unknown_args,
        "--custom_list_of_wsis",
        str(manifest_csv),
        "--job_dir",
        str(args.job_dir),
        "--sample_id_column",
        sample_id_column,
    ]
    if expanded_elastic_only_samples:
        cmd.extend(["--elastic_only_samples", ",".join(expanded_elastic_only_samples)])
    if extra_overrides:
        cmd.extend(extra_overrides)
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, unknown_args = parser.parse_known_args(argv)

    adapted_manifest_csv = args.adapted_manifest_csv
    if adapted_manifest_csv is None:
        adapted_manifest_csv = args.job_dir / "manifest_trident_3D_adapted.csv"
    source_manifest_csv = args.source_manifest_csv
    if source_manifest_csv is None:
        source_manifest_csv = args.job_dir / "manifest_trident_3D_nodo_source.csv"

    by_source_sample = adapt_manifest(
        manifest_csv=args.custom_list_of_wsis,
        output_csv=adapted_manifest_csv,
        sample_col=args.sample_id_column,
        slice_depth_col=args.slice_depth_column,
        trident_sample_col=args.trident_sample_id_column,
        source_sample_col=args.source_sample_id_column,
    )
    write_source_manifest(
        manifest_csv=args.custom_list_of_wsis,
        output_csv=source_manifest_csv,
        sample_col=args.sample_id_column,
        slice_depth_col=args.slice_depth_column,
        trident_sample_col=args.trident_sample_id_column,
        source_sample_col=args.source_sample_id_column,
    )

    requested_elastic_only = read_elastic_only_samples(
        args.elastic_only_samples,
        args.elastic_only_samples_file,
    )
    expanded_elastic_only = expand_elastic_only_samples(requested_elastic_only, by_source_sample)

    fallback_cmd = build_run_command(
        unknown_args=unknown_args,
        args=args,
        manifest_csv=adapted_manifest_csv,
        sample_id_column=args.trident_sample_id_column,
        expanded_elastic_only_samples=expanded_elastic_only,
    )
    use_optimized_nodo_reuse = (not args.no_reuse_nodo) and normalized_modality(unknown_args) == "both"
    requested_task = normalized_task(unknown_args)
    visible_gpu_tokens = parse_cuda_visible_devices()
    use_multi_gpu_feat = use_optimized_nodo_reuse and requested_task == "all" and len(visible_gpu_tokens) > 1
    coords_profile = default_coords_profile(unknown_args)

    nodo_cmd = build_run_command(
        unknown_args=unknown_args,
        args=args,
        manifest_csv=source_manifest_csv,
        sample_id_column=args.trident_sample_id_column,
        expanded_elastic_only_samples=[],
        extra_overrides=["--modality", "nodo"],
    )
    odo_cmd = build_run_command(
        unknown_args=unknown_args,
        args=args,
        manifest_csv=adapted_manifest_csv,
        sample_id_column=args.trident_sample_id_column,
        expanded_elastic_only_samples=expanded_elastic_only,
        extra_overrides=["--modality", "odo", "--task", "feat", "--nodo-job-dir", str(args.job_dir / "NODO")],
    )

    print(f"[3D] Wrote adapted manifest: {adapted_manifest_csv}")
    print(f"[3D] Wrote source-only NODO manifest: {source_manifest_csv}")
    print(
        "[3D] Adapted rows: "
        f"samples={sum(len(ids) for ids in by_source_sample.values())} "
        f"source_samples={len(by_source_sample)}"
    )
    if expanded_elastic_only:
        print(f"[3D] Expanded elastic-only sample IDs: {len(expanded_elastic_only)}")
    if visible_gpu_tokens:
        print(f"[3D] CUDA_VISIBLE_DEVICES parsed as: {','.join(visible_gpu_tokens)}")

    if use_optimized_nodo_reuse:
        if use_multi_gpu_feat:
            print("[3D] Optimized NODO reuse is enabled with multi-GPU feature sharding.")
        else:
            print("[3D] Optimized NODO reuse is enabled.")
            print("[3D] NODO command:")
            print(" ".join(nodo_cmd))
            print("[3D] ODO command:")
            print(" ".join(odo_cmd))
    else:
        print("[3D] Optimized NODO reuse is disabled; using one full adapted-manifest command.")
        print("[3D] run_batch_multimodal.py command:")
        print(" ".join(fallback_cmd))

    if args.adapt_only:
        return 0

    if use_optimized_nodo_reuse:
        if use_multi_gpu_feat:
            nodo_seg_cmd = build_run_command(
                unknown_args=unknown_args,
                args=args,
                manifest_csv=source_manifest_csv,
                sample_id_column=args.trident_sample_id_column,
                expanded_elastic_only_samples=[],
                extra_overrides=["--modality", "nodo", "--task", "seg"],
            )
            nodo_coords_cmd = build_run_command(
                unknown_args=unknown_args,
                args=args,
                manifest_csv=source_manifest_csv,
                sample_id_column=args.trident_sample_id_column,
                expanded_elastic_only_samples=[],
                extra_overrides=["--modality", "nodo", "--task", "coords"],
            )
            odo_mapping_cmd = build_run_command(
                unknown_args=unknown_args,
                args=args,
                manifest_csv=adapted_manifest_csv,
                sample_id_column=args.trident_sample_id_column,
                expanded_elastic_only_samples=expanded_elastic_only,
                extra_overrides=[
                    "--modality",
                    "odo",
                    "--task",
                    "feat",
                    "--nodo-job-dir",
                    str(args.job_dir / "NODO"),
                    "--_skip_odo_final_feat",
                ],
            )

            print("[3D] NODO seg command:")
            print(" ".join(nodo_seg_cmd))
            subprocess.run(nodo_seg_cmd, check=True)

            print("[3D] NODO coords command:")
            print(" ".join(nodo_coords_cmd))
            subprocess.run(nodo_coords_cmd, check=True)

            nodo_feature_manifest_csv = args.job_dir / "NODO" / "manifest_trident_nodo_multimodal.csv"
            if not nodo_feature_manifest_csv.exists():
                raise FileNotFoundError(
                    "Expected NODO multimodal manifest for sharded feature extraction was not found: "
                    f"{nodo_feature_manifest_csv}"
                )
            run_sharded_feature_extraction(
                stage_name="nodo_feat",
                manifest_csv=nodo_feature_manifest_csv,
                job_dir=args.job_dir / "NODO",
                coords_dir=coords_profile,
                unknown_args=unknown_args,
                gpu_tokens=visible_gpu_tokens,
            )

            alias_counts = materialize_nodo_aliases(
                job_dir=args.job_dir,
                coords_profile=coords_profile,
                patch_encoder=patch_encoder_name(unknown_args),
                by_source_sample=by_source_sample,
                alias_mode=args.alias_mode,
            )
            print(
                "[3D] Materialized NODO aliases: "
                f"patches={alias_counts['patches']} "
                f"features={alias_counts['features']} "
                f"slide_features={alias_counts['slide_features']}"
            )

            mapped_manifest_csv = run_odo_with_2d_count_mismatch_tolerance(
                odo_cmd=odo_mapping_cmd,
                unknown_args=unknown_args,
                args=args,
                run_direct_feat=False,
            )
            run_sharded_feature_extraction(
                stage_name="odo_feat",
                manifest_csv=mapped_manifest_csv,
                job_dir=args.job_dir / "ODO",
                coords_dir=f"{coords_profile}/coords",
                unknown_args=unknown_args,
                gpu_tokens=visible_gpu_tokens,
            )
        else:
            subprocess.run(nodo_cmd, check=True)
            alias_counts = materialize_nodo_aliases(
                job_dir=args.job_dir,
                coords_profile=coords_profile,
                patch_encoder=patch_encoder_name(unknown_args),
                by_source_sample=by_source_sample,
                alias_mode=args.alias_mode,
            )
            print(
                "[3D] Materialized NODO aliases: "
                f"patches={alias_counts['patches']} "
                f"features={alias_counts['features']} "
                f"slide_features={alias_counts['slide_features']}"
            )
            run_odo_with_2d_count_mismatch_tolerance(
                odo_cmd=odo_cmd,
                unknown_args=unknown_args,
                args=args,
            )
    else:
        subprocess.run(fallback_cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
