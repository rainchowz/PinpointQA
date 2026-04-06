#!/usr/bin/env python3
"""

Typical usage:
    python convert_mkv_to_mp4.py /path/to/videos

Scannet++-style usage (only process rgb.mkv files):
    python convert_mkv_to_mp4.py /path/to/Scannet++/data --pattern rgb.mkv
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class ConversionResult:
    source: Path
    target: Path
    status: str
    reason: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively remux MKV files to MP4 using ffmpeg stream copy."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory to scan recursively for MKV files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.mkv",
        help="Filename pattern to match inside the input directory (default: %(default)s).",
    )
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete the source MKV after a successful conversion.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def ensure_tool_exists(tool_name: str) -> None:
    if shutil.which(tool_name) is None:
        raise SystemExit(
            f"Required dependency not found: '{tool_name}'. Please install it and make sure it is on PATH."
        )


def has_ffprobe() -> bool:
    return shutil.which("ffprobe") is not None


def find_input_files(input_dir: Path, pattern: str) -> list[Path]:
    return sorted(path for path in input_dir.rglob(pattern) if path.is_file())


def build_output_path(source_path: Path) -> Path:
    return source_path.with_suffix(".mp4")


def is_valid_media_file(path: Path, use_ffprobe: bool) -> bool:
    if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
        return False

    if not use_ffprobe:
        return True

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False

    duration_text = result.stdout.strip()
    if not duration_text:
        return False

    try:
        return float(duration_text) > 0
    except ValueError:
        return False


def make_temp_output_path(target_path: Path) -> Path:
    return target_path.with_name(f".{target_path.stem}.tmp{target_path.suffix}")


def remux_mkv_to_mp4(source_path: Path, target_path: Path, use_ffprobe: bool) -> tuple[bool, str]:
    temp_output = make_temp_output_path(target_path)
    temp_output.unlink(missing_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_path),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(temp_output),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        temp_output.unlink(missing_ok=True)
        stderr = result.stderr.strip() or "ffmpeg returned a non-zero exit code."
        return False, stderr

    if not is_valid_media_file(temp_output, use_ffprobe):
        temp_output.unlink(missing_ok=True)
        return False, "Temporary MP4 was created but failed validation."

    os.replace(temp_output, target_path)
    return True, ""


def convert_one(source_path: Path, delete_source: bool, use_ffprobe: bool) -> ConversionResult:
    target_path = build_output_path(source_path)

    if is_valid_media_file(target_path, use_ffprobe):
        return ConversionResult(source=source_path, target=target_path, status="skipped", reason="Valid target already exists.")

    ok, reason = remux_mkv_to_mp4(source_path, target_path, use_ffprobe)
    if not ok:
        return ConversionResult(source=source_path, target=target_path, status="failed", reason=reason)

    if delete_source:
        try:
            source_path.unlink()
        except OSError as exc:
            return ConversionResult(
                source=source_path,
                target=target_path,
                status="failed",
                reason=f"Conversion succeeded, but deleting source failed: {exc}",
            )

    return ConversionResult(source=source_path, target=target_path, status="success")


def summarize(results: Iterable[ConversionResult]) -> tuple[int, int, int]:
    success = skipped = failed = 0
    for item in results:
        if item.status == "success":
            success += 1
        elif item.status == "skipped":
            skipped += 1
        else:
            failed += 1
    return success, skipped, failed


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    ensure_tool_exists("ffmpeg")
    use_ffprobe = has_ffprobe()
    if use_ffprobe:
        logging.info("Using ffprobe for output validation.")
    else:
        logging.warning("ffprobe not found. Falling back to file-size checks only.")

    input_dir: Path = args.input_dir.expanduser().resolve()
    if not input_dir.exists():
        logging.error("Input directory does not exist: %s", input_dir)
        return 2
    if not input_dir.is_dir():
        logging.error("Input path is not a directory: %s", input_dir)
        return 2

    input_files = find_input_files(input_dir, args.pattern)
    if not input_files:
        logging.warning("No files matched pattern '%s' under %s", args.pattern, input_dir)
        return 0

    logging.info("Found %d file(s) matching '%s' under %s", len(input_files), args.pattern, input_dir)

    results: list[ConversionResult] = []
    for index, source_path in enumerate(input_files, start=1):
        logging.info("[%d/%d] Processing: %s", index, len(input_files), source_path)
        result = convert_one(
            source_path=source_path,
            delete_source=args.delete_source,
            use_ffprobe=use_ffprobe,
        )
        results.append(result)

        if result.status == "success":
            logging.info("Done: %s -> %s", result.source.name, result.target.name)
        elif result.status == "skipped":
            logging.info("Skipped: %s", result.reason)
        else:
            logging.error("Failed: %s", result.reason)

    success, skipped, failed = summarize(results)

    print("\nSummary")
    print("-------")
    print(f"Total   : {len(results)}")
    print(f"Success : {success}")
    print(f"Skipped : {skipped}")
    print(f"Failed  : {failed}")

    if failed:
        print("\nFailed files")
        print("------------")
        for item in results:
            if item.status == "failed":
                print(f"{item.source}")
                if item.reason:
                    print(f"  Reason: {item.reason}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
