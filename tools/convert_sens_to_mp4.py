#!/usr/bin/env python3
"""
Convert ScanNet/ScanNet200 .sens files to MP4 videos.

Typical usage:
    python sens_to_video_open.py /path/to/sens_dir

Single-file usage:
    python sens_to_video_open.py /path/to/scene0000_00.sens

Optional custom output directory:
    python sens_to_video_open.py /path/to/sens_dir --output-dir /path/to/videos
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator


DEFAULT_FPS = 30
DEFAULT_PRESET = "veryfast"
DEFAULT_CRF = 23


@dataclass
class ConversionResult:
    source: Path
    target: Path
    status: str
    frames: int = 0
    reason: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ScanNet .sens files to MP4 videos."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a .sens file or a directory containing .sens files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated MP4 files. Defaults to '<input_name>_videos' next to the input.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"Output video frame rate (default: {DEFAULT_FPS}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing MP4 files instead of skipping valid outputs.",
    )
    parser.add_argument(
        "--preset",
        default=DEFAULT_PRESET,
        help=f"ffmpeg x264 preset (default: {DEFAULT_PRESET}).",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=DEFAULT_CRF,
        help=f"ffmpeg x264 CRF value (default: {DEFAULT_CRF}).",
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


def ensure_dependency(tool_name: str) -> None:
    if shutil.which(tool_name) is None:
        raise SystemExit(
            f"Required dependency not found: '{tool_name}'. Please install it and make sure it is available on PATH."
        )


def has_ffprobe() -> bool:
    return shutil.which("ffprobe") is not None


def read_exact(file_obj: BinaryIO, num_bytes: int) -> bytes:
    data = file_obj.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError(
            f"Expected to read {num_bytes} bytes, but only received {len(data)} bytes."
        )
    return data


def read_u32(file_obj: BinaryIO) -> int:
    return struct.unpack("<I", read_exact(file_obj, 4))[0]


def read_u64(file_obj: BinaryIO) -> int:
    return struct.unpack("<Q", read_exact(file_obj, 8))[0]


def collect_sens_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".sens":
            raise ValueError(f"Input file is not a .sens file: {input_path}")
        return [input_path]

    if input_path.is_dir():
        return sorted(path for path in input_path.rglob("*.sens") if path.is_file())

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def infer_output_dir(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.parent / "converted_videos"
    return input_path.parent / f"{input_path.name}_videos"


def output_path_for(source_path: Path, output_dir: Path, input_root: Path) -> Path:
    if input_root.is_file():
        return output_dir / f"{source_path.stem}.mp4"

    relative_path = source_path.relative_to(input_root)
    return (output_dir / relative_path).with_suffix(".mp4")


def temp_output_path_for(target_path: Path) -> Path:
    return target_path.with_suffix(".part.mp4")


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


def iter_color_jpegs(sens_path: Path) -> Iterator[bytes]:
    """
    Stream JPEG color frames from a .sens file.

    This implementation is tailored to ScanNet-style .sens files where
    the color stream is stored as JPEG-compressed images.
    """
    with sens_path.open("rb") as file_obj:
        version = read_u32(file_obj)
        if version != 4:
            raise ValueError(
                f"Unsupported .sens version in {sens_path}: {version}. Expected version 4."
            )

        sensor_name_length = read_u64(file_obj)
        read_exact(file_obj, sensor_name_length)

        # Four 4x4 float32 matrices
        read_exact(file_obj, 16 * 4)  # intrinsic_color
        read_exact(file_obj, 16 * 4)  # extrinsic_color
        read_exact(file_obj, 16 * 4)  # intrinsic_depth
        read_exact(file_obj, 16 * 4)  # extrinsic_depth

        color_compression_type = read_u32(file_obj)
        depth_compression_type = read_u32(file_obj)
        color_width = read_u32(file_obj)
        color_height = read_u32(file_obj)
        depth_width = read_u32(file_obj)
        depth_height = read_u32(file_obj)
        depth_shift = read_u32(file_obj)
        num_frames = read_u64(file_obj)

        logging.debug(
            "Header: color_comp=%s depth_comp=%s color=%sx%s depth=%sx%s depth_shift=%s frames=%s",
            color_compression_type,
            depth_compression_type,
            color_width,
            color_height,
            depth_width,
            depth_height,
            depth_shift,
            num_frames,
        )

        for frame_index in range(num_frames):
            read_exact(file_obj, 64)  # camera_to_world
            read_exact(file_obj, 8)   # timestamp_color
            read_exact(file_obj, 8)   # timestamp_depth

            color_size = read_u64(file_obj)
            depth_size = read_u64(file_obj)

            if color_size > 0:
                color_data = read_exact(file_obj, color_size)
            else:
                color_data = b""

            if depth_size > 0:
                read_exact(file_obj, depth_size)

            if color_data:
                yield color_data
            else:
                logging.debug("Frame %d has empty color data and was skipped.", frame_index)


def convert_sens_to_mp4(
    source_path: Path,
    target_path: Path,
    fps: int,
    preset: str,
    crf: int,
    use_ffprobe: bool,
) -> tuple[int, str]:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = temp_output_path_for(target_path)
    temp_path.unlink(missing_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(temp_path),
    ]

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    frame_count = 0

    try:
        assert process.stdin is not None

        for jpeg_bytes in iter_color_jpegs(source_path):
            try:
                process.stdin.write(jpeg_bytes)
            except BrokenPipeError:
                stderr_text = ""
                if process.stderr is not None:
                    stderr_text = process.stderr.read().decode("utf-8", errors="ignore")
                raise RuntimeError(
                    "ffmpeg terminated early while receiving frames.\n"
                    f"{stderr_text.strip()}"
                )
            frame_count += 1

        process.stdin.close()

        stderr_text = ""
        if process.stderr is not None:
            stderr_text = process.stderr.read().decode("utf-8", errors="ignore")

        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg failed with exit code {return_code}.\n{stderr_text.strip()}"
            )

        if frame_count == 0:
            raise RuntimeError("No color frames were extracted from the .sens file.")

        if not is_valid_media_file(temp_path, use_ffprobe):
            raise RuntimeError("Generated temporary MP4 failed validation.")

        os.replace(temp_path, target_path)
        return frame_count, ""

    except Exception as exc:
        try:
            process.kill()
        except Exception:
            pass
        temp_path.unlink(missing_ok=True)
        return 0, str(exc)


def convert_one(
    source_path: Path,
    output_dir: Path,
    input_root: Path,
    fps: int,
    preset: str,
    crf: int,
    overwrite: bool,
    use_ffprobe: bool,
) -> ConversionResult:
    target_path = output_path_for(source_path, output_dir, input_root)

    if not overwrite and is_valid_media_file(target_path, use_ffprobe):
        return ConversionResult(
            source=source_path,
            target=target_path,
            status="skipped",
            reason="Valid output already exists.",
        )

    frames, error_message = convert_sens_to_mp4(
        source_path=source_path,
        target_path=target_path,
        fps=fps,
        preset=preset,
        crf=crf,
        use_ffprobe=use_ffprobe,
    )

    if error_message:
        return ConversionResult(
            source=source_path,
            target=target_path,
            status="failed",
            frames=0,
            reason=error_message,
        )

    return ConversionResult(
        source=source_path,
        target=target_path,
        status="success",
        frames=frames,
    )


def summarize(results: list[ConversionResult]) -> tuple[int, int, int]:
    success = sum(1 for result in results if result.status == "success")
    skipped = sum(1 for result in results if result.status == "skipped")
    failed = sum(1 for result in results if result.status == "failed")
    return success, skipped, failed


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    ensure_dependency("ffmpeg")
    use_ffprobe = has_ffprobe()
    if use_ffprobe:
        logging.info("Using ffprobe to validate generated MP4 files.")
    else:
        logging.warning("ffprobe not found. Falling back to file-size checks only.")

    input_path = args.input_path.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else infer_output_dir(input_path)
    )

    try:
        sens_files = collect_sens_files(input_path)
    except Exception as exc:
        logging.error(str(exc))
        return 2

    if not sens_files:
        logging.warning("No .sens files found under %s", input_path)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Found %d .sens file(s).", len(sens_files))
    logging.info("Output directory: %s", output_dir)

    results: list[ConversionResult] = []
    for index, source_path in enumerate(sens_files, start=1):
        logging.info("[%d/%d] Processing %s", index, len(sens_files), source_path.name)
        result = convert_one(
            source_path=source_path,
            output_dir=output_dir,
            input_root=input_path,
            fps=args.fps,
            preset=args.preset,
            crf=args.crf,
            overwrite=args.overwrite,
            use_ffprobe=use_ffprobe,
        )
        results.append(result)

        if result.status == "success":
            logging.info(
                "Success: %s -> %s (%d frames)",
                result.source.name,
                result.target.name,
                result.frames,
            )
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
        for result in results:
            if result.status == "failed":
                print(result.source)
                if result.reason:
                    print(f"  Reason: {result.reason}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
