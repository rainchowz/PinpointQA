## Requirements

- Python 3.9+
- `ffmpeg` available in your system `PATH`

## Optional Python Dependency

- `tqdm` is only needed for `convert_mkv_to_mp4.py`

## Installation

```bash
pip install tqdm
```

## Usage

For MKV (e.g. ScanNet++) to MP4:

```bash
python convert_mkv_to_mp4.py /path/to/root
```

For ScanNet `.sens` to MP4:

```bash
python convert_sens_to_mp4.py /path/to/file.sens_or_folder
```
