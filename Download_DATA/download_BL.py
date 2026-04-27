from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path


BATTERYLIFE_ARCHIVE_URL = "https://zenodo.org/api/records/18646655/files-archive"
PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_DIR / "Raw_BML"
DEFAULT_ARCHIVE_NAME = "BatteryLife_files_archive.zip"


def _download_with_progress(url: str, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {url}")
    with urllib.request.urlopen(url) as response, open(dst_path, "wb") as out_file:
        total = response.headers.get("Content-Length")
        total_size = int(total) if total and total.isdigit() else None
        downloaded = 0

        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size:
                pct = downloaded * 100.0 / float(total_size)
                print(
                    f"\rDownloaded {downloaded / (1024**2):.1f} / "
                    f"{total_size / (1024**2):.1f} MB ({pct:.1f}%)",
                    end="",
                    flush=True,
                )
            else:
                print(
                    f"\rDownloaded {downloaded / (1024**2):.1f} MB",
                    end="",
                    flush=True,
                )
    print()


def _extract_zip(archive_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zf:
        members = zf.namelist()
        if not members:
            raise RuntimeError("Downloaded archive is empty.")
        zf.extractall(out_dir)
    print(f"Extracted files to: {out_dir}")


def _data_already_exists(out_dir: Path) -> bool:
    return out_dir.exists() and any(out_dir.iterdir())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download BatteryLife dataset archive from Zenodo into Raw_BML."
    )
    parser.add_argument(
        "--url",
        default=BATTERYLIFE_ARCHIVE_URL,
        help="Archive download URL (default: Zenodo BatteryLife files-archive).",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Directory where raw dataset files will be extracted.",
    )
    parser.add_argument(
        "--archive-path",
        default=None,
        help=(
            "Optional path to store archive zip. "
            "Default: <out-dir-parent>/BatteryLife_files_archive.zip"
        ),
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded archive zip after extraction.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download even if output directory already contains files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    archive_path = (
        Path(args.archive_path).expanduser().resolve()
        if args.archive_path
        else (out_dir.parent / DEFAULT_ARCHIVE_NAME)
    )

    if _data_already_exists(out_dir) and not args.force:
        print(f"BatteryLife dataset already exists: {out_dir}")
        print("Use --force to re-download.")
        return

    if args.force and out_dir.exists():
        shutil.rmtree(out_dir)

    _download_with_progress(args.url, archive_path)
    _extract_zip(archive_path, out_dir)

    if not args.keep_archive and archive_path.exists():
        archive_path.unlink()

    print(f"Done. BatteryLife raw data directory: {out_dir}")
    print(
        "If you need to extract features next, run:\n"
        "python Gen_Data/gen_discharge_data.py --data-dir ./Raw_BML --out-dir ./content_discharge"
    )


if __name__ == "__main__":
    main()
