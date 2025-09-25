#!/usr/bin/env python3

"""Download and unpack the Stanford 3D Indoor Semantics Dataset (S3DIS)."""
import argparse
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

DATA_URL = "https://cvg-data.inf.ethz.ch/s3dis/Stanford3dDataset_v1.2_Aligned_Version.zip"
ARCHIVE_NAME = "Stanford3dDataset_v1.2_Aligned_Version.zip"
EXTRACTED_DIR_NAME = "Stanford3dDataset_v1.2_Aligned_Version"


def human_readable(num_bytes: float) -> str:
    """Return a human friendly string for a byte count."""
    num_bytes = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(num_bytes)} {unit}"
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def download_archive(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    request = Request(url)
    with urlopen(request) as response:
        total = response.getheader("Content-Length")
        total_size = int(total) if total else None
        chunk_size = 1 << 20  # 1 MiB
        downloaded = 0
        with destination.open("wb") as outfile:
            try:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    outfile.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        percent = downloaded * 100.0 / total_size
                        sys.stdout.write(
                            f"Downloaded {human_readable(downloaded)} / {human_readable(total_size)} ({percent:5.1f}%)"
                        )
                    else:
                        sys.stdout.write(f"Downloaded {human_readable(downloaded)}")
                    sys.stdout.flush()
            except Exception:
                if destination.exists():
                    destination.unlink()
                raise
    sys.stdout.write("")


def extract_archive(archive_path: Path, target_dir: Path) -> None:
    print(f"Extracting {archive_path.name} into {target_dir}")
    with zipfile.ZipFile(archive_path) as zip_file:
        zip_file.extractall(target_dir)


def remove_archive(archive_path: Path) -> None:
    if archive_path.exists():
        archive_path.unlink()
        print(f"Removed {archive_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and extract the S3DIS dataset archive."
    )
    default_output = (Path(__file__).resolve().parents[1] / "datasets" / "s3dis")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory used to store the archive and extracted files (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and re-extraction even if data already exists.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    archive_path = output_dir / ARCHIVE_NAME
    extracted_dir = output_dir / EXTRACTED_DIR_NAME

    if extracted_dir.exists():
        if not args.force:
            print(f"Found existing extracted dataset at {extracted_dir}. Use --force to refresh.")
            return 0
        print(f"Removing existing directory {extracted_dir}")
        shutil.rmtree(extracted_dir)

    if archive_path.exists():
        if args.force:
            print(f"Removing existing archive {archive_path}")
            archive_path.unlink()
        else:
            print(f"Using existing archive at {archive_path}")

    if not archive_path.exists():
        try:
            download_archive(DATA_URL, archive_path)
        except KeyboardInterrupt:
            print("Download interrupted; cleaning up.")
            if archive_path.exists():
                archive_path.unlink()
            return 1
        except Exception as exc:
            print(f"Failed to download archive: {exc}")
            return 1

    try:
        extract_archive(archive_path, output_dir)
    except zipfile.BadZipFile as exc:
        print(f"Failed to extract archive: {exc}")
        if extracted_dir.exists():
            shutil.rmtree(extracted_dir)
        return 1

    remove_archive(archive_path)
    print(f"Dataset ready in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
