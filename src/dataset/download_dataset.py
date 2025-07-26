import os
import shutil
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)

LIBRISPEECH_URLS = {
    "train-clean-100": "http://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "http://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "http://www.openslr.org/resources/12/train-other-500.tar.gz",
    "dev-clean": "http://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "http://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean": "http://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "http://www.openslr.org/resources/12/test-other.tar.gz",
}
LIBRISPEECH_SUBSET_TO_DIR = {
    "train-clean-100": "train",
    "dev-clean": "valid",
    "test-clean": "test",
}


TIMIT_URL = "https://figshare.com/ndownloader/files/10256148"


def download_file(url: str, dest_path: Path) -> bool:
    """Downloads a file from a URL and saves it to the destination path."""
    try:
        logger.info(f"Downloading file: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        logger.info(f"Downloaded {dest_path}")
        return True
    except requests.RequestException as e:
        logger.exception(f"Failed to download {url}: {e}")
        return False


def extract_tarfile(tar_path: Path, extract_path: Path) -> bool:
    """Extracts a tar.gz file into the given directory."""
    try:
        logger.info(f"extracting tar: {tar_path}")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(extract_path)
        logger.info(f"Extracted {tar_path} to {extract_path}")
        return True
    except tarfile.TarError as e:
        logger.error(f"Failed to extract {tar_path}: {e}")
        return False


def extract_zipfile(zip_path: Path, extract_path: Path) -> bool:
    """Extracts a zip file into the given directory."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        logger.info(f"Extracted {zip_path} to {extract_path}")
        return True
    except zipfile.BadZipFile as e:
        logger.error(f"Failed to extract {zip_path}: {e}")
        return False


def download_librispeech_subset(
    subset: str, destination_dir: Path, remove_tar: bool = True
):
    if subset not in LIBRISPEECH_URLS:
        logger.error(f"Invalid dataset: {subset}")
        return False

    url = LIBRISPEECH_URLS[subset]
    tar_path = destination_dir.joinpath(f"{subset}.tar.gz")

    if not download_file(url, tar_path):
        return False
    if not extract_tarfile(tar_path, destination_dir):
        return False

    src_subset_dir = destination_dir.joinpath("LibriSpeech", subset)
    dst_subset_dir = destination_dir.joinpath(LIBRISPEECH_SUBSET_TO_DIR[subset])

    # replace or skip if the subset is already there
    if dst_subset_dir.exists():
        shutil.rmtree(dst_subset_dir)
    shutil.move(src_subset_dir, dst_subset_dir)

    lib_dir = destination_dir.joinpath("LibriSpeech")

    # move shared .TXT files
    for file in lib_dir.iterdir():
        file_path = lib_dir.joinpath(file)
        if file.name.endswith(".txt") and file_path.is_file():
            shutil.move(file_path, destination_dir.joinpath(file))

    # remove LibriSpeech if empty after move
    if not lib_dir.iterdir():
        os.rmdir(lib_dir)

    if remove_tar:
        tar_path.unlink()
    return True


def download_librispeech(destination_dir: Path):
    subsets = ["train-clean-100", "dev-clean", "test-clean"]
    with ThreadPoolExecutor(max_workers=len(subsets)) as executor:
        future_to_subset = {
            executor.submit(
                download_librispeech_subset, subset, destination_dir
            ): subset
            for subset in subsets
        }
        for future in as_completed(future_to_subset):
            subset = future_to_subset[future]
            try:
                success = future.result()
                if not success:
                    logger.error(f"Failed to download {subset}")
            except Exception as e:
                logger.exception(f"Exception while downloading {subset}: {e}")


def download_timit(destination_dir: Path, remove_zip: bool = True) -> bool:
    """Downloads and extracts the TIMIT dataset."""
    destination_dir.mkdir(exist_ok=True)

    zip_path = destination_dir.joinpath("timit.zip")

    if not download_file(TIMIT_URL, zip_path):
        return False

    if not extract_zipfile(zip_path, destination_dir):
        logger.info(f"Extracted {zip_path} to {destination_dir}")
        return False

    if remove_zip:
        zip_path.unlink()
        logger.info(f"Removed {zip_path}")

    # this specific url nests the dataset, so we need to move it to the destination_dir
    timit_path = destination_dir.joinpath("data/lisa/data/timit/raw/TIMIT")
    for item in ["TRAIN", "TEST"]:
        s = timit_path.joinpath(item)
        d = destination_dir.joinpath(item.lower())
        shutil.move(s, d)

    shutil.rmtree(destination_dir.joinpath("data"))
    logger.info("Finished downloading TIMIT dataset")
    return True


def main(dataset: str):
    """Main function to download the requested dataset."""
    destination_dir = Path(os.environ["DATA_ROOT"]).joinpath(dataset)
    if destination_dir.exists():
        logger.info("Dataset already downloaded")
        return

    logger.info(f"Downloading dataset: {dataset}.")
    destination_dir.mkdir(exist_ok=True)
    if dataset == "librispeech":
        download_librispeech(destination_dir)
    elif dataset == "timit":
        download_timit(destination_dir)

    else:
        logger.error(f"Invalid dataset: {dataset}")
