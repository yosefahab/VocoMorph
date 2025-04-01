import os
import tarfile
import zipfile
import requests
from src.logging.logger import get_logger
from concurrent.futures import ThreadPoolExecutor, as_completed

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

TIMIT_URL = "https://figshare.com/ndownloader/files/10256148"


def download_file(url: str, dest_path: str) -> bool:
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
        logger.error(f"Failed to download {url}: {e}")
        return False


def extract_tarfile(tar_path: str, extract_path: str) -> bool:
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


def extract_zipfile(zip_path: str, extract_path: str) -> bool:
    """Extracts a zip file into the given directory."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        logger.info(f"Extracted {zip_path} to {extract_path}")
        return True
    except zipfile.BadZipFile as e:
        logger.error(f"Failed to extract {zip_path}: {e}")
        return False


def download_librispeech(subset: str, destination_dir: str, remove_tar: bool = True):
    """Downloads and extracts a LibriSpeech dataset subset."""
    if subset not in LIBRISPEECH_URLS:
        logger.error(f"Invalid dataset: {subset}")
        return False

    dataset_dir = os.path.join(destination_dir, subset)
    os.makedirs(dataset_dir, exist_ok=True)

    url = LIBRISPEECH_URLS[subset]
    tar_path = os.path.join(destination_dir, f"{subset}.tar.gz")

    if not download_file(url, tar_path):
        return False

    if not extract_tarfile(tar_path, destination_dir):
        return False

    if remove_tar:
        os.remove(tar_path)
        logger.info(f"Removed {tar_path}")

    return True


def download_timit(destination_dir: str, remove_zip: bool = True):
    """Downloads and extracts the TIMIT dataset."""
    dataset_dir = os.path.join(destination_dir, "TIMIT")
    os.makedirs(dataset_dir, exist_ok=True)

    zip_path = os.path.join(destination_dir, "timit.zip")

    if not download_file(TIMIT_URL, zip_path):
        return False

    if not extract_zipfile(zip_path, dataset_dir):
        return False

    if remove_zip:
        os.remove(zip_path)
        logger.info(f"Removed {zip_path}")

    logger.info("Finished processing TIMIT dataset")
    return True


def main(dataset: str):
    """Main function to download the requested dataset."""
    destination_dir = os.path.join(os.environ["PROJECT_ROOT"], "data", dataset)
    if os.path.exists(destination_dir):
        logger.info("Dataset already downloaded")
        return

    os.makedirs(destination_dir, exist_ok=True)
    logger.info(f"Downloading dataset: {dataset}.")
    if dataset == "librispeech":
        subsets = ["train-clean-100", "dev-clean", "test-clean"]
        with ThreadPoolExecutor(max_workers=len(subsets)) as executor:
            future_to_subset = {
                executor.submit(download_librispeech, subset, destination_dir): subset
                for subset in subsets
            }

            for future in as_completed(future_to_subset):
                subset = future_to_subset[future]
                try:
                    success = future.result()
                    if not success:
                        logger.error(f"Failed to process {subset}")
                except Exception as e:
                    logger.error(f"Exception while processing {subset}: {e}")

    elif dataset == "timit":
        download_timit(destination_dir)

    else:
        logger.error(f"Invalid dataset: {dataset}")
