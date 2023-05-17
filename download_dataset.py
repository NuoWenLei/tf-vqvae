from helpers.dataset_helpers import download_dataset_to_local_path
from helpers.constants import LOCAL_DATASET_PATH

def download():
	download_dataset_to_local_path(dataset_name="sbu_captions", disk_dir=LOCAL_DATASET_PATH)

if __name__ == "__main__":
    download()