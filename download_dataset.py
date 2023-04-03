from dataset_helpers import download_dataset_to_local_path
from constants import LOCAL_DATASET_PATH

def main():
	download_dataset_to_local_path(dataset_name="sbu_captions", disk_dir=LOCAL_DATASET_PATH)

if __name__ == "__main__":
    main()