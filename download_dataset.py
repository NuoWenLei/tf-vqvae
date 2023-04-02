from dataset_helpers import load_dataset_by_splits
from constants import LOCAL_DATASET_PATH

def main():
    load_dataset_by_splits(
		dataset_name="sbu_captions",
		percent_splits=1,
		load_local = False,
		disk_dir=LOCAL_DATASET_PATH,
		save_local = True
	)
    

if __name__ == "__main__":
    main()