from dataset_helpers import load_dataset_by_splits, fetch_images_of_dataset, iter_dataset_by_batch
from model import get_image_vqvae
from constants import LOCAL_DATASET_PATH, BATCH_SIZE
from imports import tf
import os

def main():
    # TODO: steps:
	# - load dataset splits
	if os.path.exists(LOCAL_DATASET_PATH) and (len(os.listdir(LOCAL_DATASET_PATH)) > 0):
		splits = load_dataset_by_splits(load_local = True, disk_dir = LOCAL_DATASET_PATH)
	else:
		splits = load_dataset_by_splits()
	# - load model
	model = get_image_vqvae()
	model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3))
	# - iterate through each split
	for split in splits:
		# - run training according to the add_loss method in Keras
		image_ds = fetch_images_of_dataset(split)
		ds_iterator = iter_dataset_by_batch(image_ds, only_images = True, batch_size = BATCH_SIZE)
		model.fit(x = ds_iterator)
		# TODO: model checkpoint
	

if __name__ == "__main__":
	main()