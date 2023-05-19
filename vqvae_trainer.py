from helpers.dataset_helpers import load_dataset_by_splits, fetch_images_of_dataset, iter_dataset_by_batch
from helpers.model import get_image_vqvae
from helpers.constants import LOCAL_DATASET_PATH, BATCH_SIZE
from helpers.imports import tf
import os

def train_batch_gen(ds, batch_size, return_feat_target = False):
	"""
	Create a generator that iterates through batches of images from a huggingface dataset.

	Args:
	- ds: huggingface dataset with "image_array" and "image_status" columns
	- batch_size: size of batch
	- return_feat_target = False: whether to return images as both feature and target or just as standalone tensor

	Returns:
	- generator object
	"""
	i = 0
	num_batches = len(ds) // batch_size
	gen = iter_dataset_by_batch(ds, batch_size = batch_size, only_images = True)
	while True:
		batch = next(gen)
		i += 1
		if i == num_batches:
			ds = ds.shuffle()
			gen = iter_dataset_by_batch(ds, batch_size = batch_size, only_images = True)
			i = 0
		
		if return_feat_target:
			yield batch, batch
		else:
			yield batch

def main():
	# TODO: steps:
	# - load dataset splits
	if os.path.exists(LOCAL_DATASET_PATH) and (len(os.listdir(LOCAL_DATASET_PATH)) > 0):
		splits = load_dataset_by_splits(load_local = True, disk_dir = LOCAL_DATASET_PATH)
	else:
		splits = load_dataset_by_splits()
	# - load model
	model = get_image_vqvae(ema = True)
	model.compile(
		optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
		metrics = ["mse"])
	# - iterate through each split
	for split in splits:
		# - run training according to the add_loss method in Keras
		image_ds = fetch_images_of_dataset(split)
		ds_iterator = iter_dataset_by_batch(image_ds, only_images = True, batch_size = BATCH_SIZE)
		model.fit(x = ds_iterator)
		# TODO: model checkpoint
	

if __name__ == "__main__":
	main()