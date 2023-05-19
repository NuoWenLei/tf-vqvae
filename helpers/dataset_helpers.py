from concurrent.futures import ThreadPoolExecutor
from functools import partial

import io
import urllib
import requests

from helpers.imports import Image, np, skimage
from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent
from helpers.constants import NUM_THREADS, NUM_RETRIES, NUM_TIMEOUT, BATCH_SIZE, LOCAL_DATASET_PATH, IMAGE_HEIGHT, IMAGE_WIDTH

USER_AGENT = get_datasets_user_agent()

def resize_scale_and_pad(im, height, width):
	"""
	Resizes image while preserving aspect ratio

	Args:
	- im: image to resize (as a tensor-like object)
	- height: desired height
	- width: desired width

	Returns:
	- resized and padded image as a tensor-like object
	"""
	im_h = im.shape[0]
	im_w = im.shape[1]
	height_diff = height - im_h
	width_diff = width - im_w
	min_side = min(height_diff, width_diff)
	if min_side == height_diff:
		scale_factor = height / im_h
	else:
		scale_factor = width / im_w

	rescaled_im = skimage.transform.rescale(im, scale_factor, channel_axis = -1)
	rescaled_h = rescaled_im.shape[0]
	rescaled_w = rescaled_im.shape[1]

	if min_side == height_diff:
		padding = ((0, 0), ((width - rescaled_w) // 2, ((width - rescaled_w) // 2) + ((width - rescaled_w) % 2)), (0, 0))
	else:
		padding = (((height - rescaled_h) // 2, ((height - rescaled_h) // 2) + ((height - rescaled_h) % 2)), (0, 0), (0, 0))

	padded_im = np.pad(rescaled_im, padding, constant_values = 0.5)
	return padded_im

def fetch_single_image(image_url, timeout=None, retries=0):
	"""
	Fetch an image object from url

	Args:
	- image_url: url of image
	- timeout = None: optional timeout for request
	- retries = 0: number of retries for request

	Returns:
	- Union
		- PIL Image (if request successful)
		- None (if request failed)
	"""
	for _ in range(retries + 1):
		try:
			request = urllib.request.Request(
				image_url,
				data=None,
				headers={"user-agent": USER_AGENT},
			)
			with urllib.request.urlopen(request, timeout=timeout) as req:
				image = Image.open(io.BytesIO(req.read()))
			break
		except Exception:
			image = None
	return image

default_image = np.array(Image.open(io.BytesIO(requests.get("http://static.flickr.com/2723/4385058960_b0f291553e.jpg").content)))
preproc_im = resize_scale_and_pad(default_image, IMAGE_HEIGHT, IMAGE_WIDTH)

def fetch_images(batch, num_threads, timeout=None, retries=0):
	"""
	Asynchronously (multi-thread) fetch images from urls

	Args:
	- batch: batch of huggingface dataset with "image_url" column for image urls
	- timeout = None: optional timeout for request
	- retries = 0: number of retries for request

	Returns:
	- batch of huggingface dataset with:
		- "image_array" column storing images
		- "image_status" column storing request status
	"""
	fetch_single_image_with_args = partial(fetch_single_image, timeout=timeout, retries=retries)
	with ThreadPoolExecutor(max_workers=num_threads) as executor:
		items = []
		status = []
		for item in executor.map(fetch_single_image_with_args, batch["image_url"]):
			if item is None:
				items.append(preproc_im)
				status.append(False)
			else:
				i = np.array(item)
				if (len(i.shape) == 3) and (i.shape[-1] == 3):
					status.append(True)
					items.append(resize_scale_and_pad(i, IMAGE_HEIGHT, IMAGE_WIDTH))
				else:
					status.append(False)
					items.append(preproc_im)
		batch["image_array"] = items
		batch["image_status"] = status
	return batch

def download_dataset_to_local_path(dataset_name = "sbu_captions", disk_dir = LOCAL_DATASET_PATH):
	"""
	DEPRICATED: Use `load_dataset_by_splits` to download the dataset into runtime directly.
	Downloads huggingface dataset to local directory.

	Args:
	- dataset_name = "sbu_captions": name of huggingface dataset
	- disk_dir = LOCAL_DATASET_PATH: directory on disk to save dataset
	"""
	ds = load_dataset(dataset_name)
	ds.save_to_disk(disk_dir)

def load_dataset_by_splits(dataset_name = "sbu_captions", percent_splits = 1, load_local = False, disk_dir = None):
	"""
	Loads a huggingface dataset as percentage splits either locally or via web download.

	Args:
	- dataset_name = "sbu_captions": name of huggingface dataset
	- percent_splits = 1: percentage of data per split of the dataset
	- load_local = False: (load_local = True is DEPRICATED) whether to load from local storage or not
	- disk_dir = None: (load_local = True is DEPRICATED) directory to load locally from

	Returns:
	- huggingface dataset in splits
	"""
	if load_local:
		assert (disk_dir is not None), "local loading must provide directory of dataset"
		ds = load_dataset(disk_dir, split=[
			f'train[{k}%:{k+percent_splits}%]' for k in range(0, 100, percent_splits)])
	else:
		ds = load_dataset(dataset_name, split=[
			f'train[{k}%:{k+percent_splits}%]' for k in range(0, 100, percent_splits)])
	return ds

def fetch_images_of_dataset(ds):
	"""
	Fetch images for a huggingface dataset with the "image_url" column.

	Args:
	- Huggingface dataset with the "image_url" column
	
	Returns:
	- Huggingface dataset filtered by success of image query with
		- "image_array" column storing image data
		- "image_status" column storing query status
	"""
	image_ds = ds.map(fetch_images, batched=True, batch_size=100, fn_kwargs={
    	"num_threads": NUM_THREADS, "timeout": NUM_TIMEOUT, "retries": NUM_RETRIES})
	return image_ds.filter(lambda x: x, input_columns="image_status")

def iter_dataset_by_batch(ds, only_images = False, batch_size = BATCH_SIZE):
	"""
	Creates a generator that iterates through a huggingface dataset once

	Args:
	- ds: huggingface dataset to iterate over
	- only_images = False: whether to return only the image batch or the entire dataset batch
	- batch_size = BATCH_SIZE: size of batch

	Returns:
	- generator object
	"""
	num_batches = len(ds) // batch_size
	for b in range(num_batches):
		batch = ds.filter(lambda _, indices: (indices >= (b * batch_size)) and (indices < ((b+1) * batch_size)), with_indices = True, input_columns = "image_status")
		
		if only_images:
			yield np.asarray(batch["image_array"])
		else:
			yield batch
