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
	ds = load_dataset(dataset_name)
	ds.save_to_disk(disk_dir)

def load_dataset_by_splits(dataset_name = "sbu_captions", percent_splits = 1, load_local = False, disk_dir = None):
	if load_local:
		assert (disk_dir is not None), "local loading must provide directory of dataset"
		ds = load_dataset(disk_dir, split=[
			f'train[{k}%:{k+percent_splits}%]' for k in range(0, 100, percent_splits)])
	else:
		ds = load_dataset(dataset_name, split=[
			f'train[{k}%:{k+percent_splits}%]' for k in range(0, 100, percent_splits)])
	return ds

def fetch_images_of_dataset(ds):
	image_ds = ds.map(fetch_images, batched=True, batch_size=100, fn_kwargs={
    	"num_threads": NUM_THREADS, "timeout": NUM_TIMEOUT, "retries": NUM_RETRIES})
	return image_ds.filter(lambda x: x, input_columns="image_status")

def iter_dataset_by_batch(ds, only_images = False, batch_size = BATCH_SIZE):
	num_batches = len(ds) // batch_size
	for b in range(num_batches):
		batch = ds.filter(lambda _, indices: (indices >= (b * batch_size)) and (indices < ((b+1) * batch_size)), with_indices = True, input_columns = "image_status")
		
		if only_images:
			yield np.asarray(batch["image_array"])
		else:
			yield batch

# def load_sbu_and_return_gen(load_local_path = None, save_local_path = None, with_indices = False, only_images = True):
# 	if load_local_path is not None:
# 		ds = load_dataset_by_splits(
# 			dataset_name="sbu_captions",
# 			percent_splits=1,
# 			load_local = True,
# 			disk_dir=load_local_path
# 		)
# 	else:
# 		if save_local_path is not None:
# 			ds = load_dataset_by_splits(
# 				dataset_name="sbu_captions",
# 				percent_splits=1,
# 				load_local = False,
# 				disk_dir=save_local_path,
# 				save_local = True
# 			)
# 		else:
# 			ds = load_dataset_by_splits(
# 				dataset_name="sbu_captions",
# 				percent_splits=1,
# 				load_local = False,
# 				save_local = False
# 			)
	
# 	image_ds = fetch_images_of_dataset(ds)
# 	return iter_dataset_by_batch(image_ds, with_indices = with_indices, only_images = only_images)
	