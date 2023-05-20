# https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel

from transformers import AutoTokenizer, CLIPTextModel
from helpers.imports import tf
from typing import List

model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

def embed_text_clip(text_batch: List[str], pooler: bool = False):
	"""
	Transform text into CLIP embedding with CLIP text encoder.

	https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel

	Pooler output:
	seq-to-unit where the unit is the output of the last token in the sequence
	Rough visualization:
		   out
	    ^
	    |
	* * * => * * *
	^ ^ ^
	
	Last hidden state:
	seq-to-unit where the unit is the hidden state of the last token in the output sequence
	Rough visualization:
				out
	             ^
	             |
	* * * => * * *
	^ ^ ^

	Args:
	- text_batch, List[str]: batch of text to transform into embeddings
	- pooler, bool = False: whether to use pooler output or last hidden state

	Return:
	- Tensorflow text embedding tensor
	"""
	inputs = tokenizer(text_batch, padding=True, return_tensors="pt")
	outputs = model(**inputs)
	output = outputs.last_hidden_state if pooler else outputs.pooler_output
	return tf.convert_to_tensor(output.numpy())


