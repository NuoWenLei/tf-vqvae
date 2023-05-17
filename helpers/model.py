import tensorflow as tf
from helpers.constants import *
from helpers.quantizers import *

class ResBlock(tf.keras.layers.Layer):
	"""
	Convolutional Residual Block for Convolutional Variational Auto-Encoder
	"""
	def __init__(self, out_channels, mid_channels=None, bn=False, name = None):
		super(ResBlock, self).__init__(name = name)

		if mid_channels is None:
			mid_channels = out_channels

		layers = [
			tf.keras.layers.Activation(tf.keras.activations.relu),
			tf.keras.layers.Conv2D(mid_channels, kernel_size = 3, strides = 1, padding = "same"),
			tf.keras.layers.Activation(tf.keras.activations.relu),
			tf.keras.layers.Conv2D(out_channels, kernel_size = 1, strides = 1, padding = "valid")
		]

		if bn:
			layers.insert(2, tf.keras.layers.BatchNormalization())

		self.convs = tf.keras.Sequential(layers)

	def call(self, x):
		# Residual output
		return x + self.convs(x)




def get_encoder(latent_dim=EMBEDDING_DIM, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), batchnorm = True, name="encoder"):
	"""
	Construct Convolutional Encoder
	Args:
		- latent_dim = EMBEDDING_DIM: embedding size for auto-encoder
		- batchnorm = True: use of BatchNormalization layers in residual blocks
		- name = "encoder": name of model
	Returns:
		- tensorflow.keras.Model
	"""
	encoder_inputs = tf.keras.Input(shape=input_shape)

	res1 = ResBlock(latent_dim, bn = batchnorm, name = f"{name}_resblock1")(encoder_inputs)
	resnorm1 = tf.keras.layers.BatchNormalization()(res1)

	conv1 = tf.keras.layers.Conv2D(latent_dim, 4, strides = 2, padding = "same")(resnorm1)
	norm1 = tf.keras.layers.BatchNormalization()(conv1)
	relu1 = tf.keras.activations.relu(norm1)

	res2 = ResBlock(latent_dim, bn = batchnorm, name = f"{name}_resblock2")(relu1)
	resnorm2 = tf.keras.layers.BatchNormalization()(res2)

	conv2 = tf.keras.layers.Conv2D(latent_dim, 4, strides = 2, padding = "same")(resnorm2)
	norm2 = tf.keras.layers.BatchNormalization()(conv2)
	relu2 = tf.keras.activations.relu(norm2)

	res3 = ResBlock(latent_dim, bn = batchnorm, name = f"{name}_resblock3")(relu2)
	
	encoder_outputs = tf.keras.layers.BatchNormalization()(res3)

	return tf.keras.Model(encoder_inputs, encoder_outputs, name=name)


def get_decoder(input_shape, latent_dim=EMBEDDING_DIM, num_channels = 3, batchnorm = True, name="decoder"):
	"""
	Constructs Convolutional Decoder
	Args:
		- input_shape: input shape of decoder
		- latent_dim = EMBEDDING_DIM: embedding size for auto-encoder
		- num_channels = 3: number of output channels (RGB)
		- name = "decoder": name of model
	Returns:
		- tensorflow.keras.Model
	"""
	decoder_inputs = tf.keras.Input(shape=input_shape)

	res1 = ResBlock(latent_dim, bn = batchnorm, name = f"{name}_resblock1")(decoder_inputs)
	resnorm1 = tf.keras.layers.BatchNormalization()(res1)

	conv1 = tf.keras.layers.Conv2DTranspose(latent_dim, kernel_size = 4, strides = 2, padding = "same")(resnorm1)
	norm1 = tf.keras.layers.BatchNormalization()(conv1)
	relu1 = tf.keras.activations.relu(norm1)

	res2 = ResBlock(latent_dim, bn = batchnorm, name = f"{name}_resblock2")(relu1)
	resnorm2 = tf.keras.layers.BatchNormalization()(res2)

	conv2 = tf.keras.layers.Conv2DTranspose(latent_dim, kernel_size = 4, strides = 2, padding = "same")(resnorm2)
	norm2 = tf.keras.layers.BatchNormalization()(conv2)
	relu2 = tf.keras.activations.relu(norm2)

	conv3 = tf.keras.layers.Conv2D(latent_dim, kernel_size = 4, padding = "same")(relu2)
	norm3 = tf.keras.layers.BatchNormalization()(conv3)

	decoder_outputs = ResBlock(num_channels, bn = batchnorm, name = f"{name}_resblock3")(norm3)

	decoder_tanh = tf.keras.activations.tanh(decoder_outputs)

	return tf.keras.Model(decoder_inputs, decoder_tanh, name=name)

def get_image_vqvae(latent_dim=EMBEDDING_DIM, num_embeddings=NUM_EMBEDDINGS, image_shape=(IMAGE_HEIGHT, IMAGE_WIDTH), num_channels = 3, ema = True, name = "vq_vae"):
	"""
	Constructs VQ-VAE for Images
	Args:
		- latent_dim = EMBEDDING_DIM: embedding size for auto-encoder
		- num_embeddings = NUM_EMBEDDINGS: number of codes in the codebook
		- num_channels = 3: number of output channels (RGB)
		- ema = True: use Vector Quantizer Exponential Moving Average or normal
		- name = "vq_vae": name of model
	Returns:
		- tensorflow.keras.Model
	"""
	if ema:
		vq_layer = VectorQuantizerEMA(
			embedding_dim = latent_dim, 
			num_embeddings = num_embeddings,
			commitment_cost=COMMITMENT_COST,
			decay=DECAY,
			name="vector_quantizer")
	else:
		vq_layer = VectorQuantizer(
			embedding_dim = latent_dim, 
			num_embeddings = num_embeddings,
			commitment_cost=COMMITMENT_COST,
			name="vector_quantizer")
	encoder = get_encoder(latent_dim = latent_dim, input_shape=image_shape + (num_channels,))
	inputs = tf.keras.Input(shape=image_shape + (num_channels,))
	encoder.build(image_shape + (num_channels,))
	encoder_outputs = encoder(inputs)
	decoder = get_decoder(encoder.output_shape[1:], latent_dim = latent_dim)
	quantized_latents = vq_layer(encoder_outputs)
	reconstructions = decoder(quantized_latents)
	vq_vae = tf.keras.Model(inputs, reconstructions, name=name)
	vq_vae.build(image_shape + (num_channels,))
	return vq_vae