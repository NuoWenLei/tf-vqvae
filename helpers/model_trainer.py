from helpers.imports import tf
from helpers.model import get_image_vqvae
from helpers.constants import *

class VQVAETrainer(tf.keras.models.Model):
    def __init__(self,
                 latent_dim=EMBEDDING_DIM,
                 num_embeddings=NUM_EMBEDDINGS,
                 image_shape=(IMAGE_HEIGHT, IMAGE_WIDTH),
                 use_ema = True,
                 use_batchnorm = True,
                 name = "vqvae_trainer"):
        super().__init__(name = name)
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        self.vqvae = get_image_vqvae(
            latent_dim = self.latent_dim,
            num_embeddings = self.num_embeddings,
            image_shape = image_shape,
            num_channels = 3,
            ema = use_ema,
            batchnorm = use_batchnorm)

        self.total_loss_tracker = tf.keras.metrics.Mean(name=f"{name}_total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name=f"{name}_reconstruction_loss"
        )
        self.vq_loss_tracker = tf.keras.metrics.Mean(name=f"{name}_vq_loss")

        self.data_variance_tracker = tf.keras.metrics.Mean(name=f"{name}_data_variance")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    def train_step(self, x):
        variance = tf.math.reduce_variance(x)
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)

            # Calculate the losses.
            reconstruction_loss = (
                tf.reduce_mean((x - reconstructions) ** 2) / variance
            )
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))
        self.data_variance_tracker.update_state(variance)

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
            "data_variance": self.data_variance_tracker.result()
        }