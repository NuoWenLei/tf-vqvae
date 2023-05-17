from helpers.model import get_image_vqvae

if __name__ == "__main__":
	vq_vae = get_image_vqvae()
	print(vq_vae.summary())