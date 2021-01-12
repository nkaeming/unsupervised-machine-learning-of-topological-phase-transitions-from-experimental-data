import tensorflow as tf


class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim, encoder, decoder):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.decoder = decoder

    @tf.function
    def dephase(self, x, phase):
        mean, logvar = self.encode(x)
        reparam = self.reparameterize(mean, logvar)
        return self.decode(reparam, phase, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, phase, apply_sigmoid=False):
        phase = tf.expand_dims(phase, 1)
        logits = self.decoder(tf.concat((z, phase), axis=1))
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


class CVAEStride(CVAE):
    def __init__(self, latent_dim: int, image_size: int, kernel_size: int = 3,
                 en_filter_1: int = 64, en_filter_2: int = 81, en_filter_3: int = 128,
                 de_filter_1: int = 128, de_filter_2: int = 81, de_filter_3: int = 64,
                 decoder_fc: int = 32):
        encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(image_size, image_size, 1)),
                tf.keras.layers.Conv2D(
                    filters=en_filter_1, kernel_size=kernel_size, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=en_filter_2, kernel_size=kernel_size, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=en_filter_3, kernel_size=kernel_size, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim + 1,)),
                tf.keras.layers.Dense(units=7 * 7 * decoder_fc, activation='relu'),
                tf.keras.layers.Reshape(target_shape=(7, 7, decoder_fc)),
                tf.keras.layers.Conv2DTranspose(
                    filters=de_filter_1, kernel_size=kernel_size, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=de_filter_2, kernel_size=kernel_size, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=de_filter_3, kernel_size=kernel_size, strides=2, padding='same',
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=kernel_size, strides=1, padding='same'),
            ]
        )

        super(CVAEStride, self).__init__(latent_dim, encoder, decoder)


class CVAEPooling(CVAE):
    def __init__(self, latent_dim: int, image_size: int, kernel_size: int = 3,
                 en_filter_1: int = 64, en_filter_2: int = 81, en_filter_3: int = 128,
                 de_filter_1: int = 128, de_filter_2: int = 81, de_filter_3: int = 64,
                 decoder_fc: int = 32):
        encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(image_size, image_size, 1)),  # 56 56 1
                tf.keras.layers.Conv2D(
                    filters=en_filter_1, kernel_size=kernel_size, strides=(1, 1), activation='relu'),  # 56 56 64
                tf.keras.layers.MaxPool2D((2, 2)),  # 28 28 64
                tf.keras.layers.Conv2D(
                    filters=en_filter_2, kernel_size=kernel_size, strides=(1, 1), activation='relu'),  # 28 28 64
                tf.keras.layers.MaxPool2D((2, 2)),  # 14 14 64
                tf.keras.layers.Conv2D(
                    filters=en_filter_3, kernel_size=kernel_size, strides=(1, 1), activation='relu'),  # 14 14 128
                tf.keras.layers.MaxPool2D((2, 2)),  # 7 7 128
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim + 1,)),
                tf.keras.layers.Dense(units=7 * 7 * decoder_fc, activation='relu'),
                tf.keras.layers.Reshape(target_shape=(7, 7, decoder_fc)),  # 7 7 32
                tf.keras.layers.Conv2DTranspose(
                    filters=de_filter_1, kernel_size=kernel_size, strides=2, padding='same', activation='relu'),  # 14 14 128
                tf.keras.layers.Conv2DTranspose(
                    filters=de_filter_2, kernel_size=kernel_size, strides=2, padding='same', activation='relu'),  # 28 28 81
                tf.keras.layers.Conv2DTranspose(
                    filters=de_filter_3, kernel_size=kernel_size, strides=2, padding='same', activation='relu'),  # 56 56 64
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=kernel_size, strides=1, padding='same'),  # 56 56 1
            ]
        )

        super(CVAEPooling, self).__init__(latent_dim, encoder, decoder)
