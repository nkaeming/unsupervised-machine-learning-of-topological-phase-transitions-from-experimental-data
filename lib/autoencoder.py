from tensorflow.keras import Model

class Autoencoder(Model):
    """This class is the basse class for autoencoders
    """
    def __init__(self, encoder, decoder):
        """Initialize the autoencoders.

        Args:
            encoder (tensorflow.keras.models.Sequential): The encoder model
            decoder (tensorflow.keras.models.Sequential): The decoder model
        """
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        """Encodes the data x and returns the latent space representation of the values in x

        Args:
            x (tensorflow.Tensor): The Data

        Returns:
            (tensorflow.Tensor): The latenent space representation
        """
        return self.encoder(x)

    def decode(self, latent_x):
        """Decodes encoded latent space data.

        Args:
            latent_x (tensorflow.Tensor): The latent space representation of the data   

        Returns:
            tensorflow.Tensor: the real space representation
        """
        return self.decoder(latent_x)

    def call(self, x, training=False):
        latent_x = self.encode(x)
        return self.decode(latent_x)