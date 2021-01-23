import tensorflow as tf
import numpy as np
from lib.VAE import CVAE

@tf.function
def log_normal_pdf(sample, mean, logvar, raxis=1) -> tf.Tensor:
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis)

@tf.function
def compute_loss(model: CVAE, x, y, question):
    return -tf.reduce_mean(compute_unreduced_loss(model, x, y, question))


@tf.function
def compute_unreduced_loss(model: CVAE, x, y, question):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z, question)

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y)
    logpx_z = -tf.reduce_sum(cross_entropy, axis=(1, 2, 3))
    logpz = log_normal_pdf(z, 0.0, 0.0)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return logpx_z + logpz - logqz_x


def train_step(model: CVAE, x, y, question, optimizer: tf.optimizers.Optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, y, question)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
