import tensorflow as tf
import numpy as np
import h5py
import optuna
import pickle
import sys
from datetime import date

from lib.VAE import CVAE
from lib.VAETraining import train_step, compute_loss
import lib.data as hd

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

TEST_RATIO = 0.1
MAX_EPOCHS = 400

# Set this to true if you want to overwrite the saved models.
OVERWRITE_SAVE = False

print("Best trial:")
trial = pickle.load(open('networks/micromotion_rephasing/IIIb_VAE_optuna_trial.obj', 'rb'))

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

print("  Attributes:")
for key, value in trial.user_attrs.items():
    print("    {}: {}".format(key, value))

def set_up_encoder(trial: optuna.Trial, latent_dim: int) -> tf.keras.Model:
    encoder = tf.keras.Sequential()
    encoder.add(tf.keras.layers.InputLayer(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1), name='en_input'))
    
    n_convs = trial.suggest_int('en_n_convs', 2, 4)
    n_fcs = trial.suggest_int('en_n_fcs', 0, 4)
    
    # add convolutional blocks
    for idc in range(n_convs):
        n_filter = trial.suggest_int('en_filter_{}'.format(idc), 50, 200, 10)
        kernel_size = trial.suggest_int('en_kernel_size_{}'.format(idc), 2, 6)
        pool_size = trial.suggest_int('en_pool_size_{}'.format(idc), 2, 4)
        encoder.add(tf.keras.layers.Conv2D(n_filter, kernel_size=kernel_size,
                                           activation=None, padding='same',
                                           name='en_conv_{}'.format(idc)))
        encoder.add(tf.keras.layers.BatchNormalization()) # for Relu activations use BatchNorm before activation: arXiv:1502.03167
        encoder.add(tf.keras.layers.LeakyReLU())
        encoder.add(tf.keras.layers.MaxPool2D(pool_size, name='en_pool_{}'.format(idc)))
    
    # add dense layers
    encoder.add(tf.keras.layers.Flatten())
    for idf in range(n_fcs):
        n_neurons = trial.suggest_int('en_neurons_{}'.format(idf), 20, 500, 20)
        encoder.add(tf.keras.layers.Dense(n_neurons, activation=None, name='en_fc_{}'.format(idf)))
        encoder.add(tf.keras.layers.BatchNormalization())
        encoder.add(tf.keras.layers.LeakyReLU())
    
    # add a layer for the latent space.
    encoder.add(tf.keras.layers.Dense(latent_dim + latent_dim, name='fc_en_latenet_space'))
    
    return encoder

def set_up_decoder(trial: optuna.Trial, latent_dim: int) -> tf.keras.Model:
    decoder = tf.keras.Sequential()
    decoder.add(tf.keras.layers.InputLayer(input_shape=(latent_dim + 1,), name='de_input'))
    
    n_fcs = trial.suggest_int('de_n_fcs', 0, 4)
    for idf in range(n_fcs):
        n_neurons = trial.suggest_int('de_neurons_{}'.format(idf), 10, 300, 20)
        decoder.add(tf.keras.layers.Dense(n_neurons, activation=None, name='de_fc_{}'.format(idf)))
        decoder.add(tf.keras.layers.BatchNormalization())
        decoder.add(tf.keras.layers.Dropout(0.5))
        decoder.add(tf.keras.layers.LeakyReLU())
    
    
    layer_options = [[2, 2, 2, 7], [2, 2, 7, 2], [2, 7, 2, 2], [7, 2, 2, 2],
                    [4, 2, 7], [4, 7, 2], [2, 7, 4], [7, 4, 2], [7, 2, 4], [2, 4, 7],
                [8, 7], [7, 8], [14, 4], [4, 14], [14, 2, 2], [2, 14, 2], [2, 2, 14]]
    
    # essential fc layer
    layer_option = trial.suggest_int('layer_option', 0, len(layer_options)-1)
    layer_option = layer_options[layer_option]
    de_fc_zdim = trial.suggest_int('de_fc_zdim', 1, 10)
    de_fc_shape = layer_option[0]
    decoder.add(tf.keras.layers.Dense(de_fc_shape*de_fc_shape*de_fc_zdim, activation=None, name='de_fc_shaping'))
    decoder.add(tf.keras.layers.BatchNormalization())
    decoder.add(tf.keras.layers.LeakyReLU())
    
    decoder.add(tf.keras.layers.Reshape(target_shape=(de_fc_shape, de_fc_shape, de_fc_zdim)))
    
    n_tconvs = len(layer_option) - 1
    
    # add transposed convolutional blocks
    for idc in range(n_tconvs):
        n_filter = trial.suggest_int('de_filter_{}'.format(idc), 50, 200, 10)
        kernel_size = trial.suggest_int('de_kernel_size_{}'.format(idc), 2, 6)
        stride = layer_option[idc+1]
        decoder.add(tf.keras.layers.Conv2DTranspose(n_filter, kernel_size=kernel_size, padding='same',
                                                    strides=stride,
                                                    name='de_tconv_{}'.format(idc)))
        decoder.add(tf.keras.layers.BatchNormalization())
        decoder.add(tf.keras.layers.Dropout(0.5))
        decoder.add(tf.keras.layers.LeakyReLU())
    
    # add output layer
    kernel_size = trial.suggest_int('de_output_kernel_size', 2, 6)
    decoder.add(tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=kernel_size, strides=1,
                                                padding='same', name='de_ouput', activation='tanh'))
    
    assert decoder.output_shape == (None, IMAGE_SIZE, IMAGE_SIZE, 1)

    return decoder


def train_model(trial: optuna.Trial):
    BATCH_SIZE = trial.suggest_int('batch_size', 20, 200, 20)
    history = dict({
        'train_loss': [],
        'test_loss': [],
        'train_accuracy': [],
        'test_accuracy': []
    })
    # prepare data
    num_test = int(number_of_training_tuples * TEST_RATIO)
    num_training = number_of_training_tuples - num_test
    complete_dataset = tf.data.Dataset.from_tensor_slices((index_x, index_y)).shuffle(number_of_training_tuples)
    test_dataset = complete_dataset.take(num_test).batch(500)
    training_dataset = complete_dataset.skip(num_test).batch(BATCH_SIZE)

    print("Total number of tuples: {} Using {} for testing and {} for training.".format(number_of_training_tuples, num_test, num_training))
    
    latent_dim = trial.suggest_int('latent_dim', 1, 15)
    
    # set up the model
    encoder = set_up_encoder(trial, latent_dim)
    decoder = set_up_decoder(trial, latent_dim)
    model = CVAE(latent_dim, encoder, decoder)
    
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    # set up the metrics
    train_loss = tf.keras.metrics.Mean('train_loss')
    train_accuracy = tf.keras.metrics.Mean('train_accuracy')
    test_loss = tf.keras.metrics.Mean('test_loss')
    test_accuracy = tf.keras.metrics.Mean('test_accuracy')
    
    for epoch in range(1, MAX_EPOCHS+1):
        # reset all metrics
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        
        tf_images = tf.constant(images)
        # training data loop
        for x_index, y_index in training_dataset:
            x_index = x_index.numpy()
            y_index = y_index.numpy()
            # get data from indexes
            x_ims = tf.gather(tf_images, x_index)
            y_ims = tf.gather(tf_images, y_index)
            delta_mm_phase = parameter['micromotion_phase'][y_index] - parameter['micromotion_phase'][x_index]
            loss = train_step(model, x_ims, y_ims, delta_mm_phase, optimizer)
            dephased_images = model.dephase(x_ims, delta_mm_phase)
            # the accurracy is now the structual similarity
            accuracy = tf.image.ssim(dephased_images, y_ims, max_val=1.0)
            
            train_loss(loss)
            train_accuracy(accuracy)
            
        # test data loop
        for x_index, y_index in test_dataset:
            x_index = x_index.numpy()
            y_index = y_index.numpy()
            # get data from indexes
            x_ims = tf.constant(images[x_index])
            y_ims = tf.constant(images[y_index])
            delta_mm_phase = parameter['micromotion_phase'][y_index] - parameter['micromotion_phase'][x_index]
            loss = compute_loss(model, x_ims, y_ims, delta_mm_phase)
            dephased_images = model.dephase(x_ims, delta_mm_phase)
            accuracy = tf.image.ssim(dephased_images, y_ims, max_val=1.0)
            
            test_loss(loss)
            test_accuracy(accuracy)

        epoch_train_loss = float(train_loss.result())
        epoch_test_loss = float(test_loss.result())
        epoch_train_accuracy = float(train_accuracy.result())
        epoch_test_accuracy = float(test_accuracy.result())
        history['train_loss'].append(epoch_train_loss)
        history['test_loss'].append(epoch_test_loss)
        history['train_accuracy'].append(epoch_train_accuracy)
        history['test_accuracy'].append(epoch_test_accuracy)
        print('Epoch {}: TrainL: {} TrainA: {} TestL: {} TestA: {}'.format(epoch,
                                               epoch_train_loss, epoch_train_accuracy,
                                               epoch_test_loss, epoch_test_accuracy))
        
    return model, history

# load the data
data_source = 'data/phase_diagram_56.h5'

images, parameter = hd.load_dataset(data_source, {'freq', 'hold', 'micromotion_phase', 'phase'})
IMAGE_SIZE = images.shape[1]

# prepare the data
images = hd.normalize_single_images(images)
images = hd.prepare_images_tensorflow(images)

index_x, index_y = hd.create_combination_index(parameter, ['freq', 'phase'])
number_of_training_tuples = len(index_x)

model, history = train_model(trial)

if OVERWRITE_SAVE:
    model.encoder.save('networks/micromotion_rephasing/IIIb_VAE_micromotion_encoder.h5')
    model.decoder.save('networks/micromotion_rephasing/IIIb_VAE_micromotion_decoder.h5')

    with h5py.File('models/IIIb_VAE_micromotion_history.h5', 'w') as hf:
        for key, value in history.items():
            value = np.array(value)
            hf[key] = value
        for key, value in trial.params.items():
            hf.attrs[key] = value

