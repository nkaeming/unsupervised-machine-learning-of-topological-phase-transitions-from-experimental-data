from typing import Set, Tuple, Dict, List, Any
import h5py
import numpy as np
from itertools import product as combvec

import torch
from torch.utils.data import DataLoader, Dataset


def load_dataset(path: str, parameter_list: Set[str]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Loads a dataset from disk.
    Args:
        path: The path to the hdf5 file.
        parameter_list: A dictionary with the key names in the dataset.

    Returns:
        A dictionary with numpy arrays holding the complete data.
    """
    data = {}
    with h5py.File(path, 'r') as h5f:
        images = h5f['images'][:].astype('float32')
        for parameter_name in parameter_list:
            data[parameter_name] = h5f['parameter/' + parameter_name][:].astype('float32')
    return images, data


def crop_images(images: np.ndarray, crop_size: int) -> np.ndarray:
    """
    Crops quadratic images around origin.
    Args:
        images: an nxmxmx matrix where n is the number of images and m is the size
        crop_size: the crop size

    Returns:
        The image array with the size nxcrop_sizexcrop_size
    """
    image_size = images.shape[1]
    center = image_size/2
    crop_padding = crop_size/2
    crop_start_pixel = int(center-crop_padding)
    crop_end_pixel = int(center+crop_padding)
    return images[:, crop_start_pixel:crop_end_pixel, crop_start_pixel:crop_end_pixel]


def prepare_images_tensorflow(images: np.ndarray) -> np.ndarray:
    """
    Prepares an image ndarray to fit into the tensorflow NHWC
    Args:
        images: The images in an nxmxm format

    Returns:
        The images in the NHWC tensorflow format.
    """
    return images.reshape(*images.shape, 1)


def normalize_complete_images(images: np.ndarray) -> np.ndarray:
    """
    Normalize the given images with respect to all images.
    Args:
        images: THe images in a NHW format.

    Returns:
        The normalized images in a NHW format.
    """
    min_pv = np.min(images)
    # shift pixel values to a minimum of 0
    images = images - min_pv
    # new maximum of the images
    max_pv = np.max(images)
    # the images values are set to 0-1
    return images / max_pv


def normalize_single_images(images: np.ndarray) -> np.ndarray:
    """
    Normalize each image individually to a value range between 0 and 1
    Args:
        images: The images in a NHW format

    Returns:
        The normalized images in a NHW format.
    """
    for idx in range(len(images)):
        images[idx] = normalize_complete_images(images[idx])
    return images


def create_combination_index(parameter: Dict[str, np.ndarray], uniques: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    # the x index of the image
    x_index = np.empty(0, dtype=int)
    # the y index of the image
    y_index = np.empty(0, dtype=int)
    unique_iterator, keys = _multi_uniques(parameter, uniques)
    number_of_entries = len(parameter[keys[0]])
    for unique_point in unique_iterator:
        number_of_parameter = len(unique_point)
        selection_map = np.arange(number_of_entries)
        for idx in range(number_of_parameter):
            selection_map = np.intersect1d(selection_map, np.where(parameter[keys[idx]] == unique_point[idx])[0])

        for idx in selection_map:
            for idy in selection_map:
                x_index = np.append(x_index, idx)
                y_index = np.append(y_index, idy)
    return x_index, y_index


def _multi_uniques(params: Dict[str, np.ndarray], uniques: List[str]) -> Tuple[combvec, List[str]]:
    """
    Creates an iterator for all combinations of unique tuples of the uniques values.
    Args:
        params: All parameters
        uniques: he parameters to be uniquely paired

    Returns:
        (iterator, parameter key list)
    """
    unique_values = {}
    keys = []
    for val in uniques:
        unique_values[val] = np.unique(params[val])
        keys.append(val)
    iterator = combvec(*unique_values.values())
    return iterator, keys

## For influence functions' notebooks

# generate dataset (it transforms numpy arrays to torch tensors)
class NumpyToPyTorch_DataLoader(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X, Y, transform=None):
        """
        Args:
            path
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.X = torch.from_numpy(X).float()    # image
        # self.Y = torch.from_numpy(Y).float()     # label for regression
        self.Y = torch.from_numpy(Y).long()     # label for classification
        # i, j = self.Y.size()[0], self.Y.size()[1]
        # self.Y = self.Y.view(i, 1, j)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        label = self.Y[index]
        img = self.X[index]

        if self.transform:
            img = self.transform(img)

        return img, label

class Downloader(object):
    def __init__(self, data_name, batch):
        self.data_name = data_name
        self.batch = batch
                
        # Follow information in README.md to download these datasets and save them to folder 'data'  
        if self.data_name is 'single_cut_rephased':
            self.training_path = './data/validation_single_cut_rephased.h5'
            self.testing_path = './data/single_cut_rephased.h5'
            self.labels_path = './influence_functions/data_and_masks/validation_single_cut_theoretical_labels.npy'
            self.test_labels_path = './influence_functions/data_and_masks/single_cut_theoretical_labels.npy'
            self.training_mask_path = './influence_functions/data_and_masks/validation_single_cut_training_mask.npy'
            self.validation_mask_path = './influence_functions/data_and_masks/validation_single_cut_validation_mask.npy'
            self.testing_mask_path = './influence_functions/data_and_masks/single_cut_test_mask.npy'
            self.training_mean = 0.46388384212120864
            self.training_std = 0.17022241233110608
        
        if self.data_name is 'single_cut_with_micromotion':
            self.training_path = './data/validation_single_cut_56.h5'
            self.testing_path = './data/single_cut_56.h5'
            self.test_labels_path = './influence_functions/data_and_masks/single_cut_theoretical_labels.npy'
            self.labels_path = './influence_functions/data_and_masks/validation_single_cut_theoretical_labels.npy'
            self.training_mask_path = './influence_functions/data_and_masks/validation_single_cut_training_mask.npy'
            self.validation_mask_path = './influence_functions/data_and_masks/validation_single_cut_validation_mask.npy'
            self.testing_mask_path = './influence_functions/data_and_masks/single_cut_test_mask.npy'
            self.training_mean = 0.07900989899839451
            self.training_std = 0.044015503630440524
        
        if self.data_name is 'phase_diagram_rephased':
            self.training_path = './data/phase_diagram_rephased.h5'
            self.testing_path = './data/phase_diagram_rephased.h5'
            self.labels_path = './influence_functions/data_and_masks/phase_diagram_anomalydetected_labels.npy'
            self.test_labels_path = './influence_functions/data_and_masks/phase_diagram_anomalydetected_labels.npy'
            self.training_mask_path = './influence_functions/data_and_masks/phase_diagram_training_mask.npy'
            self.validation_mask_path = './influence_functions/data_and_masks/phase_diagram_validation_mask.npy'
            self.testing_mask_path = './influence_functions/data_and_masks/phase_diagram_test_mask.npy'
            self.training_mean = 0.47376012469840956
            self.training_std = 0.16998128006793728
            
    def train_loader(self, batch_size = None, shuffle = False):

        if batch_size is None:
            batch_size = self.batch

        exp_data = h5py.File(self.training_path, 'r')
        data = exp_data['images']
        labels = np.load(self.labels_path)
        data_size = data.shape[0]

        # Load the training mask
        mask = np.load(self.training_mask_path)

        data = np.array(data) # h5py does not support fancy indexing, like masks
        self.training_data = data[mask]
        self.train_samples_no = self.training_data.shape[0]

        # Zerocenter normalization (with training data's mean and std)
        # To avoid the leak of data
        self.training_data = (self.training_data - self.training_mean) / self.training_std

        self.training_labels = labels[mask]

        train_set = NumpyToPyTorch_DataLoader(self.training_data, self.training_labels)

        train_loader = DataLoader(train_set,
                          batch_size = batch_size,
                          shuffle = False,
                          num_workers = 1,
                          pin_memory = True # CUDA only, this lets your DataLoader allocate the samples in page-locked memory, which speeds-up the transfer from CPU to GPU during training
                         )
        return train_loader#, mask - it's always the same, thanks to the fixed random seed, so I needed to save it only once

    def test_loader(self, batch_size = None):

        if batch_size is None:
            batch_size = self.batch

        exp_data = h5py.File(self.testing_path, 'r')
        data = exp_data['images']
        labels = np.load(self.test_labels_path)
        data_size = data.shape[0]

        # Load the test mask
        mask = np.load(self.testing_mask_path)

        data = np.array(data) # h5py does not support fancy indexing, like masks
        self.test_data = data[mask]
        self.test_labels = labels[mask]
        self.test_samples_no = self.test_data.shape[0]

        # Zerocenter normalization (with training data's mean and std)
        # To avoid the leak of data
        self.test_data = (self.test_data - self.training_mean) / self.training_std

        test_set = NumpyToPyTorch_DataLoader(self.test_data, self.test_labels)

        test_loader = DataLoader(test_set,
                        batch_size = batch_size,
                        shuffle = False,
                        num_workers = 1,
                        pin_memory=True # CUDA only
                        )
        return test_loader

    def validation_loader(self, batch_size = None):

        if batch_size is None:
            batch_size = self.batch

        exp_data = h5py.File(self.training_path, 'r')
        data = exp_data['images']
        labels = np.load(self.labels_path)
        data_size = data.shape[0]

        # Load the validation mask
        mask = np.load(self.validation_mask_path)

        data = np.array(data) # h5py does not support fancy indexing, like masks
        self.validation_data = data[mask] # 3 991
        self.validation_samples_no = self.validation_data.shape[0]
        self.validation_labels = labels[mask]

        # Zerocenter normalization (with training data's mean and std)
        # To avoid the leak of data
        self.validation_data = (self.validation_data - self.training_mean) / self.training_std

        validation_set = NumpyToPyTorch_DataLoader(self.validation_data, self.validation_labels)

        validation_loader = DataLoader(validation_set,
                        batch_size = batch_size,
                        shuffle = False,
                        num_workers = 1,
                        #pin_memory=True # CUDA only
                        )
        return validation_loader

    def training_samples_no(self):
        # Load the training mask
        mask = np.load(self.training_mask_path)

        return mask.size