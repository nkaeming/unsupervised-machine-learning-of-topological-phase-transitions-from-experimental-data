from typing import Set, Tuple, Dict, List, Any
import h5py
import numpy as np
from itertools import product as combvec


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
