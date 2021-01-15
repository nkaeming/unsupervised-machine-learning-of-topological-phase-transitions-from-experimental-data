import os
import torch
import six
import inspect
import collections
import numpy as np

np.set_printoptions(threshold=10010)

def flatten_grad(grad):
    tuple_to_list = []
    for tensor in grad:
        tuple_to_list.append(tensor.view(-1))
    all_flattened = torch.cat(tuple_to_list)
    return all_flattened

def is_equal(a, b, epsilon):
    if ((a > (b - epsilon)) and (a < (b + epsilon)) ):
        return True
    else:
        return False

def save_to_file(list, filename, folder_name = None):
    if folder_name is None:
        folder_name = 'tmp'

    if isinstance(list, torch.Tensor):
        if list.requires_grad is True:
            list = list.detach().numpy()

    list_to_string = np.array2string(np.asarray(list), separator=' ', max_line_width=np.inf)
    list_wo_brackets = list_to_string.translate({ord(i): None for i in '[]'})
    file = open(folder_name + '/' + filename, 'w')
    file.write(list_wo_brackets)
    file.close()

def append_to_file(list, filename, folder_name = None, delimiter = ' '):
    if folder_name is None:
        folder_name = 'tmp'
    
    if isinstance(list, torch.Tensor):
        if list.requires_grad is True:
            list = list.detach().numpy()

    list_to_string = np.array2string(np.asarray(list), separator=delimiter, max_line_width=np.inf)
    list_wo_brackets = list_to_string.translate({ord(i): None for i in '[]'})

    file = open(folder_name + '/' + filename, 'a')
    file.write("\n")
    file.close()

    file = open(folder_name + '/' + filename, 'a')
    file.write(list_wo_brackets)
    file.close()

def make_dir_one(path):
    if not os.path.exists(path):
        os.makedirs(path)

def make_dir(path):
    separated_path = path.split('/')
    tmp_path = ''
    for directory in separated_path:
        tmp_path = tmp_path + directory + '/'
        if directory == '.':
            continue
        make_dir_one(tmp_path)
    return True

def find_files(path, affix_flag=False):
    if path[-1] == '/':
        path = path[:-1]
    if affix_flag is False:
        return [path + '/' + name for name in os.listdir(path)]
    else:
        return [name for name in os.listdir(path)]

def remove_slash(path):
    return path[:-1] if path[-1] == '/' else path

# nD list to 1D list
def flatten(list):
    if isinstance(list, collections.Iterable):
        return [a for i in list for a in flatten(i)]
    else:
        return [list]

# Print keys of h5py dataset
def keys(f):
    return [key for key in f.keys()]