import csv
import h5py
import numpy as np
import tensorflow as tf


def save_log(log, log_name, fieldnames, save_dir):
    '''
    Saves training logs
    '''
    with open(save_dir.joinpath(log_name + '.csv'), mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log)

def load_test_dataset(test_file):
    '''
    Loads hdf5 dataset file to python dictionaries

    :param test_file: File Path to test file

    :return: X, E, Ux and Uy arrays of test data

    '''
    with h5py.File(test_file, 'r') as data_file:
        x = data_file["testing/X_test"][()]
        E = data_file["testing/E_test"][()]
        u = data_file["testing/u_test"][()]
        if len(E.shape) == 3:
            E = E[:, :, 0]
        ux = u[:, :, 0]
        uy = u[:, :, 1]

    return x, E, ux, uy
def load_train_dataset(filename):
    '''
    Loads hdf5 dataset file to python dictionaries

    :param dir_name: Folder Name where dataset file can be found

    :return: Train and Boundary datasets and Test Points to evaluate generator progress

    '''
    with h5py.File(filename, 'r') as f:
        train_data = {}
        boundary_data = {}
        for key in f['data'].keys():
            if key == 'snapshots':
                snapshots = f['data'][key]
                idxs = random_idxs(snapshots.shape[0], snapshots.shape[0])
                train_data.update({key : tf.Variable(snapshots[idxs, :])})
            elif key == 'boundary_conditions':
                for k in f['data'][key]:
                    boundary_data.update({k: tf.Variable(f['data'][key][k])})
            else:
                train_data.update({key: tf.Variable(f['data'][key])})

        for attr in f['data'].attrs:
            train_data.update({attr: f['data'].attrs[attr]})

        
        return train_data, boundary_data

def save_samples(E, U, X, save_file):
    '''
    Saves generated samples from generator at intermediate and final training checkpoints
    
    :param E: Generated E data
    :param U: Generated U data
    :param X: Generated X data
    :param save_file: Filename to save results to
    '''
    
    with h5py.File(save_file, 'w') as data_file:
        samples_group = data_file.create_group('samples')

        samples_group.create_dataset('X', data=X)
        samples_group.create_dataset('generated_u', data=U.numpy())
        samples_group.create_dataset('generated_E', data=E.numpy())
def random_idxs(idxs, num_to_choose):
    '''
    Returns a sorted list of indexes to use
    '''
    idxs = np.sort(np.random.choice(idxs, num_to_choose, replace=False))
    return idxs
