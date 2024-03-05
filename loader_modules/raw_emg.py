import os
import numpy as np
from scipy.io import loadmat
import torch
from torch.nn.functional import pad, one_hot

from training import LoaderModule
from loader_modules.utils import get_split_indices, split_array_by_indices


class RawEMGLabelled(LoaderModule):
    """Loader module that retrieves a paired EMG / MU set"""

    def __init__(
        self,
        file_path: str,
        test_fraction: float = 0.2,
        group_size: int = 1,
        one_hot_labels: bool = False,
        batch_size: int = 64,
        shuffle_data: bool = True,
        flatten_input: bool = False,
    ):
        (
            train_emg,
            train_labels,
            val_emg,
            val_labels,
            test_emg,
            test_labels
        ) = self._get_data(
            file_path,
            test_fraction,
            group_size,
            one_hot_labels,
            shuffle_data,
            flatten_input,
        )

        super().__init__(
            train_data=[train_emg, train_labels],
            val_data=[val_emg, val_labels],
            test_data=[test_emg, test_labels],
            batch_size=batch_size,
            input_shape=train_emg.shape[1:],
            output_shape=train_emg.shape[1:],
        )


    def _get_data(
        self,
        file_path: str = 'emg_data_folder/gesture_set_1',
        test_fraction: float = 0.2,
        group_size: int = 1,
        shuffle_data: bool = True,
        one_hot_labels: bool = False,
        flatten_input: bool = False
    ):
        print('Loading raw-EMG data...')

        emg_data = []
        labels_data = []
        # load data
        for filename in os.listdir(os.path.join(file_path, 'emg')):
            if not filename.startswith('.DS_Store'):
                #emg
                file_folder = os.path.join(file_path, 'emg', filename)
                data = np.load(file_folder, allow_pickle=True)
                data = np.array(data)
                emg_data.append(data)
                #labels
                labels_folder = os.path.join(file_path, 'labels', 'LABELS_'+filename)
                labels = np.load(labels_folder, allow_pickle=True)
                labels = np.array(labels)
                labels_data.append(labels)

        # convert to arrays
        '''emg_data = np.array(emg_data)
        labels_data = np.array(labels_data)'''

        emg_data = np.concatenate(emg_data, axis=0)
        labels_data = np.concatenate(labels_data)
        # swap axes
        emg_data = np.swapaxes(emg_data, -1, -2)
        #labels_data = np.swapaxes(labels_data, -1, -2)
        # convert to tensors
        emg_data = torch.tensor(emg_data)
        labels_data = torch.tensor(labels_data)
        emg_data = emg_data.unsqueeze(1)
        # convert to float32 for gpu computation
        emg_data = emg_data.to(torch.float32)
        labels_data = labels_data.to(torch.float32)

        # EMG data will be in the shape (num_samples, 1, time, channels)


        # shuffle data
        if shuffle_data:
            num_samples = emg_data.shape[0]
            indices = torch.randperm(num_samples)
            emg_data = emg_data[indices]
            labels_data = labels_data[indices]

        # standardization - this loses information about channels....
        # should compute the variance across the whole emg data instead
        mean = torch.mean(emg_data, dim=-1, keepdim=True)
        # std = torch.std(emg_data, dim=2, keepdim=True) # this loses information about channels....
        # std = torch.std(emg_data, keepdim=True) # this computes std across all channels, time and trials.. information between channels is preserved, but trials are mixed
        std = torch.std(emg_data, dim=[-1, -2], keepdim=True)  # this computes std across all channels and time.. information between channels is preserved and trials are not mixed
        emg_data = (emg_data - mean) / std



        # Split out train, val and test sets using grouped k fold
        train_data, test_data = split_array_by_indices(
            (emg_data, labels_data),
            get_split_indices(
                num_samples=emg_data.shape[0],
                split_fraction=test_fraction,
                group_size=group_size,
            ),
        )
        train_data, val_data = split_array_by_indices(
            train_data,
            get_split_indices(
                num_samples=train_data[0].shape[0],
                split_fraction=0.2,
                group_size=group_size,
            ),
        )
        (
            [train_emg, train_labels],
            [val_emg, val_labels],
            [test_emg, test_labels],
        ) = [train_data, val_data, test_data]

        # Reduce one hot labels if needed
        #todo: implement one hot labels
        '''if not one_hot_labels:
            [train_labels, val_labels, test_labels] = [
                label.argmax(1)
                for label in [train_labels, val_labels, test_labels]
            ]'''

        # Z-score standardise emg sets with statistics from train set
        #todo: understand here better
        '''mean = train_emg.mean((0, -1), keepdims=True)
        std = train_emg.std((0, -1), keepdims=True)
        [train_emg, val_emg, test_emg] = [
            ((e - mean) / std) for e in [train_emg, val_emg, test_emg]
        ]'''

        print('Training samples: ', len(train_emg))
        print('Test samples: ', len(test_emg))
        print('Validation samples: ', len(val_emg))

        return (
            train_emg,
            train_labels,
            val_emg,
            val_labels,
            test_emg,
            test_labels,
        )
