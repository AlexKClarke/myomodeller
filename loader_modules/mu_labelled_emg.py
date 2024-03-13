from scipy.io import loadmat
import torch
from torch.nn.functional import pad, one_hot

from training import LoaderModule
from loader_modules.utils import get_split_indices, split_array_by_indices


class MotorUnitLabelledEMG(LoaderModule):
    """Loader module that retrieves a paired EMG / MU set"""

    def __init__(
        self,
        mat_path: str,
        half_window_size: int = 50,
        test_fraction: float = 0.2,
        group_size: int = 100,
        one_hot_labels: bool = False,
        batch_size: int = 64,
        weighted_sampler: bool = True,
        train_sample_weighting: float = 1.0,
        test_sample_weighting: float = 1.0,
        flatten_samples: bool = False,
    ):
        (
            train_emg,
            train_labels,
            val_emg,
            val_labels,
            test_emg,
            test_labels,
            input_shape,
            output_shape,
        ) = self._get_data(
            mat_path,
            half_window_size,
            test_fraction,
            group_size,
            one_hot_labels,
            flatten_samples,
        )

        super().__init__(
            train_data=[train_emg, train_labels],
            val_data=[val_emg, val_labels],
            test_data=[test_emg, test_labels],
            batch_size=batch_size,
            weighted_sampler=weighted_sampler,
            train_sample_weighting=train_sample_weighting,
            test_sample_weighting=test_sample_weighting,
            input_shape=input_shape,
            output_shape=output_shape,
        )

    def _get_data(
        self,
        mat_path,
        half_window_size,
        test_fraction,
        group_size,
        one_hot_labels,
        flatten_samples,
    ):
        """Loads paired EMG and MUs from demuse file"""

        # Get emg and motor unit timestamps
        mat = loadmat(mat_path)
        emg = torch.stack(
            [
                torch.from_numpy(c.squeeze())
                for c in mat["SIG"].flatten()
                if c.size != 0
            ],
            1,
        ).to(torch.float32)
        stamps = [m.squeeze() for m in mat["MUPulses"].squeeze()]

        # Embed timestamps in binary data
        labels = torch.concat(
            [
                torch.zeros([emg.shape[0], len(stamps)]),
                torch.ones([emg.shape[0], 1]),
            ],
            1,
        ).to(dtype=torch.int64)
        for i, s in enumerate(stamps):
            labels[s, i] = 1
            labels[s, -1] = 0

        # Pull out windows from emg of size 2 * half_window_size
        emg = pad(emg, [0, 0, half_window_size, half_window_size])
        emg = torch.stack(
            [
                emg.roll(int(shift), 0)[half_window_size:-half_window_size]
                for shift in torch.arange(-half_window_size, half_window_size)
            ]
        ).permute([1, 2, 0])

        if flatten_samples:
            emg = emg.flatten(1)

        # Split out train, val and test sets using grouped k fold
        train_data, test_data = split_array_by_indices(
            (emg, labels),
            get_split_indices(
                num_samples=emg.shape[0],
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
        if not one_hot_labels:
            [train_labels, val_labels, test_labels] = [
                label.argmax(1)
                for label in [train_labels, val_labels, test_labels]
            ]

        # Z-score standardise emg sets with statistics from train set
        mean = train_emg.mean((0, -1), keepdims=True)
        std = train_emg.std((0, -1), keepdims=True)
        [train_emg, val_emg, test_emg] = [
            ((e - mean) / std) for e in [train_emg, val_emg, test_emg]
        ]

        return (
            train_emg,
            train_labels,
            val_emg,
            val_labels,
            test_emg,
            test_labels,
            train_emg.shape[1:],
            [len(stamps) + 1],
        )
