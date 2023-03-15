from typing import Optional, Tuple, Union, Sequence, List, Dict

import torch
import numpy as np
from sklearn.model_selection import GroupKFold


def get_split_indices(
    targets: Optional[Union[torch.Tensor, np.ndarray]] = None,
    num_samples: Optional[int] = None,
    split_fraction: float = 0.2,
    group_size: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Wrapper function for sklearn GroupKFold, returning two arrays of
    indices based on the split fraction. As GroupKFold uses a deterministic
    split, it should always return the same splits given the same inputs (as
    if the seed was fixed).

    Arguments:

    -targets
        Given a tensor/array of targets, will return two
        arrays of 1D indices that gives a roughly equal proportion of each
        class. This will not work for one-hot labels, please argmax first.
    -num_samples
        Gives a random split instead of size num_samples
        (for unsupervised learning). Cannot be used if targets is used.
    -split_fraction
        Fraction to split out. The fist array is the remaining indices and the
        second array is the split out indices. E.g. a split fraction of
        0.2 with 100 samples would return arrays of sizes 80 and 20.
        Because GroupKFold requires splits to be specified as
        folds, there will be some rounding i.e. round(1/split_fraction)
    -group_size
        The group_size argument specifies the number of contiguous samples to
        keep in each split, which is important if the data is time series.
        Default is 1, i.e. no grouping.

    """

    if targets is None:
        assert num_samples is not None, "Please specify num_samples."
        targets = np.ones((num_samples, 1))
    else:
        assert (
            num_samples is None
        ), "Please only specify either targets or num_samples, not both."

    num_groups = targets.shape[0] // group_size
    leftover = targets.shape[0] % group_size

    groups = np.arange(num_groups).repeat(group_size)
    groups = np.concatenate((groups, groups[-1] * np.ones(leftover)))

    num_folds = int(np.round(1 / split_fraction))

    return list(GroupKFold(num_folds).split(targets, targets, groups))[0]


def split_array_by_indices(
    inputs: Sequence[Union[torch.Tensor, np.ndarray]],
    indices: Sequence[np.ndarray],
) -> List[List[Union[torch.Tensor, np.ndarray]]]:
    """Splits lists of input tensors/arrays into a lists of tensor/arrays based
    on a list of indices"""

    out = []
    for idx in indices:
        out.append([input[idx] for input in inputs])

    return out


def array_to_tensor(
    input: Union[np.ndarray, Sequence[np.ndarray]]
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Converts numpy array or list of numpy arrays to torch tensors"""

    if isinstance(input, np.ndarray):
        return torch.from_numpy(input)
    else:
        return [torch.from_numpy(inp) for inp in input]
