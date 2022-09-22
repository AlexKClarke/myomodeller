import torch


def train_valid_test_split(data, data_splits=[0.6, 0.8]):
    num_train = int(data.shape[0] * data_splits[0])
    num_val = int(data.shape[0] * data_splits[1]) - num_train
    num_test = data.shape[0] - num_train - num_val
    return data.split([num_train, num_val, num_test], 0)


def get_params(data):
    mean, std = data.mean(0, keepdims=True), data.std(0, keepdims=True)
    std = torch.tensor([1 if s == 0 else s for s in std.flatten()]).reshape(std.shape)
    return mean, std


def standardise(data, mean, std):
    return (data - mean) / std


# ------------------------------------------------------------------------------


def get_standardised_library(data, targets, data_splits=[0.6, 0.8]):
    library = {}

    train_data, val_data, test_data = train_valid_test_split(data, data_splits)
    mean, std = get_params(train_data)
    library["train_data"] = standardise(train_data, mean, std)
    library["valid_data"] = standardise(val_data, mean, std)
    library["test_data"] = standardise(test_data, mean, std)

    train_targets, val_targets, test_targets = train_valid_test_split(
        targets, data_splits
    )
    library["train_targets"] = train_targets
    library["valid_targets"] = val_targets
    library["test_targets"] = test_targets

    return library
