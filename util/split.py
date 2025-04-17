import random

import matplotlib.pyplot as plt



def train_val_test_split(data, train_size=0.8, val_size=0.15, shuffle=True, random_seed=None):
    """
    Splits data into train, validation, and test sets.

    Args:
        data (list or array-like): The dataset to be split.
        train_size (float): Proportion of the dataset to include in the train split.
        val_size (float): Proportion of the dataset to include in the validation split.
        test_size (float, optional): Proportion of the dataset to include in the test split.
                                     If None, it is set to 1 - train_size - val_size.
        shuffle (bool): Whether to shuffle the data before splitting.
        random_seed (int, optional): Seed for the random number generator.

    Returns:
        train_data, val_data, test_data: The splits of the dataset.
    """

    if random_seed is not None:
        random.seed(random_seed)

    if shuffle:
        data = data.copy()
        random.shuffle(data)

    total_length = len(data)

    train_end = int(train_size * total_length)
    val_end = train_end + int(val_size * total_length)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


def check_distributions(train_data, val_data, test_data):
    train_targets = [sample.y for sample in train_data]
    val_targets = [sample.y for sample in val_data]
    test_targets = [sample.y for sample in test_data]

    plt.figure(figsize=(15, 5))
    plt.hist(train_targets, bins=50, alpha=0.5, label='Train')
    plt.hist(val_targets, bins=50, alpha=0.5, label='Validation')
    plt.hist(test_targets, bins=50, alpha=0.5, label='Test')
    plt.legend(loc='upper right')
    plt.title('Target Variable Distribution in Each Split')
    plt.show()
