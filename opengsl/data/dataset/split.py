'''
This file is to split Coauthor/Amazon dataset
'''

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch


def sample_per_class(labels, num_examples_per_class, forbidden_indices=None):
    num_samples = len(labels)
    num_classes = labels.max() + 1
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index] == class_index:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [np.random.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])


def get_split_per_class(labels, train_examples_per_class=None, val_examples_per_class=None, test_examples_per_class=None,
                        train_size=None, val_size=None, test_size=None, **kwargs):
    num_samples = len(labels)
    num_classes = labels.max() + 1
    remaining_indices = np.arange(num_samples)

    if train_examples_per_class is not None:
        train_indices = sample_per_class(labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices, _ = train_test_split(remaining_indices, train_size=train_size, stratify=labels)
        # train_indices = np.random.choice(remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices, _ = train_test_split(remaining_indices, train_size=val_size, stratify=labels[remaining_indices])
        # val_indices = np.random.choice(remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))

    if test_examples_per_class is not None:
        test_indices = sample_per_class(labels, test_examples_per_class, forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices, _ = train_test_split(remaining_indices, train_size=test_size, stratify=labels[remaining_indices])
        # test_indices = np.random.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate((train_indices, val_indices, test_indices))) == num_samples

    return train_indices, val_indices, test_indices


def get_split_ratio(labels, train_size=0.6, val_size=0.2, test_size=None, **kwargs):
    if test_size is None:
        test_size = 1 - train_size - val_size
    assert train_size + val_size + test_size == 1
    n_val = round(val_size * len(labels))
    indices = np.arange(len(labels))
    train_indices, val_test_indices = train_test_split(indices, train_size=train_size, stratify=labels)
    val_indices, test_indices = train_test_split(val_test_indices, train_size=n_val, stratify=labels[val_test_indices])

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))

    assert len(np.concatenate((train_indices, val_indices, test_indices))) == len(labels)

    return train_indices, val_indices, test_indices


def get_split(labels, split_params):
    if 'train_examples_per_class' in split_params:
        return get_split_per_class(labels, **split_params)
    else:
        return get_split_ratio(labels, **split_params)


def k_fold(labels, folds):
    skf = StratifiedKFold(folds, shuffle=True)
    test_indices, train_indices = [], []
    for _, idx in skf.split(np.zeros(len(labels)), labels):
        test_indices.append(idx.astype(np.int64))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = np.ones(len(labels), dtype=np.int64)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero()[0])

    return train_indices, test_indices, val_indices


if __name__ == '__main__':
    from opengsl.data.dataset.pyg_load import pyg_load_dataset
    np.random.seed(0)
    dataset = pyg_load_dataset('citeseer_full')
    print(dataset[0], dataset.num_classes)
    train_indices, val_indices, test_indices = k_fold(dataset[0].y, 10)
    print(len(train_indices[0]))
    print(train_indices[0])
    print(len(val_indices[0]))
    print(val_indices[0])
    print(len(test_indices[0]))
    print(test_indices[0])