from os import listdir
from sys import argv

import numpy as np


def load_data(filename):
    return np.loadtxt(filename, delimiter='\t')


def compute_nrmse(gt, mask, imputations):
    # Compute normalized root mean squared error for a column.
    std = gt.std()
    gt = gt[mask]
    imputations = imputations[mask]
    pred = imputations.mean(1)
    return np.sqrt(((pred - gt) ** 2).mean()) / std


def compute_pfc(gt, mask, imputations):
    # Compute the proportion of falsely classified entries.
    imputations = np.round(imputations).astype('int')
    gt = gt[mask]
    imputations = imputations[mask]
    categories = sorted(list(set(imputations.ravel()).union(set(gt.ravel()))))
    imputations_cat = [(imputations == category).sum(1)
                       for category in categories]
    imputations_cat = np.hstack([x.reshape(-1, 1) for x in imputations_cat])
    pred = np.argmax(imputations_cat, 1)
    return (pred != gt).mean()


# parse arguments
# first argument is dataset name
dataset = argv[1]
# other arguments are one-hot max sizes (see README.md for more infromation)
one_hot_max_sizes = map(int, argv[2:])

# read data
groundtruth = load_data('train_test_split/{}_groundtruth.tsv'.format(dataset))
input_data = load_data('train_test_split/{}_train.tsv'.format(dataset))
output_data = load_data('imputations/{}_imputed.tsv'.format(dataset))

# reshape imputation results
results = output_data.reshape(input_data.shape[0], -1, input_data.shape[1])

# define what was imputed
mask = np.isnan(input_data)

# compute NRMSE or PFC for each column
nrmses = []
pfcs = []
for col_id, size in enumerate(one_hot_max_sizes):
    args = groundtruth[:, col_id], mask[:, col_id], results[:, :, col_id]
    if size <= 1:
        nrmse = compute_nrmse(*args)
        nrmses.append(nrmse)
        print('Column %02d, NRMSE: %g' % (col_id + 1, nrmse))
    else:
        pfc = compute_pfc(*args)
        pfcs.append(pfc)
        print('Column %02d, PFC: %g' % (col_id + 1, pfc))

# print average NRMSE and PFC over all columns
print()
print('NRMSE: %g' % (sum(nrmses) / max(1, len(nrmses))))
print('PFC: %g' % (sum(pfcs) / max(1, len(pfcs))))
