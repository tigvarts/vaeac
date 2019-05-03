from os import makedirs
from os.path import join

import numpy as np
import pandas as pd


mcar_prob = 0.5
random_seed = 239


def yeast_loader(path):
    # read and preprocess yeast dataset
    data = [[y for y in x.split(' ') if y][1:]
            for x in open(join(path, 'yeast.data')).read().split('\n') if x]
    target_id = {x: i for i, x in enumerate(set([x[-1] for x in data]))}
    data = [x[:-1] + [target_id[x[-1]]] for x in data]
    data = [[float(y) for y in x] for x in data]
    return np.array(data)


def white_loader(path):
    # read and preprocess white-wine dataset
    data = pd.read_csv(join(path, 'winequality-white.csv'), sep=';')
    return np.array(data)


def mushroom_loader(path):
    # read and preprocess mushroom dataset
    data = pd.read_csv(join(path, 'agaricus-lepiota.data'),
                       header=None, na_values='?')
    target = np.array(data[0] == 'e')
    data.drop(0, axis=1, inplace=True)
    data.drop(16, axis=1, inplace=True)
    categorical_sizes = []
    mtx = []
    for column_name in data.columns:
        column = np.array(pd.get_dummies(data[column_name])).astype('float')
        categorical_sizes.append(column.shape[1])
        column = column.dot(np.arange(column.shape[1]).reshape(-1, 1))
        column[np.array(data[column_name].isnull()), :] = np.nan
        mtx.append(column)
    data = np.hstack(mtx + [target.reshape(-1, 1)])
    categorical_sizes += [2]
    return data


def corrupt_data_mcar(data):
    # return a copy of data with missing values with density mcar_prob
    mask = np.random.choice(2, size=data.shape, p=[mcar_prob, 1 - mcar_prob])
    nw_data = data.copy()
    nw_data[(1 - mask).astype('bool')] = np.nan
    return nw_data


def save_data(filename, data):
    np.savetxt(filename, data, delimiter='\t')


for loader, name in [
    (yeast_loader, 'yeast'),
    (white_loader, 'white'),
    (mushroom_loader, 'mushroom')
]:
    data = loader('original_data')
    np.random.seed(random_seed)
    train_data = corrupt_data_mcar(data)

    makedirs('train_test_split', exist_ok=True)
    save_data(join('train_test_split', '{}_train.tsv'.format(name)),
              train_data)
    save_data(join('train_test_split', '{}_groundtruth.tsv'.format(name)),
              data)
