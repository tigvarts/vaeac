#!/bin/bash

mkdir -p original_data
cd original_data
wget http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data
wget http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
wget http://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data
cd ..
