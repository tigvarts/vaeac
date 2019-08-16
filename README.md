# Variational Autoencoder with Arbitrary Conditioning

Variational Autoencoder with Arbitrary Conditioning (VAEAC) is
a neural probabilistic model based on variational autoencoder
that can be conditioned on an arbitrary subset of observed features and
then sample the remaining features.

For more detail, see the following paper:\
Oleg Ivanov, Michael Figurnov, Dmitry Vetrov.
Variational Autoencoder with Arbitrary Conditioning, ICLR 2019,
[link](https://openreview.net/forum?id=SyxtJh0qYm).

This PyTorch code implements the model and reproduces the results
from the paper.

## Setup

Install prerequisites from `requirements.txt`.
This code was tested on Linux (but it should work on Windows as well),
Python 3.6.4 and PyTorch 1.0.

To run experiments with CelebA download dataset into some directory,
unzip `img_align_celeba.zip` and set correct `celeba_root_dir`
(i. e. which points to the root of the unzipped folder) in file `datasets.py`.

## Experiments

## Missing Feature Multiple Imputation

To impute missing features with VAEAC one can use `impute.py`.

`impute.py` works with real-valued and categorical features.
It takes tab-separated values (tsv) file as an input.
NaNs in the input file indicate the missing features.

The output file is also a tsv file, where for each object
there is `num_imputations` copies of it with NaNs replaced
with different imputations.
These copies with imputations are consecutive in the output file.
For example, if `num_imputations` is 2,
then the output file is structured as follows
```
object1_imputation1
object1_imputation2
object2_imputation1
object2_imputation2
object3_imputation1
...
```
By default `num_imputations` is 5.

One-hot max size is the number of different values of a categorical feature.
The values are assumed to be integers from 0 to K - 1,
where K is one-hot max size.
For the real-valued feature one-hot max size is assumed to be 0 or 1.

For example, for a dataset with a binary feature, three real-valued features
and a categorical feature with 10 classes the correct `--one_hot_max_sizes`
arguments are 2 1 1 1 10.

Validation ratio is the ratio of objects which will be used for validation
and the best model selection.

So the minial working example of calling `impute.py` is
```
python impute.py --input_file input_data.tsv --output_file data_imputed.tsv \
                 --one_hot_max_sizes 2 1 1 1 10 --num_imputations 25 \
                 --epochs 1000 --validation_ratio 0.15
```

Validation IWAE samples is a number of latent samples
for each object IWAE evaluation.

Use last checkpoint flag forces `impute.py` to use the state of the model
at the end of the training procedure for imputation.
By default, the best model according to IWAE validation score is used.

See `python impute.py --help` for more options.

One can reproduce paper results for mushroom, yeast and white wine datasets
by the following commands:
```
cd data
./fetch_data.sh
python prepare_data.py
mkdir -p imputations
python ../impute.py --input_file train_test_split/yeast_train.tsv \
                    --output_file imputations/yeast_imputed.tsv \
                    --one_hot_max_sizes 1 1 1 1 1 1 1 1 10 \
                    --num_imputations 10 --epochs 300 --validation_ratio 0.15
python ../impute.py --input_file train_test_split/mushroom_train.tsv \
                    --output_file imputations/mushroom_imputed.tsv \
                    --one_hot_max_sizes 6 4 10 2 9 2 2 2 12 2 4 4 4 9 9 4 3 5 9 6 7 2 \
                    --num_imputations 10 --epochs 50 --validation_ratio 0.15
python ../impute.py --input_file train_test_split/white_train.tsv \
                    --output_file imputations/white_imputed.tsv \
                    --one_hot_max_sizes 1 1 1 1 1 1 1 1 1 1 1 1 \
                    --num_imputations 10 --epochs 500 --validation_ratio 0.15
python evaluate_results.py yeast 1 1 1 1 1 1 1 1 10
python evaluate_results.py mushroom 6 4 10 2 9 2 2 2 12 2 4 4 4 9 9 4 3 5 9 6 7 2
python evaluate_results.py white 1 1 1 1 1 1 1 1 1 1 1 1
cd ..
```

## Inpainting

Unlike missing features imputation, image inpainting usually use
a dataset with no missing features and an unobserved region mask generator
to learn to inpaint.

In this repository there is all necessary code to reproduce CelebA
inpaintings from the paper.
It includes CelebA dataset wrapper, all mask generators from the paper,
and a model architecture.
The code is written in such way, so you'll find it easy to use
it with new datasets, mask generators, model architectures,
reconstruction losses, optimizers, etc.

Image inpainting process is splitted into several stages:
1. Firstly one define a model together with its optimizer, loss and
mask generator in `model.py` file in a separate directory.
Such model for the paper is provided in `celeba_model` directory.
2. Secondly, one implement image datasets (train, validation and test images
together with test masks), and add them into `datasets.py`.
One can use CelebA dataset which is already implemented (but not downloaded!)
and skip this step.
3. Then one train the model using
```
python train.py --model_dir celeba_model --epochs 40 \
                --train_dataset celeba_train --validation_dataset celeba_val
```
See `python train.py --help` for more options.

As a result two files are created in `celeba_model` directory:
`last_checkpoint.tar` and `best_checkpoint.tar`.
Second one is the best checkpoint according to IWAE on the validation set.
It is used for inpainting by deafult.

If these files are already in `model_dir` when `train.py` is started,
`train.py` use `last_checkpoint.tar` as an initial state for training.

One can also download pretrained model
from [here](https://yadi.sk/d/l4cRWuuHIaZQJQ),
put it into `celeba_model` directory and skip this step.

4. After that, one can inpaint the test set by calling
```
python inpaint.py --model_dir celeba_model --num_samples 3 \
                  --masks celeba_inpainting_masks --dataset celeba_test \
                  --out_dir celeba_inpaintings
```
See `python inpaint.py --help` for more options.

## Citation

If you find this code useful in your research,
please consider citing the paper:
```
@inproceedings{
    ivanov2018variational,
    title={Variational Autoencoder with Arbitrary Conditioning},
    author={Oleg Ivanov and Michael Figurnov and Dmitry Vetrov},
    booktitle={International Conference on Learning Representations},
    year={2019},
    url={https://openreview.net/forum?id=SyxtJh0qYm},
}
```
