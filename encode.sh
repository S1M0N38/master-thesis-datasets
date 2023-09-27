#!/bin/bash
#
# Script to encode a dataset using various encoders
#
# Usage:
#   encode_dataset.sh <DATASET>
#
#   <DATASET> - The name or path of the dataset to be encoded.
#
# This script encodes the specified dataset using several different encoders
# with various configuration options.

DATASET=$1

# onehot
python encoders/onehot.py --dataset $DATASET

# barz-denzler
python encoders/barz-denzler.py --dataset $DATASET

# b3p
python encoders/b3p.py --dataset $DATASET --beta 0.1
python encoders/b3p.py --dataset $DATASET --beta 0.2
python encoders/b3p.py --dataset $DATASET --beta 0.3
python encoders/b3p.py --dataset $DATASET --beta 0.4
python encoders/b3p.py --dataset $DATASET --beta 0.5
python encoders/b3p.py --dataset $DATASET --beta 0.6
python encoders/b3p.py --dataset $DATASET --beta 0.7
python encoders/b3p.py --dataset $DATASET --beta 0.8
python encoders/b3p.py --dataset $DATASET --beta 0.9

# mbm
python encoders/mbm.py --dataset $DATASET --beta 1
python encoders/mbm.py --dataset $DATASET --beta 2
python encoders/mbm.py --dataset $DATASET --beta 3
python encoders/mbm.py --dataset $DATASET --beta 4
python encoders/mbm.py --dataset $DATASET --beta 5
python encoders/mbm.py --dataset $DATASET --beta 10
python encoders/mbm.py --dataset $DATASET --beta 15
python encoders/mbm.py --dataset $DATASET --beta 20
python encoders/mbm.py --dataset $DATASET --beta 30

# desc-pca
python encoders/desc-pca.py --dataset $DATASET --writer austen --embedder ada --n_components 100 --random_state 42
python encoders/desc-pca.py --dataset $DATASET --writer austen --embedder ada --n_components 200 --random_state 42
python encoders/desc-pca.py --dataset $DATASET --writer austen --embedder ada --n_components 300 --random_state 42
python encoders/desc-pca.py --dataset $DATASET --writer austen --embedder ada --n_components 400 --random_state 42
python encoders/desc-pca.py --dataset $DATASET --writer austen --embedder ada --n_components 500 --random_state 42
