import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from deepctr_torch.inputs import (DenseFeat, SparseFeat, VarLenSparseFeat,
                                  get_feature_names)
from deepctr_torch.models.din import DIN

# for some tf warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def get_xy_fd():
    feature_columns = [SparseFeat('user', 3, embedding_dim=8),
                       SparseFeat('gender', 2, embedding_dim=8),
                       SparseFeat('item', 3 + 1, embedding_dim=8),
                       SparseFeat('item_gender', 2 + 1, embedding_dim=8),
                       DenseFeat('score', 1)]

    feature_columns += [
                            VarLenSparseFeat(
                                SparseFeat('hist_item', 3 + 1, embedding_dim=8),
                                4,
                                length_name="seq_length"
                            ),
                            VarLenSparseFeat(
                                SparseFeat('hist_item_gender', 2 + 1, embedding_dim=8),
                                4,
                                length_name="seq_length"
                            )
                        ]

    behavior_feature_list = ["item", "item_gender"]
    # user id
    uid = np.array([0, 1, 2])
    ugender = np.array([0, 1, 0])
    iid = np.array([1, 2, 3])  # 0 is mask value
    igender = np.array([1, 2, 1])  # 0 is mask value
    score = np.array([0.1, 0.2, 0.3])

    hist_iid = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0]])
    hist_igender = np.array([[1, 1, 2, 0], [2, 1, 1, 0], [2, 1, 0, 0]])
    behavior_length = np.array([3, 3, 2])

    feature_dict = {'user': uid,
                    'gender': ugender,
                    'item': iid,
                    'item_gender': igender,
                    'hist_item': hist_iid,
                    'hist_item_gender': hist_igender,
                    'score': score,
                    "seq_length": behavior_length}

    x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
    y = np.array([1, 0, 1])

    return x, y, feature_columns, behavior_feature_list


if __name__ == "__main__":
    # input arguments
    parser = argparse.ArgumentParser()
    # mode as either train or test
    parser.add_argument('--mode', action='store', nargs=1, dest='mode', required=True)
    # processed features path (.npz)
    parser.add_argument('--data_path', action='store', nargs=1, dest='data_path', required=True)
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False)
    args = parser.parse_args()

    mode = args.mode[0]
    data_path = args.data_path[0]
    verbose = args.verbose

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    # load features


    # dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    # history_feature_list: list to indicate sequence sparse field
    model = DIN(dnn_feature_columns=feature_columns,
                history_feature_list=behavior_feature_list,
                device=device,
                att_weight_normalization=True)
    model.compile('adagrad', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
