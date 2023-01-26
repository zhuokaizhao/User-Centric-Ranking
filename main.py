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


# load and process MovieLens data
def load_data(data_dir, real_occupation=False):
    # movies
    movies_path = os.path.join(data_dir, "movies.dat")
    movies_df = pd.read_csv(movies_path,
                            encoding='iso-8859-1',
                            delimiter='::',
                            engine='python',
                            header=None,
                            names=['movie_name', 'genre'])

    # users
    users_path = os.path.join(data_dir, 'users.dat')
    users_df = pd.read_csv(users_path,
                            delimiter='::',
                            engine='python',
                            header=None,
                            names=['user_id', 'gender', 'age', 'occupation', 'zip_code'])
    # use README to swap numbers to actual occupation
    if real_occupation:
        # load readme
        readme_path = os.path.join(data_dir, 'README')
        readme_text = np.array(open(readme_path).read().splitlines())
        start = np.flatnonzero(
                            np.core.defchararray.find(readme_text,'Occupation is chosen') != -1
                      )[0]
        end = np.flatnonzero(
                            np.core.defchararray.find(readme_text,'MOVIES FILE DESCRIPTION')!=-1
                    )[0]
        occupation_list = [x.split('"')[1] for x in readme_text[start:end][2:-1].tolist()]
        occupation_dict = dict(zip(range(len(occupation_list)), occupation_list))

        # replace the info
        users_df['occupation'] = users_df['occupation'].replace(occupation_dict)

    # ratings
    ratings_path = os.path.join(data_dir, 'ratings.dat')
    ratings_df = pd.read_csv(ratings_path,
                            delimiter='::',
                            engine='python',
                            header=None,
                            names=['user_id', 'movie_id', 'rating', 'time'])




    return movies_df, users_df, ratings_df



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
    parser.add_argument('--mode', action='store', nargs=1, dest='mode', required=True)
    parser.add_argument('--data_dir', action='store', nargs=1, dest='data_dir', required=True)
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False)
    args = parser.parse_args()

    mode = args.mode[0]
    data_dir = args.data_dir[0]
    verbose = args.verbose

    if verbose:
        print(f'\nMode: {mode}')
        print(f'Data dir: {data_dir}\n')

    # load data
    movies_df, users_df, ratings_df = load_data(data_dir, real_occupation=False)
    print(movies_df.head())
    print(users_df.head())
    print(ratings_df.head())

    # x, y, feature_columns, behavior_feature_list = get_xy_fd()
    # device = 'cpu'
    # use_cuda = True
    # if use_cuda and torch.cuda.is_available():
    #     print('cuda ready...')
    #     device = 'cuda:0'

    # # dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    # # history_feature_list: list to indicate sequence sparse field
    # model = DIN(dnn_feature_columns=feature_columns,
    #             history_feature_list=behavior_feature_list,
    #             device=device,
    #             att_weight_normalization=True)
    # model.compile('adagrad', 'binary_crossentropy',
    #               metrics=['binary_crossentropy'])