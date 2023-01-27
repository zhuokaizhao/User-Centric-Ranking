import os
import argparse
import torch
import numpy as np
from deepctr_torch.inputs import (DenseFeat, SparseFeat, VarLenSparseFeat,
                                  get_feature_names)
from deepctr_torch.models.din import DIN

# for some tf warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# process features into format for DIN
def process_features_din(feature_path, feature_type):

    # loaded features keys can be found in process_data.py
    loaded_features = np.load(feature_path, allow_pickle=True)

    # list to indicate sequence sparse field
    behavior_feature_list = ['movie_id']
    # users
    user_id = loaded_features['user_id']
    gender = loaded_features['gender']
    age = loaded_features['age']
    occupation = loaded_features['occupation']
    zip_code = loaded_features['zip_code']
    if len(user_id) != len(gender) != len(age) != len(occupation) != len(zip_code):
        raise Exception("User data length not matched")

    # movies
    movie_id = loaded_features['movie_id']  # 0 is mask value
    title = loaded_features['title']
    genre = loaded_features['genre']
    if len(movie_id) != len(title) != len(genre):
        raise Exception("Movie data length not matched")

    # ic/uc features
    if feature_type == 'IC':
        behavior_feature = loaded_features['ic_feature']
        behavior_length = loaded_features['ic_feature_length']
    elif feature_type == 'UC':
        behavior_feature = loaded_features['uc_feature']
        behavior_length = loaded_features['uc_feature_length']
    else:
        raise Exception(f'Unrecognized feature type {feature_type}')
    if len(behavior_feature) != len(behavior_length):
        raise Exception("History data length not matched")


    # DNN feature columns for the deep part of DIN
    feature_columns = [SparseFeat('user_id', len(user_id), embedding_dim=8),
                       SparseFeat('gender', 2, embedding_dim=8),
                       SparseFeat('age', 7, embedding_dim=8),
                       SparseFeat('occupation', 21, embedding_dim=8),
                       SparseFeat('zip_code', len(set(zip_code)), embedding_dim=8),
                       SparseFeat('movie_id', len(movie_id) + 1, embedding_dim=8),
                       SparseFeat('title', len(title), embedding_dim=8),
                       SparseFeat('genre', len(set(genre)), embedding_dim=8),]
    # ic/uc feature
    feature_columns += [
                            VarLenSparseFeat(
                                SparseFeat(
                                    f'{feature_type}_feature',
                                    len(behavior_feature)+1,
                                    embedding_dim=8
                                ),
                                len(behavior_feature)+1,
                                length_name="seq_length"
                            ),
                        ]

    # feature dictrionary
    feature_dict = {'user_id': user_id,
                    'gender': gender,
                    'age': age,
                    'occupation': occupation,
                    'zip_code': zip_code,
                    'movie_id': movie_id,
                    'title': title,
                    'genre': genre,
                    f'{feature_type}_feature': behavior_feature,
                    'seq_length': behavior_length}

    x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
    y = np.array([1, 0, 1])

    return x, y, feature_columns, behavior_feature_list


if __name__ == "__main__":
    # input arguments
    parser = argparse.ArgumentParser()
    # mode as either train or test
    parser.add_argument('--mode', action='store', nargs=1, dest='mode', required=True)
    # IC or UC
    parser.add_argument(
        '--feature_type', action='store', nargs=1, dest='feature_type', required=True
    )
    # processed features path (.npz)
    parser.add_argument(
        '--feature_path', action='store', nargs=1, dest='feature_path', required=True
    )
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False)
    args = parser.parse_args()

    mode = args.mode[0]
    feature_type = args.feature_type[0]
    feature_path = args.feature_path[0]
    verbose = args.verbose

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    # load features
    x, y, feature_columns, behavior_feature_list = process_features_din(feature_path, feature_type)

    # dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    # history_feature_list: list to indicate sequence sparse field
    model = DIN(
        dnn_feature_columns=feature_columns,
        history_feature_list=behavior_feature_list,
        device=device,
        att_weight_normalization=True
    )
    model.compile(
        optimizer='adagrad',
        loss='binary_crossentropy',
        metrics=['accuracy'],
        # metrics=['binary_crossentropy'],
    )
    # verbose 1: progress bar, verbose 2: one line per epoch
    history = model.fit(
        x, y, batch_size=3, epochs=10, verbose=1, validation_split=0.2, shuffle=True
    )
