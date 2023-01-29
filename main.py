# for some tf warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse
import torch
import numpy as np
import pandas as pd
from deepctr_torch.inputs import (DenseFeat, SparseFeat, VarLenSparseFeat,
                                  get_feature_names)
from deepctr_torch.models.din import DIN
# from inputs import (DenseFeat, SparseFeat, VarLenSparseFeat,
#                                   get_feature_names)
# from din import DIN




# process features into format for DIN
def process_features_din(
    mode, sparse_feature_path, hist_feature_path, hist_feature_type, split=0.2
):

    # loaded features keys can be found in process_data.py
    sparse_features = pd.read_csv(
        sparse_feature_path,
        engine='python',
        header='infer',
    )
    # IC/UC features
    hist_features = np.load(hist_feature_path, allow_pickle=True)

    # users
    user_id = sparse_features['user_id'].to_numpy()
    gender = sparse_features['gender'].to_numpy()
    age = sparse_features['age'].to_numpy()
    occupation = sparse_features['occupation'].to_numpy()

    # movies
    movie_id = sparse_features['movie_id'].to_numpy()  # 0 is mask value
    score = sparse_features['rating'].to_numpy()
    # movie_name = sparse_features['movie_name'].to_numpy()
    # genre = sparse_features['genre'].to_numpy()

    # ic/uc features
    if hist_feature_type == 'IC':
        positive_behavior_feature = hist_features['positive_ic_feature'].astype(int)
        positive_behavior_length = hist_features['positive_ic_feature_length'].astype(int)
        negative_behavior_feature = hist_features['negative_ic_feature'].astype(int)
        negative_behavior_length = hist_features['negative_ic_feature_length'].astype(int)
    elif hist_feature_type == 'UC':
        positive_behavior_feature = hist_features['positive_uc_feature'].astype(int)
        positive_behavior_length = hist_features['positive_uc_feature_length'].astype(int)
        negative_behavior_feature = hist_features['negative_uc_feature'].astype(int)
        negative_behavior_length = hist_features['negative_uc_feature_length'].astype(int)
    else:
        raise Exception(f'Unrecognized feature type {hist_feature_type}')

    if len(positive_behavior_feature) != len(positive_behavior_feature):
        raise Exception("History data length not matched")

    # Make sure that the sparse and IC/UC features should have the same length
    if len(sparse_features) != len(positive_behavior_feature):
        raise Exception(
            f"Sparse ({len(sparse_features)}) and IC/UC ({len(positive_behavior_feature)}) features should have the same length"
        )

    # labels
    labels = sparse_features['labels'].to_numpy()

    # DNN feature columns for the deep part of DIN
    # duplicate user_id and movie_id for both positive and negative
    feature_columns = [
        SparseFeat('positive_user_id', len(user_id), embedding_dim=32),
        SparseFeat('negative_user_id', len(user_id), embedding_dim=32),
        SparseFeat('gender', 2, embedding_dim=8),
        SparseFeat('age', 57, embedding_dim=8),
        SparseFeat('occupation', 21, embedding_dim=8),
        SparseFeat('positive_movie_id', len(movie_id)+1, embedding_dim=32), # 0 is mask value
        SparseFeat('negative_movie_id', len(movie_id)+1, embedding_dim=32), # 0 is mask value
        DenseFeat('score', 1),
        # SparseFeat('movie_name', len(set(movie_name)), embedding_dim=8),
        # SparseFeat('genre', len(set(genre)), embedding_dim=8),
    ]
    # ic/uc feature
    # list to indicate sequence sparse field
    if feature_type == 'IC':
        behavior_feature_list = [
            'positive_movie_id',
            'negative_movie_id',
        ]
    elif feature_type == 'UC':
        behavior_feature_list = [
            'positive_user_id',
            'negative_user_id'
        ]
    feature_columns += [
                            VarLenSparseFeat(
                                SparseFeat(
                                    f'hist_{behavior_feature_list[0]}',
                                    len(positive_behavior_feature) + 1,
                                    embedding_dim=32
                                ),
                                maxlen=max(positive_behavior_length),
                                length_name='positive_seq_length',
                            ),
                            VarLenSparseFeat(
                                SparseFeat(
                                    f'hist_{behavior_feature_list[1]}',
                                    len(negative_behavior_feature) + 1,
                                    embedding_dim=32
                                ),
                                maxlen=max(negative_behavior_length),
                                length_name='negative_seq_length'
                            ),
                        ]

    # feature dictrionary
    feature_dict = {
        'positive_user_id': user_id,
        'negative_user_id': user_id,
        'gender': gender,
        'age': age,
        'occupation': occupation,
        'positive_movie_id': movie_id,
        'negative_movie_id': movie_id,
        'score': score,
        # 'movie_name': movie_name,
        # 'genre': genre,
        f'hist_{behavior_feature_list[0]}': positive_behavior_feature,
        'positive_seq_length': positive_behavior_length,
        f'hist_{behavior_feature_list[1]}': negative_behavior_feature,
        'negative_seq_length': negative_behavior_length,
    }

    if verbose:
        print('Feature dict includes:')
        for name in get_feature_names(feature_columns):
            print(name, feature_dict[name].dtype)

    # train/val or test split
    if mode == 'train':
        num_train_samples = int(len(sparse_features) * (1-split))
        train_input = {
            name: feature_dict[name][:num_train_samples]
                    for name in get_feature_names(feature_columns)
        }
        train_label = labels[:num_train_samples]
        val_input = {
            name: feature_dict[name][num_train_samples:]
                    for name in get_feature_names(feature_columns)
        }
        val_label = labels[num_train_samples:]

        return train_input, train_label, val_input, val_label, feature_columns, behavior_feature_list

    elif mode == 'test':
        num_train_samples = int(len(sparse_features) * (1-split))
        test_input = {
            name: feature_dict[name][num_train_samples:]
                    for name in get_feature_names(feature_columns)
        }
        test_label = labels[num_train_samples:]

        return test_input, test_label, feature_columns, behavior_feature_list


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
        '--feature_dir', action='store', nargs=1, dest='feature_dir', required=True
    )
    parser.add_argument(
        '--num_epoch', action='store', nargs=1, dest='num_epoch'
    )
    parser.add_argument(
        '--batch_size', action='store', nargs=1, dest='batch_size'
    )
    parser.add_argument('--model_dir', action='store', nargs=1, dest='model_dir', required=True)
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False)
    args = parser.parse_args()

    mode = args.mode[0]
    feature_type = args.feature_type[0]
    feature_dir = args.feature_dir[0]
    model_dir = args.model_dir[0]
    if args.num_epoch:
        num_epoch = int(args.num_epoch[0])
    else:
        num_epoch =10
    if args.batch_size:
        batch_size = int(args.batch_size[0])
    else:
        batch_size =256
    verbose = args.verbose

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    # load features
    sparse_feature_path = os.path.join(feature_dir, 'movie_lens_1M_sparse_features.csv')
    hist_feature_path = os.path.join(feature_dir, 'movie_lens_1M_IC_UC_features.npz')

    # data for training DIN
    if mode == 'train':
        train_input, \
        train_label, \
        val_input, \
        val_label, \
        feature_columns, \
        behavior_feature_list = process_features_din(
            mode, sparse_feature_path, hist_feature_path, feature_type
        )

        # model
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
            train_input,
            train_label,
            batch_size=batch_size,
            epochs=num_epoch,
            verbose=1,
            validation_split=0.0,
            validation_data=(val_input, val_label),
            shuffle=True,
        )

        # save trained model
        model_path = os.path.join(model_dir, f'din_{num_epoch}_{batch_size}.pt')
        if torch.cuda.device_count() > 1:
            model_checkpoint = {
                                    'epoch': num_epoch,
                                    'state_dict': model.module.state_dict(),
                                    'history': history,
                                }
        else:
            model_checkpoint = {
                                    'epoch': num_epoch,
                                    'state_dict': model.state_dict(),
                                    'history': history,
                                }

        torch.save(model_checkpoint, model_path)
        print(f'\nTrained model checkpoint has been saved to {model_path}\n')

    elif mode == 'test':
        test_input, \
        test_label, \
        feature_columns, \
        behavior_feature_list = process_features_din(
            mode, sparse_feature_path, hist_feature_path, feature_type
        )
        # model
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
        # run prediction
        pred_ans = model.predict(
            test_input,
            batch_size=batch_size
        )

    else:
        raise Exception(f"Unrecognized mode {mode}")


