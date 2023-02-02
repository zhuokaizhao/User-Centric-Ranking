# for some tf warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import torch.utils.data as Data
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from deepctr_torch.inputs import (DenseFeat, SparseFeat, VarLenSparseFeat,
#                                   get_feature_names)
# from deepctr_torch.models.din import DIN
from sklearn.metrics import roc_auc_score
from inputs import (DenseFeat, SparseFeat, VarLenSparseFeat,
                                  get_feature_names)
from din import DIN

random.seed(10)
np.random.seed(10)



# process features into format for DIN
def process_features_din(
    data_type,
    sparse_feature_path,
    hist_feature_path,
    hist_feature_type,
    verbose=False,
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
    if data_type == '1M':
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
    if data_type == '1M':
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
    else:
        feature_columns = [
            SparseFeat('positive_user_id', len(user_id), embedding_dim=32),
            SparseFeat('negative_user_id', len(user_id), embedding_dim=32),
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
    if data_type == '1M':
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
    else:
        feature_dict = {
            'positive_user_id': user_id,
            'negative_user_id': user_id,
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

    # get all the data with associated users
    data_input = {
        name: feature_dict[name] for name in get_feature_names(feature_columns)
    }
    data_label = labels


    return data_input, data_label, feature_columns, behavior_feature_list



if __name__ == "__main__":
    # input arguments
    parser = argparse.ArgumentParser()
    # mode as either train or test
    parser.add_argument(
        '--mode', action='store', nargs=1, dest='mode', required=True
    )
    # sum or attenntion
    parser.add_argument(
        '--model_type', action='store', nargs=1, dest='model_type', required=True
    )
    # 1M, 10M, 20M or 25M
    parser.add_argument(
        '--data_type', action='store', nargs=1, dest='data_type', required=True
    )
    # IC or UC
    parser.add_argument(
        '--feature_type', action='store', nargs=1, dest='feature_type',required=True
    )
    # train data dir
    parser.add_argument(
        '--train_dir', action='store', nargs=1, dest='train_dir',
    )
    # val data dir
    parser.add_argument(
        '--val_dir', action='store', nargs=1, dest='val_dir',
    )
    # test data dir
    parser.add_argument(
        '--test_dir', action='store', nargs=1, dest='test_dir',
    )
    # output(train) model directory
    parser.add_argument(
        '--output_model_dir', action='store', nargs=1, dest='output_model_dir'
    )
    # input model path for continue training or testing
    parser.add_argument(
        '--input_model_path', action='store', nargs=1, dest='input_model_path'
    )
    # output(train) history directory
    parser.add_argument(
        '--output_hist_dir', action='store', nargs=1, dest='output_hist_dir', required=True
    )
    parser.add_argument(
        '--num_epoch', action='store', nargs=1, dest='num_epoch'
    )
    parser.add_argument(
        '--batch_size', action='store', nargs=1, dest='batch_size'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', dest='verbose', default=False
    )
    args = parser.parse_args()
    mode = args.mode[0]
    model_type = args.model_type[0]
    data_type = args.data_type[0]
    feature_type = args.feature_type[0]
    output_hist_dir = args.output_hist_dir[0]
    if mode == 'train':
        output_model_dir = args.output_model_dir[0]
        train_dir = args.train_dir[0]
        val_dir = args.val_dir[0]
        continue_training = False
        if args.input_model_path:
            input_model_path = args.input_model_path[0]
            continue_training = True
    if mode == 'test':
        input_model_path = args.input_model_path[0]
        test_dir= args.test_dir[0]
    if args.num_epoch:
        num_epoch = int(args.num_epoch[0])
    else:
        num_epoch = 10
    if args.batch_size:
        batch_size = int(args.batch_size[0])
    else:
        batch_size = 256
    verbose = args.verbose

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    # data for training DIN
    if mode == 'train':
        # load features
        # get the number of files in folder
        train_num_files = len(os.listdir(train_dir)) // 2
        val_num_files = len(os.listdir(val_dir)) // 2
        # file indices
        train_file_indices = [i for i in range(train_num_files)]
        val_file_indices = [i for i in range(val_num_files)]
        # shuffle both indices
        random.shuffle(train_file_indices)
        random.shuffle(val_file_indices)
        # train feature paths
        train_sparse_feature_paths, train_hist_feature_paths = [], []
        for i in train_file_indices:
            train_sparse_feature_paths.append(
                os.path.join(train_dir, f'movie_lens_{data_type}_sparse_features_train_{i}.csv')
            )
            train_hist_feature_paths.append(
                os.path.join(train_dir, f'movie_lens_{data_type}_IC_UC_features_train_{i}.npz')
            )
        # val feature paths
        val_sparse_feature_paths, val_hist_feature_paths = [], []
        for i in val_file_indices:
            val_sparse_feature_paths.append(
                os.path.join(val_dir, f'movie_lens_{data_type}_sparse_features_test_{i}.csv')
            )
            val_hist_feature_paths.append(
                os.path.join(val_dir, f'movie_lens_{data_type}_IC_UC_features_test_{i}.npz')
            )

        # use the first path to initialize the DIN model
        sparse_feature_path = train_sparse_feature_paths[0]
        hist_feature_path = train_hist_feature_paths[0]
        _, _, feature_columns, behavior_feature_list = process_features_din(
            data_type, sparse_feature_path, hist_feature_path, feature_type
        )
        model = DIN(
            dnn_feature_columns=feature_columns,
            history_feature_list=behavior_feature_list,
            pooling_type=model_type,
            device=device,
            att_weight_normalization=True
        )

        # define training attributes
        optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
        loss_function = F.binary_cross_entropy
        metric_function = roc_auc_score
        print(f'\nModel constructed successfully. Running on {device}')

        if continue_training:
            checkpoint = torch.load(input_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            trained_epoch = checkpoint['num_epoch']
            history = checkpoint['history']
        else:
            trained_epoch = 0
            # training loss and validation metric for each epoch
            history = defaultdict(list)

        # outer loop as epoch
        for e in range(trained_epoch, num_epoch):
            print(f'\nEpoch {e+1}/{num_epoch}')
            # for each epoch, re-shuffle data files ordering
            random.shuffle(train_file_indices)
            random.shuffle(val_file_indices)
            # train feature paths
            train_sparse_feature_paths, train_hist_feature_paths = [], []
            for i in train_file_indices:
                train_sparse_feature_paths.append(
                    os.path.join(train_dir, f'movie_lens_{data_type}_sparse_features_train_{i}.csv')
                )
                train_hist_feature_paths.append(
                    os.path.join(train_dir, f'movie_lens_{data_type}_IC_UC_features_train_{i}.npz')
                )
            # val feature paths
            val_sparse_feature_paths, val_hist_feature_paths = [], []
            for i in val_file_indices:
                val_sparse_feature_paths.append(
                    os.path.join(val_dir, f'movie_lens_{data_type}_sparse_features_test_{i}.csv')
                )
                val_hist_feature_paths.append(
                    os.path.join(val_dir, f'movie_lens_{data_type}_IC_UC_features_test_{i}.npz')
                )

            # each batch's loss in this epoch
            cur_epoch_train_losses = []
            cur_epoch_train_metrics = []

            # set model to train
            model.train(True)

            # start training on all files
            for n in range(train_num_files):
                # current data file path
                sparse_feature_path = train_sparse_feature_paths[n]
                hist_feature_path = train_hist_feature_paths[n]

                # process features
                train_input, train_label, _, _ = process_features_din(
                    data_type,
                    sparse_feature_path,
                    hist_feature_path,
                    feature_type
                )

                # process input format, double check on shapes
                if isinstance(train_input, dict):
                    train_input = [train_input[feature] for feature in model.feature_index]

                if len(train_input[0]) == 0:
                    continue

                for i in range(len(train_input)):
                    if len(train_input[i].shape) == 1:
                        train_input[i] = np.expand_dims(train_input[i], axis=1)

                # create training tensors
                train_data = Data.TensorDataset(
                                torch.from_numpy(np.concatenate(train_input, axis=-1)),
                                torch.from_numpy(train_label)
                            )
                # create dataloader
                train_loader = DataLoader(
                    dataset=train_data, shuffle=True, batch_size=batch_size
                )
                print(f'Batch {n+1}/{train_num_files}: file {train_file_indices[n]}, {len(train_data)} samples')
                for mini_batch_idx, (x_train, y_train) in tqdm(enumerate(train_loader), desc='Mini batch'):
                    # send data to training device
                    x = x_train.to(device).float()
                    y = y_train.to(device).float()
                    # train model prediction
                    y_pred = model(x)
                    # zero grad before back prop
                    optimizer.zero_grad()
                    # compute loss and save it
                    train_batch_loss = loss_function(
                        y_pred.squeeze(), y.squeeze(), reduction='sum'
                    )
                    cur_epoch_train_losses.append(train_batch_loss.item())
                    train_batch_metric = metric_function(
                        y.cpu().data.numpy(), y_pred.cpu().data.numpy()
                    )
                    cur_epoch_train_metrics.append(train_batch_metric)
                    # backprop and update optimizer
                    train_batch_loss.backward()
                    optimizer.step()

            # after training on all the files, compute average loss
            epoch_avg_train_loss = sum(cur_epoch_train_losses) / len(cur_epoch_train_losses)
            epoch_avg_train_metric = sum(cur_epoch_train_metrics) / len(cur_epoch_train_metrics)
            history['all_training_losses'].append(epoch_avg_train_loss)
            history['all_training_metrics'].append(epoch_avg_train_metric)

            # run validation
            print('Running Validation')
            cur_epoch_val_losses = []
            cur_epoch_val_metrics = []
            model.train(False)
            for n in range(val_num_files):
                # current data file path
                sparse_feature_path = val_sparse_feature_paths[n]
                hist_feature_path = val_hist_feature_paths[n]

                val_input, val_label, _, _ = process_features_din(
                    data_type,
                    sparse_feature_path,
                    hist_feature_path,
                    feature_type
                )

                # process input format, double check on shapes
                if isinstance(val_input, dict):
                    val_input = [val_input[feature] for feature in model.feature_index]

                if len(val_input[0]) == 0:
                    continue

                for i in range(len(val_input)):
                    if len(val_input[i].shape) == 1:
                        val_input[i] = np.expand_dims(val_input[i], axis=1)

                # create validation tensors
                val_data = Data.TensorDataset(
                                torch.from_numpy(np.concatenate(val_input, axis=-1)),
                                torch.from_numpy(val_label)
                            )

                # create dataloader
                val_loader = DataLoader(
                    dataset=val_data, shuffle=True, batch_size=batch_size
                )
                print(f'Batch {n+1}/{val_num_files}: file {val_file_indices[n]}, {len(val_data)} samples')

                with torch.no_grad():
                    for mini_batch_idx, (x_val, y_val) in tqdm(enumerate(val_loader), desc='Mini batch'):
                        # send data to training device
                        x = x_val.to(device).float()
                        y = y_val.to(device).float()
                        y_pred = model(x)
                        val_batch_loss = loss_function(
                            y_pred.squeeze(), y.squeeze(), reduction='sum'
                        )
                        cur_epoch_val_losses.append(val_batch_loss.item())
                        val_batch_metric = metric_function(
                            y.cpu().data.numpy(), y_pred.cpu().data.numpy()
                        )
                        cur_epoch_val_metrics.append(val_batch_metric)

            # after training on all the files, compute average loss
            epoch_avg_val_loss = sum(cur_epoch_val_losses) / len(cur_epoch_val_losses)
            epoch_avg_val_metric = sum(cur_epoch_val_metrics) / len(cur_epoch_val_metrics)
            history['all_val_losses'].append(epoch_avg_val_loss)
            history['all_val_metrics'].append(epoch_avg_val_metric)
            print(f'Epoch {e+1}/{num_epoch} Completed.')
            print(f'Avg Train Loss: {epoch_avg_train_loss}, Avg Train AUC: {epoch_avg_train_metric}')
            print(f'Avg Val Loss: {epoch_avg_val_loss}, Avg Val AUC: {epoch_avg_val_metric}')

            # save trained model every 5 epoch
            if (n+1) % 5 == 0:
                model_path = os.path.join(
                    output_model_dir,
                    f'DIN_{model_type}_{feature_type}_{data_type}_{num_epoch}_{batch_size}.pt'
                )
                if torch.cuda.device_count() > 1:
                    model_checkpoint = {
                        'num_epoch': n+1,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'history': history
                    }
                else:
                    model_checkpoint = {
                        'num_epoch': n+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'history': history
                    }

                torch.save(model_checkpoint, model_path)
                print(f'\nTrained model checkpoint has been saved to {model_path}\n')

        # save the history by pandas
        history_df = pd.DataFrame(history)
        hist_csv_path = os.path.join(
            output_hist_dir, 'hist_{model_type}_{feature_type}_{data_type}_{num_epoch}_{batch_size}.csv'
        )
        history_df.to_csv(hist_csv_path)
        print(f'\nAssociated model history has been saved to {hist_csv_path}\n')

    elif mode == 'test':
        # load features
        # get the number of files in folder
        test_num_files = len(os.listdir(test_dir)) // 2
        # file indices
        test_file_indices = [i for i in range(test_num_files)]
        # shuffle both indices
        random.shuffle(test_file_indices)
        # train feature paths
        test_sparse_feature_paths, test_hist_feature_paths = [], []
        for i in test_file_indices:
            test_sparse_feature_paths.append(
                os.path.join(train_dir, f'movie_lens_{data_type}_sparse_features_test_{i}.csv')
            )
            test_hist_feature_paths.append(
                os.path.join(train_dir, f'movie_lens_{data_type}_IC_UC_features_test_{i}.npz')
            )

        # use the first path to initialize the DIN model
        sparse_feature_path = test_sparse_feature_paths[0]
        hist_feature_path = test_hist_feature_paths[0]
        _, _, feature_columns, behavior_feature_list = process_features_din(
            data_type, sparse_feature_path, hist_feature_path, feature_type
        )
        model = DIN(
            dnn_feature_columns=feature_columns,
            history_feature_list=behavior_feature_list,
            pooling_type=model_type,
            device=device,
            att_weight_normalization=True
        )

        # define training attributes
        loss_function = F.binary_cross_entropy
        metric_function = roc_auc_score
        # load trained model
        checkpoint = torch.load(input_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'\nModel constructed successfully. Running on {device}')

        test_losses = []
        test_metrics = []
        model.train(False)
        for n in range(test_num_files):
            # current data file path
            sparse_feature_path = test_sparse_feature_paths[n]
            hist_feature_path = test_hist_feature_paths[n]

            test_input, test_label, _, _ = process_features_din(
                data_type,
                sparse_feature_path,
                hist_feature_path,
                feature_type
            )

            # process input format, double check on shapes
            if isinstance(test_input, dict):
                test_input = [test_input[feature] for feature in model.feature_index]

            if len(test_input[0]) == 0:
                continue

            for i in range(len(test_input)):
                if len(test_input[i].shape) == 1:
                    test_input[i] = np.expand_dims(test_input[i], axis=1)

            # create validation tensors
            test_data = Data.TensorDataset(
                            torch.from_numpy(np.concatenate(test_input, axis=-1)),
                            torch.from_numpy(test_label)
                        )

            # create dataloader
            test_loader = DataLoader(
                dataset=test_data, shuffle=True, batch_size=batch_size
            )
            print(f'Batch {n+1}/{test_num_files}: file {test_file_indices[n]}, {len(test_data)} samples')

            with torch.no_grad():
                for mini_batch_idx, (x_test, y_test) in tqdm(enumerate(test_loader), desc='Mini batch'):
                    # send data to training device
                    x = x_test.to(device).float()
                    y = y_test.to(device).float()
                    y_pred = model(x)
                    test_batch_loss = loss_function(
                        y_pred.squeeze(), y.squeeze(), reduction='sum'
                    )
                    test_losses.append(test_batch_loss.item())
                    test_batch_metric = metric_function(
                        y.cpu().data.numpy(), y_pred.cpu().data.numpy()
                    )
                    test_metrics.append(test_batch_metric)

        # after training on all the files, compute average loss
        avg_test_loss = sum(test_losses) / len(test_losses)
        avg_test_metric = sum(test_metrics) / len(test_metrics)
        print(f'Avg Test Loss: {avg_test_loss}, Avg Test AUC: {avg_test_metric}')

    else:
        raise Exception(f"Unrecognized mode {mode}")


