# The file splits the dataset into users and movies and saves the corresponding lists
# get the unique user number from all the data
import os
import pandas as pd
import random
import numpy as np
import argparse

def get_stats(sparse_feature_path):
    num_ratings = 0
    unique_users = []
    unique_movies = []
    for cur_path in sparse_feature_path:
        cur_sparse_features = pd.read_csv(
            cur_path,
            engine='python',
            header='infer',
        )
        unique_users += list(cur_sparse_features['user_id'].to_numpy())
        unique_movies += list(cur_sparse_features['movie_id'].to_numpy())
        num_ratings += len(cur_sparse_features)

    unique_users = list(set(unique_users))
    unique_movies = list(set(unique_movies))

    return unique_users, unique_movies, num_ratings


# process features into format for DIN
def split_data(
    output_dir,
    train_user_ids_list,
    test_user_ids_list,
    data_type,
    all_sparse_feature_paths,
    all_hist_feature_paths,
):

    train_index = 0
    test_index = 0
    for i in range(len(all_sparse_feature_paths)):
        sparse_feature_path = all_sparse_feature_paths[i]
        hist_feature_path = all_hist_feature_paths[i]

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
        movie_name = sparse_features['movie_name'].to_numpy()
        genre = sparse_features['genre'].to_numpy()

        # ic/uc features
        positive_ic_feature = hist_features['positive_ic_feature'].astype(int)
        positive_ic_feature_length = hist_features['positive_ic_feature_length'].astype(int)
        negative_ic_feature = hist_features['negative_ic_feature'].astype(int)
        negative_ic_feature_length = hist_features['negative_ic_feature_length'].astype(int)
        positive_uc_feature = hist_features['positive_uc_feature'].astype(int)
        positive_uc_feature_length = hist_features['positive_uc_feature_length'].astype(int)
        negative_uc_feature = hist_features['negative_uc_feature'].astype(int)
        negative_uc_feature_length = hist_features['negative_uc_feature_length'].astype(int)

        # Make sure that the sparse and IC/UC features should have the same length
        if len(sparse_features) != len(positive_ic_feature):
            raise Exception(
                f"Sparse ({len(sparse_features)}) and IC/UC ({len(positive_ic_feature)}) features should have the same length"
            )

        # labels
        labels = sparse_features['labels'].to_numpy()

        # train and test split based on loaded user_id (they should all be related)
        temp_train_indices = [np.where(user_id == cur_id) for cur_id in train_user_ids_list]
        train_indices = []
        for cur_set in temp_train_indices:
            for cur_id in cur_set[0]:
                train_indices.append(cur_id)

        # if trian is not empty
        if train_indices != []:
            # create dataframes for sparse features
            train_df = pd.DataFrame()
            train_df['user_id'] = user_id[train_indices]
            if data_type == '1M':
                train_df['gender'] = gender[train_indices]
                train_df['age'] = age[train_indices]
                train_df['occupation'] = occupation[train_indices]
            train_df['movie_id'] = movie_id[train_indices]
            train_df['score'] = score[train_indices]
            train_df['movie_name'] = movie_name[train_indices]
            train_df['genre'] = genre[train_indices]
            train_df['labels'] = labels[train_indices]

            # save splited features
            # sparse features
            train_df_path = os.path.join(
                output_dir, 'train', f'movie_lens_{data_type}_sparse_features_train_{train_index}.csv'
            )
            # create dict and save
            train_df.to_csv(train_df_path)
            print(f'Train sparse features has been saved to {train_df_path}')

            # IC/UC features
            train_ic_uc_path = os.path.join(
                output_dir, 'train', f'movie_lens_{data_type}_IC_UC_features_train_{train_index}.npz'
            )

            # IC and UC features
            # create dict and save
            train_arrays_to_save = {
                "positive_ic_feature": positive_ic_feature[train_index],
                "positive_ic_feature_length": positive_ic_feature_length[train_index],
                "negative_ic_feature": negative_ic_feature[train_index],
                "negative_ic_feature_length": negative_ic_feature_length[train_index],
                "positive_uc_feature": positive_uc_feature[train_index],
                "positive_uc_feature_length": positive_uc_feature_length[train_index],
                "negative_uc_feature": negative_uc_feature[train_index],
                "negative_uc_feature_length": negative_uc_feature_length[train_index],
            }
            np.savez(train_ic_uc_path, **train_arrays_to_save)
            print(f'IC/UC features has been saved to {train_ic_uc_path}')

            # update train file index number
            train_index += 1


        temp_test_indices = [np.where(user_id == cur_id) for cur_id in test_user_ids_list]
        test_indices = []
        for cur_set in temp_test_indices:
            for cur_id in cur_set[0]:
                test_indices.append(cur_id)

        # if test is not empty
        if test_indices != []:
            # create dataframes for sparse features
            test_df = pd.DataFrame()
            test_df['user_id'] = user_id[test_indices]
            if data_type == '1M':
                test_df['gender'] = gender[test_indices]
                test_df['age'] = age[test_indices]
                test_df['occupation'] = occupation[test_indices]
            test_df['movie_id'] = movie_id[test_indices]
            test_df['score'] = score[test_indices]
            test_df['movie_name'] = movie_name[test_indices]
            test_df['genre'] = genre[test_indices]
            test_df['labels'] = labels[test_indices]

            # save splited features
            # sparse features
            test_df_path = os.path.join(
                output_dir, 'test', f'movie_lens_{data_type}_sparse_features_test_{test_index}.csv'
            )
            # create dict and save
            test_df.to_csv(test_df_path)
            print(f'Test sparse features has been saved to {test_df_path}')

            # IC/UC features
            test_ic_uc_path = os.path.join(
                output_dir, 'test', f'movie_lens_{data_type}_IC_UC_features_test_{test_index}.npz'
            )

            # IC and UC features
            # create dict and save
            test_arrays_to_save = {
                "positive_ic_feature": positive_ic_feature[test_index],
                "positive_ic_feature_length": positive_ic_feature_length[test_index],
                "negative_ic_feature": negative_ic_feature[test_index],
                "negative_ic_feature_length": negative_ic_feature_length[test_index],
                "positive_uc_feature": positive_uc_feature[test_index],
                "positive_uc_feature_length": positive_uc_feature_length[test_index],
                "negative_uc_feature": negative_uc_feature[test_index],
                "negative_uc_feature_length": negative_uc_feature_length[test_index],
            }
            np.savez(test_ic_uc_path, **test_arrays_to_save)
            print(f'IC/UC features has been saved to {test_ic_uc_path}')

            # update train file index number
            test_index += 1



if __name__ == "__main__":
    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_type', action='store', nargs=1, dest='data_type', required=True
    )
    parser.add_argument(
        '--feature_dir', action='store', nargs=1, dest='feature_dir', required=True
    )
    parser.add_argument(
        '--output_dir', action='store', nargs=1, dest='output_dir', required=True
    )
    args = parser.parse_args()
    data_type = args.data_type[0]
    feature_dir = args.feature_dir[0]
    output_dir = args.output_dir[0]

    # load files
    num_files = len(os.listdir(feature_dir)) // 2
    print(f'\n{num_files} data files loaded')
    file_indices = [i for i in range(num_files)]
    all_sparse_feature_paths, all_hist_feature_paths = [], []
    for i in file_indices:
        all_sparse_feature_paths.append(
            os.path.join(feature_dir, f'movie_lens_{data_type}_sparse_features_{i}.csv')
        )
        all_hist_feature_paths.append(
            os.path.join(feature_dir, f'movie_lens_{data_type}_IC_UC_features_{i}.npz')
        )
    all_unique_users, all_unique_movies, all_num_ratings = get_stats(all_sparse_feature_paths)
    print(f'Total: {len(all_unique_users)} users, {len(all_unique_movies)} movies and {all_num_ratings} ratings')

    # split train and val/test based on users
    num_train_users = int(len(all_unique_users) * 0.8)
    train_users_list = np.random.choice(np.array(all_unique_users), num_train_users, replace=False)
    test_users_list = np.array(
        [user_id for user_id in all_unique_users if user_id not in train_users_list]
    )
    print(f'Splitted into')
    print(f'Training: {len(train_users_list)} users')
    print(f'Validation: {len(test_users_list)} users')

    # split and save
    split_data(
        output_dir,
        train_users_list,
        test_users_list,
        data_type,
        all_sparse_feature_paths,
        all_hist_feature_paths,
    )

    # save users list
    train_path = os.path.join(output_dir, 'train_users_list.npy')
    test_path = os.path.join(output_dir, 'val_users_list.npy')
    np.save(train_path, train_users_list)
    np.save(test_path, test_users_list)
