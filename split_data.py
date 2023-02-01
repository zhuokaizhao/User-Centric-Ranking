# The file splits the dataset into users and movies and saves the corresponding lists
# get the unique user number from all the data
import os
import pandas as pd
import random
import numpy as np

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



feature_dir = '/home/zhuokai/Desktop/Meta/UCR/data/features_10m'
data_type = '10M'
num_files = len(os.listdir(feature_dir)) // 2
print(f'\n{num_files} data files loaded')
file_indices = [i for i in range(num_files)]
random.shuffle(file_indices)
all_sparse_feature_paths, all_hist_feature_paths = [], []
for i in file_indices:
    all_sparse_feature_paths.append(
        os.path.join(feature_dir, f'movie_lens_{data_type}_sparse_features_{i}.csv')
    )
    all_hist_feature_paths.append(
        os.path.join(feature_dir, f'movie_lens_{data_type}_ic_uc_features_{i}.npz')
    )
all_unique_users, all_unique_movies, all_num_ratings = get_stats(all_sparse_feature_paths)
print(f'Total: {len(all_unique_users)} users, {len(all_unique_movies)} movies and {all_num_ratings} ratings')

# split train and val based on users
num_train_users = int(len(all_unique_users) * 0.8)
train_users_list = np.array(all_unique_users[:num_train_users])
val_users_list = np.array(all_unique_users[num_train_users:])
print(f'Splitted into')
print(f'Training: {len(train_users_list)} users')
print(f'Validation: {len(val_users_list)} users')

# save users list
train_path = os.path.join(feature_dir, 'train_users_list.npy')
val_path = os.path.join(feature_dir, 'val_users_list.npy')
np.save(train_path, train_users_list)
np.save(val_path, val_users_list)
