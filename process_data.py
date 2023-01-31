# The script loads MovieLens data and generate IC, UC features
import os
from tqdm import tqdm
from enum import Enum
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Process, set_start_method, cpu_count
# set_start_method('forkserver')



# load and process MovieLens data
def load_data(data_dir, data_type, real_occupation=False):

    # for movie lens 1M
    if data_type == '1M':
        # movies
        movies_path = os.path.join(data_dir, f'movies.dat')
        movies_df = pd.read_csv(
            movies_path,
            encoding='iso-8859-1',
            delimiter='::',
            engine='python',
            header=None,
            names=['movie_name', 'genre']
        )

        # users
        users_path = os.path.join(data_dir, f'users.dat')
        users_df = pd.read_csv(
            users_path,
            delimiter='::',
            engine='python',
            header=None,
            names=['user_id', 'gender', 'age', 'occupation', 'zip_code']
        )
        # use README to swap numbers to actual occupation for analysis purpose
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
        ratings_path = os.path.join(data_dir, f'ratings.dat')
        ratings_df = pd.read_csv(
            ratings_path,
            delimiter='::',
            engine='python',
            header=None,
            names=['user_id', 'movie_id', 'rating', 'time']
        )

        return movies_df, users_df, ratings_df

    # for movie lens 10M
    if data_type == '10M':
        # movies
        movies_path = os.path.join(data_dir, f'movies.dat')
        movies_df = pd.read_csv(
            movies_path,
            encoding='iso-8859-1',
            delimiter='::',
            engine='python',
            header=None,
            names=['movie_id', 'movie_name', 'genre']
        )

        # ratings
        ratings_path = os.path.join(data_dir, f'ratings.dat')
        ratings_df = pd.read_csv(
            ratings_path,
            delimiter='::',
            engine='python',
            header=None,
            names=['user_id', 'movie_id', 'rating', 'time']
        )

        # tags
        tags_path = os.path.join(data_dir, f'tags.dat')
        tags_df = pd.read_csv(
            tags_path,
            delimiter='::',
            engine='python',
            header=None,
            names=['user_id', 'movie_id', 'tag', 'time']
        )

        return movies_df, ratings_df, tags_df

    # for movie lens 25M
    elif data_type == '20M' or data_type == '25M':
        # movies
        movies_path = os.path.join(data_dir, f'movies.csv')
        movies_df = pd.read_csv(
            movies_path,
            encoding='UTF-8',
            engine='python',
            header=0,
            names=['movie_id', 'movie_name', 'genre'],
        )

        # ratings
        ratings_path = os.path.join(data_dir, f'ratings.csv')
        ratings_df = pd.read_csv(
            ratings_path,
            encoding='UTF-8',
            engine='python',
            header=0,
            names=['user_id', 'movie_id', 'rating', 'time'],
        )

        # tags
        tags_path = os.path.join(data_dir, f'tags.csv')
        tags_df = pd.read_csv(
            tags_path,
            encoding='UTF-8',
            engine='python',
            header=0,
            names=['user_id', 'movie_id', 'tag', 'time'],
        )

        return movies_df, ratings_df, tags_df

    else:
        raise Exception(f'Unrecognized data type {data_type}')



# generate features from loaded 1M data
def make_features_1M(movies_df,
                     users_df,
                     ratings_df,
                     feature_length=128,
                     save_feat=True,
                     output_dir=None):

    # initialize sparse features
    gender, age, occupation, movie_name, genre = [], [], [], [], []

    # IC features: list of movies that each user watches
    positive_ic_feature = np.zeros((len(ratings_df), feature_length))
    positive_ic_feature_length = np.zeros(len(ratings_df))
    negative_ic_feature = np.zeros((len(ratings_df), feature_length))
    negative_ic_feature_length = np.zeros(len(ratings_df))
    # UC features: list of users that each movie is watched by
    positive_uc_feature = np.zeros((len(ratings_df), feature_length))
    positive_uc_feature_length = np.zeros(len(ratings_df))
    negative_uc_feature = np.zeros((len(ratings_df), feature_length))
    negative_uc_feature_length = np.zeros(len(ratings_df))

    # labels
    labels = np.zeros(len(ratings_df))

    for i in tqdm(range(len(ratings_df))):

        # current user and movie id
        cur_user_id = ratings_df['user_id'][i]
        cur_movie_id = ratings_df['movie_id'][i]

        # users attributes
        cur_gender_str = users_df.query(f'user_id=={cur_user_id}')['gender'].to_numpy()[0]
        # assign 0 to male and 1 to female
        cur_gender = 0 if cur_gender_str=='M' else 1
        gender.append(cur_gender)
        age.append(int(users_df.query(f'user_id=={cur_user_id}')['age'].to_numpy()[0]))
        occupation.append(
            int(users_df.query(f'user_id=={cur_user_id}')['occupation'].to_numpy()[0])
        )

        # movies attributes
        movie_name.append(movies_df['movie_name'][cur_movie_id])
        genre.append(movies_df['genre'][cur_movie_id])

        # IC features
        # positive: user rating >= 4 as positive engagement, descending in time
        positive_movie_list = ratings_df \
                            .query(f'user_id=={cur_user_id} & rating>=4')[['movie_id', 'time']] \
                            .sort_values(by='time', ascending=False)['movie_id'].to_numpy()
        # if length is over max feature length, random sample
        if len(positive_movie_list) > feature_length:
            positive_movie_list = np.random.choice(positive_movie_list, feature_length)
            positive_ic_feature_length[i] = feature_length

        positive_ic_feature[i][:len(positive_movie_list)] = positive_movie_list
        positive_ic_feature_length[i] = len(positive_movie_list)
        # negative: user rating < 4 as negative engagement, descending in time
        negative_movie_list = ratings_df \
                            .query(f'user_id=={cur_user_id} & rating<4')[['movie_id', 'time']] \
                            .sort_values(by='time', ascending=False)['movie_id'].to_numpy()
        # if length is over max feature length, random sample
        if len(negative_movie_list) > feature_length:
            negative_movie_list = np.random.choice(negative_movie_list, feature_length)
            negative_ic_feature_length[i] = feature_length

        negative_ic_feature[i][:len(negative_movie_list)] = negative_movie_list
        negative_ic_feature_length[i] = len(negative_movie_list)

        # UC feautures
        # positive: user rating >= 4 as positive engagement
        positive_user_list = ratings_df \
                            .query(f'movie_id=={cur_movie_id} & rating>=4')[['user_id', 'time']] \
                            .sort_values(by='time', ascending=False)['user_id'].to_numpy()
        # if length is over max feature length, random sample
        if len(positive_user_list) > feature_length:
            positive_user_list = np.random.choice(positive_user_list, feature_length)
            positive_uc_feature_length[i] = feature_length

        positive_uc_feature[i-1][:len(positive_user_list)] = positive_user_list
        positive_uc_feature_length[i-1] = len(positive_user_list)
        # negative: user rating < 4 as negative engagement
        negative_user_list = ratings_df \
                            .query(f'movie_id=={cur_movie_id} & rating<4')[['user_id', 'time']] \
                            .sort_values(by='time', ascending=False)['user_id'].to_numpy()
        # if length is over max feature length, random sample
        if len(negative_user_list) > feature_length:
            negative_user_list = np.random.choice(negative_user_list, feature_length)
            negative_uc_feature_length[i] = feature_length

        negative_uc_feature[i-1][:len(negative_user_list)] = negative_user_list
        negative_uc_feature_length[i-1] = len(negative_user_list)

        # labels (binary)
        if ratings_df['rating'][i] >= 4.0:
            labels[i] = 1
        else:
            labels[i] = 0

    # features, dropping time
    features_df = ratings_df[['user_id', 'movie_id', 'rating']].copy(deep=True)
    features_df['gender'] = gender
    features_df['age'] = age
    features_df['occupation'] = occupation
    features_df['movie_name'] = movie_name
    features_df['genre'] = genre
    features_df['labels'] = labels

    # save generated features
    if save_feat:
        # sparse features
        features_df_path = os.path.join(output_dir, f'movie_lens_1M_sparse_features.csv')
        # remove the old one
        if os.path.exists(features_df_path):
            os.remove(features_df_path)
            print('\nRemoved previously generated sparse features')

        # create dict and save
        features_df.to_csv(features_df_path)
        print(f'Sparse features has been saved to {features_df_path}')

        # IC/UC features
        ic_uc_path = os.path.join(output_dir, f'movie_lens_1M_IC_UC_features.npz')
        if os.path.exists(ic_uc_path):
            os.remove(ic_uc_path)
            print('\nRemoved previously generated IC/UC features')

        # IC and UC features
        # create dict and save
        arrays_to_save = {
            "positive_ic_feature": positive_ic_feature,
            "positive_ic_feature_length": positive_ic_feature_length,
            "negative_ic_feature": negative_ic_feature,
            "negative_ic_feature_length": negative_ic_feature_length,
            "positive_uc_feature": positive_uc_feature,
            "positive_uc_feature_length": positive_uc_feature_length,
            "negative_uc_feature": negative_uc_feature,
            "negative_uc_feature_length": negative_uc_feature_length,
        }
        np.savez(ic_uc_path, **arrays_to_save)
        print(f'IC/UC features has been saved to {ic_uc_path}')


# generate features from loaded 10M data
def make_features(movies_df,
                    ratings_df,
                    tags_df, # not using tags for now
                    feature_length=256,
                    save_feat=True,
                    output_dir=None,
                    truncate_index=None,
):

    def task(truncate_index, sub_ratings_df, start, end):
        # labels
        num_ratings = int(end-start)
        labels = np.zeros(num_ratings)

        # initialize sparse features
        movie_name, genre = [], []

        # IC features: list of movies that each user watches
        positive_ic_feature = np.zeros((num_ratings, feature_length))
        positive_ic_feature_length = np.zeros(num_ratings)
        negative_ic_feature = np.zeros((num_ratings, feature_length))
        negative_ic_feature_length = np.zeros(num_ratings)
        # UC features: list of users that each movie is watched by
        positive_uc_feature = np.zeros((num_ratings, feature_length))
        positive_uc_feature_length = np.zeros(num_ratings)
        negative_uc_feature = np.zeros((num_ratings, feature_length))
        negative_uc_feature_length = np.zeros(num_ratings)

        for i in tqdm(range(num_ratings)):
            # current user and movie id
            cur_user_id = sub_ratings_df['user_id'][i+start]
            cur_movie_id = sub_ratings_df['movie_id'][i+start]

            # movies attributes
            movie_name.append(movies_df.query(f'movie_id=={cur_movie_id}')['movie_name'])
            genre.append(movies_df.query(f'movie_id=={cur_movie_id}')['genre'])

            # IC features
            # positive: user rating >= 4 as positive engagement, descending in time
            positive_movie_list = sub_ratings_df \
                                .query(f'user_id=={cur_user_id} & rating>=4')[['movie_id', 'time']] \
                                .sort_values(by='time', ascending=False)['movie_id'].to_numpy()
            # if length is over max feature length, random sample
            if len(positive_movie_list) > feature_length:
                positive_movie_list = np.random.choice(positive_movie_list, feature_length)
                positive_ic_feature_length[i] = feature_length

            positive_ic_feature[i][:len(positive_movie_list)] = positive_movie_list
            positive_ic_feature_length[i] = len(positive_movie_list)
            # negative: user rating < 4 as negative engagement, descending in time
            negative_movie_list = sub_ratings_df \
                                .query(f'user_id=={cur_user_id} & rating<4')[['movie_id', 'time']] \
                                .sort_values(by='time', ascending=False)['movie_id'].to_numpy()
            # if length is over max feature length, random sample
            if len(negative_movie_list) > feature_length:
                negative_movie_list = np.random.choice(negative_movie_list, feature_length)
                negative_ic_feature_length[i] = feature_length

            negative_ic_feature[i][:len(negative_movie_list)] = negative_movie_list
            negative_ic_feature_length[i] = len(negative_movie_list)

            # UC feautures
            # positive: user rating >= 4 as positive engagement
            positive_user_list = sub_ratings_df \
                                .query(f'movie_id=={cur_movie_id} & rating>=4')[['user_id', 'time']] \
                                .sort_values(by='time', ascending=False)['user_id'].to_numpy()
            # if length is over max feature length, random sample
            if len(positive_user_list) > feature_length:
                positive_user_list = np.random.choice(positive_user_list, feature_length)
                positive_uc_feature_length[i] = feature_length

            positive_uc_feature[i-1][:len(positive_user_list)] = positive_user_list
            positive_uc_feature_length[i-1] = len(positive_user_list)
            # negative: user rating < 4 as negative engagement
            negative_user_list = sub_ratings_df \
                                .query(f'movie_id=={cur_movie_id} & rating<4')[['user_id', 'time']] \
                                .sort_values(by='time', ascending=False)['user_id'].to_numpy()
            # if length is over max feature length, random sample
            if len(negative_user_list) > feature_length:
                negative_user_list = np.random.choice(negative_user_list, feature_length)
                negative_uc_feature_length[i] = feature_length

            negative_uc_feature[i-1][:len(negative_user_list)] = negative_user_list
            negative_uc_feature_length[i-1] = len(negative_user_list)

            # labels (binary)
            if sub_ratings_df['rating'][i+start] >= 4.0:
                labels[i] = 1
            else:
                labels[i] = 0


        # features, dropping time
        features_df = sub_ratings_df[['user_id', 'movie_id', 'rating']].copy(deep=True)
        features_df['movie_name'] = movie_name
        features_df['genre'] = genre
        features_df['labels'] = labels

        # save generated features
        if save_feat:
            # sparse features
            features_df_path = os.path.join(
                output_dir, f'movie_lens_{data_type}_sparse_features_{truncate_index}.csv'
            )
            # remove the old one
            if os.path.exists(features_df_path):
                os.remove(features_df_path)
                print('\nRemoved previously generated sparse features')

            # create dict and save
            features_df.to_csv(features_df_path)
            print(f'Sparse features has been saved to {features_df_path}')

            # IC/UC features
            ic_uc_path = os.path.join(
                output_dir, f'movie_lens_{data_type}_IC_UC_features_{truncate_index}.npz'
            )
            if os.path.exists(ic_uc_path):
                os.remove(ic_uc_path)
                print('\nRemoved previously generated IC/UC features')

            # IC and UC features
            # create dict and save
            arrays_to_save = {
                "positive_ic_feature": positive_ic_feature,
                "positive_ic_feature_length": positive_ic_feature_length,
                "negative_ic_feature": negative_ic_feature,
                "negative_ic_feature_length": negative_ic_feature_length,
                "positive_uc_feature": positive_uc_feature,
                "positive_uc_feature_length": positive_uc_feature_length,
                "negative_uc_feature": negative_uc_feature,
                "negative_uc_feature_length": negative_uc_feature_length,
            }
            np.savez(ic_uc_path, **arrays_to_save)
            print(f'IC/UC features has been saved to {ic_uc_path}')

    # prepare multiprocessing
    print("Number of cpu : ", cpu_count())
    # # truncate into 1M ratings
    # num_truncate = int(np.floor(len(ratings_df) / 1e6))
    # print(f'Truncated into {num_truncate} processes')
    # processes = []
    # for t in range(num_truncate):
    #     start = int(t*1e6)
    #     if data_type == '10M':
    #         max_truncate = 9
    #     elif data_type == '20M':
    #         max_truncate = 19
    #     if t == max_truncate:
    #         end = int(len(ratings_df))
    #     else:
    #         end = int((t+1)*1e6)

    #     sub_ratings_df = ratings_df[start:end]
    #     # create subprocess
    #     processes.append(Process(target=task, args=(t, sub_ratings_df, start, end)))

    # for process in processes:
    #     process.start()
    # for process in processes:
    #     process.join()
    # print('Done', flush=True)
    start = int(truncate_index*1e6)
    if data_type == '10M':
        max_truncate = 9
    elif data_type == '20M':
        max_truncate = 19
    if truncate_index == max_truncate:
        end = int(len(ratings_df))
    else:
        end = int((truncate_index+1)*1e6)
    sub_ratings_df = ratings_df[start:end]
    print(f'Processing truncate index {truncate_index} of {data_type}, from {start} to {end-1}')
    task(truncate_index, sub_ratings_df, start, end)



if __name__ == "__main__":
    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', action='store', nargs=1, dest='data_type', required=True)
    parser.add_argument('--input_dir', action='store', nargs=1, dest='input_dir', required=True)
    parser.add_argument('--output_dir', action='store', nargs=1, dest='output_dir', required=True)
    parser.add_argument('--truncate_index', action='store', nargs=1, dest='truncate_index')
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False)
    args = parser.parse_args()
    data_type = args.data_type[0]
    input_dir = args.input_dir[0]
    output_dir = args.output_dir[0]
    if args.truncate_index:
        truncate_index = int(args.truncate_index[0])
    verbose = args.verbose

    if verbose:
        print(f'Data dir: {input_dir}\n')

    # generate and save the features
    if data_type == '1M':
        # load data
        movies_df, users_df, ratings_df = load_data(input_dir, data_type, real_occupation=False)
        if verbose:
            print(f'Number of users: {len(users_df)}')
            print(f'{users_df.head()}\n')
            print(f'Number of movies: {len(movies_df)}')
            print(f'{movies_df.head()}\n')
            print(f'Number of ratings: {len(ratings_df)}')
            print(f'{ratings_df.head()}\n')

        # make and save features
        make_features_1M(
            movies_df,
            users_df,
            ratings_df,
            save_feat=True,
            output_dir=output_dir
        )
    elif data_type == '10M' or data_type == '20M':
        # load data
        movies_df, ratings_df, tags_df = load_data(input_dir, data_type, real_occupation=False)
        if verbose:
            if data_type == '10M':
                print(f'Number of movies: {len(movies_df)}/10681')
                print(f'{movies_df.head()}\n')
                print(f'Number of ratings: {len(ratings_df)}/10000054')
                print(f'{ratings_df.head()}\n')
                print(f'Number of tags: {len(tags_df)}/95580')
                print(f'{ratings_df.head()}\n')
            elif data_type == '20M':
                print(f'Number of movies: {len(movies_df)}/27278')
                print(f'{movies_df.head()}\n')
                print(f'Number of ratings: {len(ratings_df)}/20000263')
                print(f'{ratings_df.head()}\n')
                print(f'Number of tags: {len(tags_df)}/465564')
                print(f'{ratings_df.head()}\n')

        # make and save features
        make_features(
            movies_df,
            ratings_df,
            ratings_df,
            save_feat=True,
            output_dir=output_dir,
            truncate_index=truncate_index,
        )
    else:
        raise Exception(f'Unrecognized data type {data_type}')


