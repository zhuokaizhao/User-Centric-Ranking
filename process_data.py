# The script loads MovieLens data and generate IC, UC features
import os
import argparse
import numpy as np
import pandas as pd

# load and process MovieLens data
def load_data(data_dir, data_type, real_occupation=False):

    if data_type == '1M':
        file_type = 'dat'
        # movies
        movies_path = os.path.join(data_dir, f"movies.{file_type}")
        movies_df = pd.read_csv(movies_path,
                                encoding='iso-8859-1',
                                delimiter='::',
                                engine='python',
                                header=None,
                                names=['movie_name', 'genre'])

        # users
        users_path = os.path.join(data_dir, f'users.{file_type}')
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
        ratings_path = os.path.join(data_dir, f'ratings.{file_type}')
        ratings_df = pd.read_csv(ratings_path,
                                delimiter='::',
                                engine='python',
                                header=None,
                                names=['user_id', 'movie_id', 'rating', 'time'])
    elif data_type == '25M':
        file_type = 'csv'
        # TODO


    return movies_df, users_df, ratings_df


# generate features from loaded data
def make_features(data_type,
                  movies_df,
                  users_df,
                  ratings_df,
                  feature_length=128,
                  save_feat=False,
                  output_dir=None):

    # features
    # from users_df
    user_id = users_df['user_id'].to_numpy() # user id, 0 is mask value
    gender = users_df['gender'].to_numpy() # user gender
    age = users_df['age'].to_numpy() # user age
    occupation = users_df['occupation'].to_numpy() # user occupuation
    zip_code = users_df['zip_code'].to_numpy() # user zip code

    # from movies_df
    movie_id = np.array(movies_df.index.values) # movie id, 0 is mask value
    title = movies_df['movie_name'].to_numpy() # movies title
    genre = movies_df['genre'].to_numpy() # movies genre

    # from ratings_df
    # IC features: list of movies that each user watches
    ic_feat = np.zeros((len(user_id), feature_length))
    for i in range(1, len(user_id)):
        # user rating >= 3 as positive engagement
        positive_movie_list = ratings_df.query(f'user_id=={i} & rating>=3')['movie_id'].to_numpy()
        # if length is over max feature length, random sample
        if len(positive_movie_list) > feature_length:
            positive_movie_list = np.random.choice(positive_movie_list, feature_length)

        ic_feat[i-1][:len(positive_movie_list)] = positive_movie_list

    # UC features: list of users that each movie is watched by
    uc_feat = np.zeros((len(movie_id), feature_length))
    for i in range(1, len(movie_id)):
        # user rating >= 3 as positive engagement
        positive_user_list = ratings_df.query(f'movie_id=={i} & rating>=3')['user_id'].to_numpy()
        # if length is over max feature length, random sample
        if len(positive_user_list) > feature_length:
            positive_user_list = np.random.choice(positive_user_list, feature_length)

        uc_feat[i-1][:len(positive_user_list)] = positive_user_list

    if save_feat:
        output_path = os.path.join(output_dir, f'movie_lens_{data_type}_ic_uc_features.npz')
        arrays_to_save = {
            "user_id": user_id,
            "movie_id": movie_id,
            "gender": gender,
            "age": age,
            "occupation": occupation,
            "zip_code": zip_code,
            "title": title,
            "genre": genre,
            "ic_feat": ic_feat,
            "uc_feat": uc_feat,
        }
        np.savez(output_path, **arrays_to_save)



if __name__ == "__main__":
    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', action='store', nargs=1, dest='data_type', required=True)
    parser.add_argument('--input_dir', action='store', nargs=1, dest='input_dir', required=True)
    parser.add_argument('--output_dir', action='store', nargs=1, dest='output_dir', required=True)
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False)
    args = parser.parse_args()
    data_type = args.data_type[0]
    input_dir = args.input_dir[0]
    output_dir = args.output_dir[0]
    verbose = args.verbose

    if verbose:
        print(f'Data dir: {input_dir}\n')

    # load data
    movies_df, users_df, ratings_df = load_data(input_dir, data_type, real_occupation=False)
    if verbose:
        print(f'Number of users: {len(users_df)}')
        print(f'{users_df.head()}\n')
        print(f'Number of movies: {len(movies_df)}')
        print(f'{movies_df.head()}\n')
        print(f'Number of ratings: {len(ratings_df)}')
        print(f'{ratings_df.head()}\n')

    # generate and save the features
    all_features = make_features(data_type,
                                 movies_df,
                                 users_df,
                                 ratings_df,
                                 save_feat=True,
                                 output_dir=output_dir)


