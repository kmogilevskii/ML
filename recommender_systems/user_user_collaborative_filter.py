import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from sklearn.utils import shuffle
from sortedcontainers import SortedList
from sklearn.metrics import mean_squared_error

# https://www.kaggle.com/grouplens/movielens-20m-dataset
df = pd.read_csv('./rating.csv')

# user ids are ordered sequentially from 1..138493
# with no missing numbers
# movie ids are integers from 1..131262
# NOT all movie ids appear
# there are only 26744 movie ids
# write code to check it yourself!


# make the user ids go from 0...N-1
df.userId = df.userId - 1

# create a mapping for movie ids
unique_movie_ids = set(df.movieId.values)
movie2idx = {}
count = 0
for movie_id in unique_movie_ids:
    movie2idx[movie_id] = count
    count += 1

# add them to the data frame
# takes awhile
df['movie_idx'] = df.apply(lambda row: movie2idx[row.movieId], axis=1)

# no need for these fields
df = df.drop(columns=['timestamp', 'movieId'])

N = df.userId.max() + 1
M = df.movieId.max() + 1
print(f'We have {N} unique users and {M} unique movies.')

# here we want to reduce our initial dataset to be able to run the training algorithm
# we do this by picking first n most active users and m most rated movies
n = 1000
m = 200

user_ids_count = Counter(df.userId)
movie_ids_count = Counter(df.movieId)

most_common_users = [u for u, c in user_ids_count.most_common(n)]
most_common_movies = [m for m, c in movie_ids_count.most_common(m)]

df_small = df[df.userId.isin(most_common_users) & df.movie_idx.isin(most_common_movies)].copy()
print(f'New shape is {df_small.shape}')

new_user_mapping = {}
i = 0
for old in most_common_users:
    new_user_mapping[old] = i
    i += 1
print(f'i: {i}')

new_movies_mapping = {}
j = 0
for old in most_common_movies:
    new_movies_mapping[old] = j
    j += 1
print(f'j: {j}')

df_small.loc[:, 'userId'] = df_small.apply(lambda row: new_user_mapping[row.userId], axis=1)
df_small.loc[:, 'movie_idx'] = df_small.apply(lambda row: new_movies_mapping[row.movie_idx], axis=1)

print(f'Final user id {df_small.userId.max()}')
print(f'Final movie id {df_small.movie_idx.max()}')

# we actually want to work not with dataframe, but with set of dictionaries
# this will make our code easier to implement
tqdm.pandas()

df_small = shuffle(df_small)
cutoff = int(.8 * df_small.shape[0])
df_train = df_small.iloc[:cutoff]
df_test = df_small.iloc[cutoff:]

user2movie = {}
movie2user = {}
usermovie2rating = {}


def get_all_train_mappings(row):
    i = int(row.userId)
    j = int(row.movie_idx)
    if i not in user2movie:
        user2movie[i] = [j]
    else:
        user2movie[i].append(j)

    if j not in movie2user:
        movie2user[j] = [i]
    else:
        movie2user[j].append(i)

    usermovie2rating[(i, j)] = row.rating


df_train.progress_apply(get_all_train_mappings, axis=1)

usermovie2rating_test = {}


def get_test_mapping(row):
    i = int(row.userId)
    j = int(row.movie_idx)

    usermovie2rating_test[(i, j)] = row.rating


df_test.progress_apply(get_test_mapping, axis=1)

N = np.max(list(user2movie.keys())) + 1
m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (u, m), r in usermovie2rating_test.items()])
M = max(m1, m2) + 1
print(f'Number of users: {N} and number of movies: {M}')

# next step would be to find top K neighbors for every user
# and convert from absolute ratings to deviations
K = 25
limit = 5
neighbors = []
biases = []
deviations = []

for i in tqdm(range(N)):
    movies_i = user2movie[i]
    movies_i_set = set(movies_i)

    ratings_i = {movie: usermovie2rating[(i, movie)] for movie in movies_i}
    bias_i = np.mean(list(ratings_i.values()))
    dev_i = {movie: (rating - bias_i) for movie, rating in ratings_i.items()}
    dev_i_vals = np.asarray(list(dev_i.values()))
    sigma_i = np.sqrt(dev_i_vals.dot(dev_i_vals))

    deviations.append(dev_i)
    biases.append(bias_i)

    sl = SortedList()
    for j in range(N):
        if j != i:
            movies_j = user2movie[j]
            movies_j_set = set(movies_j)
            common_movies = movies_i_set & movies_j_set
            if len(common_movies) > limit:
                ratings_j = {movie: usermovie2rating[(j, movie)] for movie in movies_j}
                bias_j = np.mean(list(ratings_j.values()))
                dev_j = {movie: (rating - bias_j) for movie, rating in ratings_j.items()}
                dev_j_vals = np.asarray(list(dev_j.values()))
                sigma_j = np.sqrt(dev_j_vals.dot(dev_j_vals))

                numerator = np.sum([dev_i[movie] * dev_j[movie] for movie in common_movies])
                denominator = sigma_i * sigma_j

                w_ij = numerator / denominator
                sl.add((-w_ij, j))
                if len(sl) > K:
                    del sl[-1]

    neighbors.append(sl)


# function to make a prediction
def predict(i, m):
    numerator = 0
    denominator = 0
    for neg_w, j in neighbors[i]:
        try:
            numerator += -neg_w * deviations[j][m]
            denominator += np.abs(neg_w)
        except KeyError:
            pass

    if denominator == 0:
        prediction = biases[i]
    else:
        prediction = numerator / denominator + biases[i]
    prediction = min(5, prediction)
    prediction = max(.5, prediction)

    return prediction

# testing
train_predictions = []
train_target = []
for (i, m), target in usermovie2rating.items():
  prediction = predict(i, m)

  train_predictions.append(prediction)
  train_target.append(target)

test_predictions = []
test_target = []
for (i, m), target in usermovie2rating_test.items():
  prediction = predict(i, m)

  test_predictions.append(prediction)
  test_target.append(target)

print(f'Train MSE: {mean_squared_error(train_target, train_predictions):.3f}')
print(f'Test MSE: {mean_squared_error(test_target, test_predictions):.3f}')
