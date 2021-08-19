import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from sklearn.utils import shuffle
from sortedcontainers import SortedList
from sklearn.metrics import mean_squared_error

warnings.simplefilter(action='ignore', category=FutureWarning)

# loading dataset
df = pd.read_csv('./rating.csv')

# We need to perform remapping of userIds and movieIds,
# so that they start to count from 0 up to max value.
# We'll use it later for indexing from our dictionaries.
print('Before remapping:')
print(f'User with min index: {np.min(df.userId)} and max index: {np.max(df.userId)}')
print(f'Movie with min index: {np.min(df.movieId)} and with max index: {np.max(df.movieId)}')

df.userId = df.userId - 1

# For movieId we have to define new mapping
# as they're not sequential
unique_movies = set(df.movieId)
movie2idx = {}
count = 0
for movie in unique_movies:
    if movie not in movie2idx.keys():
        movie2idx[movie] = count
        count += 1
    else:
        print('Someone rated the same movie twice')

tqdm.pandas()
df['movie_idx'] = df.progress_apply(lambda row: movie2idx[row.movieId], axis=1)
print('After remapping:')
print(f'User with min index: {np.min(df.userId)} and max index: {np.max(df.userId)}')
print(f'Movie with min index: {np.min(df.movie_idx)} and with max index: {np.max(df.movie_idx)}')

df.drop(['timestamp', 'movieId'], axis=1, inplace=True)

# As the dataset is too big we will pick small subset
# of the most common users and most rated movies
user_counter = Counter(df.userId)
movie_counter = Counter(df.movie_idx)

n = 1000
m = 200

most_common_users = [u for u, c in user_counter.most_common(n)]
most_common_movies = [m for m, c in movie_counter.most_common(m)]

df_small = df[(df.userId.isin(most_common_users)) & (df.movie_idx.isin(most_common_movies))].copy()

i = 0
new_user_mapping = {}
for old in most_common_users:
    new_user_mapping[old] = i
    i += 1

j = 0
new_movie_mapping = {}
for old in most_common_movies:
    new_movie_mapping[old] = j
    j += 1

df_small.loc[:, 'userId'] = df_small.progress_apply(lambda row: new_user_mapping[row.userId], axis=1)
df_small.loc[:, 'movie_idx'] = df_small.progress_apply(lambda row: new_movie_mapping[row.movie_idx], axis=1)

print(f'Min userId: {df_small.userId.min()} and max userId: {df_small.userId.max()}')
print(f'Min movie_idx: {df_small.movie_idx.min()} and max movie_idx: {df_small.movie_idx.max()}')

df_small = shuffle(df_small)
cutoff = int(.8 * len(df_small))
df_train = df_small.iloc[:cutoff]
df_test = df_small.iloc[cutoff:]

# creating new mappings
user2movie = {}
movie2user = {}
usermovie2rating = {}


def get_train_mappings(row):
    i, j = int(row.userId), int(row.movie_idx)

    if i not in user2movie:
        user2movie[i] = [j]
    else:
        user2movie[i].append(j)

    if j not in movie2user:
        movie2user[j] = [i]
    else:
        movie2user[j].append(i)

    usermovie2rating[(i, j)] = row.rating


df_train.progress_apply(get_train_mappings, axis=1)

usermovie2rating_test = {}


def get_test_mapping(row):
    i, j = int(row.userId), int(row.movie_idx)
    usermovie2rating_test[(i, j)] = row.rating


df_test.progress_apply(get_test_mapping, axis=1)

M = len(movie2user)

K = 25
limit = 5
# average rating for each movie
biases = []
# top K most similar movies for every movie measured as weights
neighbors = []
# movie rating minus its bias
deviations = []

for i in tqdm(range(M)):
    users_i = movie2user[i]
    users_i_set = set(users_i)

    ratings_i = {user: usermovie2rating[(user, i)] for user in users_i}
    bias_i = np.mean(list(ratings_i.values()))
    dev_i = {user: rating - bias_i for user, rating in ratings_i.items()}
    dev_i_vals = np.asarray(list(dev_i.values()))
    sigma_i = np.sqrt(dev_i_vals.dot(dev_i_vals))

    biases.append(bias_i)
    deviations.append(dev_i)

    sl = SortedList()
    for j in range(M):

        if j != i:
            users_j = movie2user[j]
            users_j_set = set(users_j)
            common_users = users_i_set & users_j_set

            if len(common_users) > limit:
                ratings_j = {user: usermovie2rating[(user, j)] for user in users_j}
                bias_j = np.mean(list(ratings_j.values()))
                dev_j = {user: rating - bias_j for user, rating in ratings_j.items()}
                dev_j_vals = np.asarray(list(dev_j.values()))
                sigma_j = np.sqrt(dev_j_vals.dot(dev_j_vals))

                numerator = sum((dev_i[user] * dev_j[user]) for user in common_users)
                w_ij = numerator / (sigma_i * sigma_j)
                sl.add((-w_ij, j))

                if len(sl) > K:
                    del sl[-1]

    neighbors.append(sl)


# given user and movie we want to predict his rating for this movie
def predict(u, m):
    numerator = 0
    denominator = 0
    for neg_w, movie in neighbors[m]:

        try:
            numerator += -neg_w * deviations[movie][u]
            denominator += abs(neg_w)
        except KeyError:
            pass

    if denominator == 0:
        prediction = biases[m]
    else:
        prediction = numerator / denominator + biases[m]

    prediction = min(5, prediction)
    prediction = max(.5, prediction)

    return prediction


# computing train and test loss
train_predictions = []
train_targets = []
for (u, m), target in usermovie2rating.items():
    # calculate the prediction for this movie
    prediction = predict(u, m)

    # save the prediction and target
    train_predictions.append(prediction)
    train_targets.append(target)

test_predictions = []
test_targets = []
# same thing for test set
for (u, m), target in usermovie2rating_test.items():
    # calculate the prediction for this movie
    prediction = predict(u, m)

    # save the prediction and target
    test_predictions.append(prediction)
    test_targets.append(target)

print('train mse:', mean_squared_error(train_targets, train_predictions))
print('test mse:', mean_squared_error(test_targets, test_predictions))
