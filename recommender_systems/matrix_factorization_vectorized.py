import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.utils import shuffle
from sortedcontainers import SortedList
from sklearn.metrics import mean_squared_error

warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv('./rating.csv')

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

user2movie_rating = {}
for u, movies in user2movie.items():
    ratings = np.asarray([usermovie2rating[(u, m)] for m in movies])
    user2movie_rating[u] = (movies, ratings)

movie2user_rating = {}
for m, users in movie2user.items():
    ratings = np.asarray([usermovie2rating[(u, m)] for u in users])
    movie2user_rating[m] = (users, ratings)

movie2user_rating_test = {}
for (u, m), r in usermovie2rating_test.items():
    if m not in movie2user_rating_test.keys():
        movie2user_rating_test[m] = [[u], [r]]
    else:
        movie2user_rating_test[m][0].append(u)
        movie2user_rating_test[m][1].append(r)

for m, (u, r) in movie2user_rating_test.items():
    movie2user_rating_test[m][1] = np.asarray(r)

N = max(list(user2movie.keys())) + 1
m1 = max(list(movie2user.keys()))
m2 = max([m for (u, m), r in usermovie2rating_test.items()])
M = max(m1, m2) + 1

K = 10  # latent dimensionality
W = np.random.randn(N, K)
b = np.zeros(N)
U = np.random.randn(M, K)
c = np.zeros(M)
mu = np.mean(list(usermovie2rating.values()))


def get_loss(m2ur):
    N = 0
    sse = 0
    for m, (u, r) in m2ur.items():
        p = W[u].dot(U[m]) + b[u] + c[m] + mu
        delta = p - r
        sse += delta.dot(delta)
        N += len(u)

    return sse / N


reg = 20.
epochs = 25
train_loss = []
test_loss = []

for _ in tqdm(range(epochs)):

    # Updating W matrix without additional inner loop
    for u in range(N):
        movies, ratings = user2movie_rating[u]
        ratings = np.asarray(ratings)
        matrix = U[movies].T.dot(U[movies]) + np.eye(K) * reg
        vector = (ratings - b[u] - c[movies] - mu).dot(U[movies])
        W[u] = np.linalg.solve(matrix, vector)
        b[u] = (ratings - U[movies].dot(W[u]) - c[movies] - mu).sum() / (len(movies) + reg)

    # Updating U matrix without additional inner loop
    for m in range(M):

        try:
            users, ratings = movie2user_rating[m]
            ratings = np.asarray(ratings)
            matrix = W[users].T.dot(W[users]) + np.eye(K) * reg
            vector = (ratings - b[users] - c[m] - mu).dot(W[users])
            U[m] = np.linalg.solve(matrix, vector)
            c[m] = (ratings - W[users].dot(U[m]) - b[users] - mu).sum() / (len(users) + reg)
        except KeyError:
            pass

    train_loss.append(get_loss(movie2user_rating))
    test_loss.append(get_loss(movie2user_rating_test))

plt.plot(train_loss, label='train')
plt.plot(test_loss, label='test')
plt.legend()

plt.show()
