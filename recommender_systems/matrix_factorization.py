import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.utils import shuffle


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


# go over all the ratings in the dict,
# make a prediction as dot product of user's
# preferences and movie attributes,
# compute sum of squared errors
def get_loss(d):
    N = float(len(d))
    sse = 0
    for (u, m), r in d.items():
        p = W[u].dot(U[m]) + b[u] + c[m] + mu
        sse += (r - p) * (r - p)

    return sse / N


epochs = 25
reg = 20.
train_loss = []
test_loss = []

for _ in tqdm(range(epochs)):

    # update of user matrix W
    for u in range(N):

        matrix = np.eye(K) * reg
        vector = np.zeros(K)

        b_u = 0
        for m in user2movie[u]:
            r = usermovie2rating[(u, m)]
            matrix += np.outer(U[m], U[m])
            vector += (r - b[u] - c[m] - mu) * U[m]
            b_u += r - W[u].dot(U[m]) - c[m] - mu

        b[u] = b_u / (len(user2movie[u]) + reg)
        W[u] = np.linalg.solve(matrix, vector)

    # update of movie matrix U
    for m in range(M):

        matrix = np.eye(K) * reg
        vector = np.zeros(K)

        c_m = 0
        for u in movie2user[m]:
            r = usermovie2rating[(u, m)]
            matrix += np.outer(W[u], W[u])
            vector += (r - b[u] - c[m] - mu) * W[u]
            c_m += r - W[u].dot(U[m]) - b[u] - mu

        c[m] = c_m / (len(movie2user[m]) + reg)
        U[m] = np.linalg.solve(matrix, vector)

    train_loss.append(get_loss(usermovie2rating))
    test_loss.append(get_loss(usermovie2rating_test))

plt.plot(train_loss, label='train')
plt.plot(test_loss, label='test')
plt.legend()

plt.show()
