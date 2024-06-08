import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split, ShuffleSplit

from sklearn.cluster import KMeans

from sklearn.metrics import *
from sklearn.cluster import AgglomerativeClustering

from tqdm.auto import tqdm


def plot_error(error, title):
    error = np.array(error)
    plt.figure(figsize=(13, 7))
    plt.plot(error[:, 0], error[:, 1])
    plt.xticks(error[:, 0])
    plt.grid(True)
    plt.title(title)
    plt.show()


def get_prediction_strength(k, train_centroids, x_train_pred, x_test, test_labels):
    n_test = len(x_test)

    D = np.zeros(shape=(n_test, n_test))
    for c1 in range(n_test):
      for c2 in range(c1 + 1, n_test):
        D[c1, c2] = int(x_train_pred[c1] == x_train_pred[c2])
        D[c2, c1] = D[c1, c2]

    ss = []
    for j in range(k):
        s = 0
        n_examples_j = sum(test_labels == j)
        for l1, c1 in zip(test_labels, list(range(n_test))):
            for l2, c2 in zip(test_labels, list(range(n_test))):
                if l1 == l2 and l1 == j:
                    s += D[c1,c2]
        ss.append(s / (n_examples_j * (n_examples_j - 1)))
    prediction_strength = min(ss)

    return prediction_strength


def task5(df):
    inertia_list = []
    silhouette_list = []
    centers_list = []

    model = KMeans(n_clusters=1, init='random', n_init=1, random_state=1).fit(df)
    inertia_list.append([1, model.inertia_])

    for i in range(2, 11):
        model = KMeans(n_clusters=i, init='random', n_init=1, random_state=1).fit(df)
        inertia_list.append([i, model.inertia_])
        silhouette_list.append([i, silhouette_score(df, model.labels_)])
        centers_list.append(model.cluster_centers_)

    plot_error(inertia_list, 'elbow method')
    plot_error(silhouette_list, 'average silhouette method')

    # x_train, x_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=1)

    spliter = ShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
    train_df_i, test_df_i = list(spliter.split(df))[0]
    x_train, x_test = df.loc[train_df_i], df.loc[test_df_i]

    strengths = []
    for k in tqdm(range(1, 11)):
        model_train = KMeans(n_clusters=k, init='random', n_init=1, random_state=1).fit(x_train)
        model_test = KMeans(n_clusters=k, init='random', n_init=1, random_state=1).fit(x_test)

        pred_str = get_prediction_strength(k, model_train.cluster_centers_, model_train.predict(x_test), x_test,
                                           model_test.labels_)
        strengths.append(pred_str)

    _, ax = plt.subplots()
    ax.plot(range(1, 11), strengths, '-o', color='black')
    ax.axhline(y=0.8, c='red')
    ax.set(title='Determining the optimal number of clusters',
           xlabel='number of clusters',
           ylabel='prediction strength')


def task6(df):
    best_model = [None, 0]
    chosen_model = 0
    silhouette_list = []

    for i in range(10):
        model = KMeans(n_clusters=2, init='k-means++', n_init=1)
        model.fit(df)
        silhouette_list.append([i, silhouette_score(df, model.labels_)])
        if best_model[1] < silhouette_list[-1][1]:
            best_model[0] = model
            chosen_model = i + 1
            best_model[1] = silhouette_list[-1][1]

    print(f'Top model - {chosen_model}')

    plot_error(silhouette_list, 'silhoute score detecting')

    return best_model


def task7(df):
    model_AgglomerativeClustering = AgglomerativeClustering(n_clusters=2).fit(df)

    # Getting cluster labels and unique labels
    cluster_labels = np.unique(model_AgglomerativeClustering.labels_)

    # Computing cluster representatives
    cluster_centers = []
    for label in cluster_labels:
        cluster_centers.append(np.mean(df[model_AgglomerativeClustering.labels_ == label], axis=0))

    print("Cluster Centers:")
    for center in cluster_centers:
        print(center)

    return model_AgglomerativeClustering


def task8(df, best_model, model_AgglomerativeClustering):
    AgglomerativeClustering_score = silhouette_score(df, model_AgglomerativeClustering.labels_)
    print(AgglomerativeClustering_score)

    kmeans_score = silhouette_score(df, best_model[0].labels_)
    print(kmeans_score)


def main():
    # 1
    df = pd.read_csv('WQ-R.csv', sep=';')

    # 2
    print("Rows:", df.shape[0])
    print("Cols:", df.shape[1])

    # 3
    df = df.drop('quality', axis=1)

    # 4
    for i, c in enumerate(df.columns):
        print(f'{i + 1}) {c}')

    # 5
    task5(df)

    # 6
    best_model = task6(df)

    # 7
    model_AgglomerativeClustering = task7(df)

    # 8
    task8(df, best_model, model_AgglomerativeClustering)


if __name__ == '__main__':
    main()
