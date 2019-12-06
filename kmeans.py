import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    headers = ['stg', 'scg', 'str', 'lpr', 'peg', 'uns']
    df = pd.read_csv('dataset/data_user_modeling.csv', header=None, names=headers)

    kmeans = KMeans(n_clusters=4, random_state=0)
    kmeans.fit(df[headers[:-1]])

    # for idx, row in enumerate(df['uns']):
    #     print(row, kmeans.labels_[idx])

    print(davies_bouldin_score(df[headers[:-1]], kmeans.labels_))
    print(silhouette_score(df[headers[:-1]], kmeans.labels_))
