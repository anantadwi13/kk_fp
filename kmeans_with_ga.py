import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from ga import GeneticAlgorithm


# # custom fitness function
# def fitness_function_knn(**kwargs) -> float:
#     try:
#         k = int("".join(x for x in kwargs['chromosome']))
#         model = KNeighborsClassifier(n_neighbors=k)
#         model.fit(kwargs['train_x'], kwargs['train_y'])
#         pred_y = model.predict(kwargs['test_x'])
#         return precision_score(kwargs['test_y'], pred_y, average='micro')
#     except Exception as e:
#         # exit(e.args[0])
#         return 0


if __name__ == '__main__':
    headers = ['stg', 'scg', 'str', 'lpr', 'peg', 'uns']
    df = pd.read_csv('dataset/data_user_modeling.csv', header=None, names=headers)

    kmeans = KMeans(n_clusters=4, random_state=0)
    kmeans.fit(df[headers[:-1]])

    # for idx, row in enumerate(df['uns']):
    #     print(row, kmeans.labels_[idx])

    print(davies_bouldin_score(df[headers[:-1]], kmeans.labels_))
    print(silhouette_score(df[headers[:-1]], kmeans.labels_))
