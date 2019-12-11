import pandas as pd
import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from ga import GeneticAlgorithm


# custom fitness function
def fitness_function_kmeans(**kwargs) -> float:
    try:
        cluster_size = kwargs['cluster_size']
        kmeans = KMeans(n_clusters=cluster_size, random_state=0, init=np.array(kwargs['chromosome']), n_init=1)
        headers = kwargs['headers']
        kmeans.fit(kwargs['df'][headers[:-1]])
        return davies_bouldin_score(kwargs['df'][headers[:-1]], kmeans.labels_)
    except Exception as e:
        # exit(e.args[0])
        return 0


if __name__ == '__main__':
    headers = ['stg', 'scg', 'str', 'lpr', 'peg', 'uns']
    df = pd.read_csv('dataset/data_user_modeling.csv', header=None, names=headers)

    ga = GeneticAlgorithm(population_size=200, max_generation=10, gene_type='array_range_float',
                          genes_size=3, cluster_size=3,  # gene_size harus == cluster_size
                          num_features=5, range_start=0, range_end=10, fitness_func=fitness_function_kmeans,
                          fitness_sort='asc', df=df, headers=headers)
    ga.run()
