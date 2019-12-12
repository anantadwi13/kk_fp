import pandas as pd
from sklearn.metrics import classification_report, precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ga import GeneticAlgorithm


# custom fitness function
def fitness_function_knn(**kwargs) -> float:
    try:
        k = int("".join(x for x in kwargs['chromosome']))
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(kwargs['train_x'], kwargs['train_y'])
        pred_y = model.predict(kwargs['test_x'])
        return precision_score(kwargs['test_y'], pred_y, average='micro')
    except Exception as e:
        # exit(e.args[0])
        return 0


if __name__ == '__main__':

    # Preparing dataset
    headers = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    df = pd.read_csv('dataset/car.data', header=None, names=headers)
    classes = []
    for header in headers:
        le = LabelEncoder()
        df[header + '_le'] = le.fit_transform(df[header])
        if header is 'class':
            classes = le.classes_
    df = df.drop(headers, axis='columns')

    # Splitting dataset
    train_x, test_x, train_y, test_y = train_test_split(df[[item + '_le' for item in headers[:-1]]],
                                                        df['class_le'], random_state=3, test_size=0.1)

    # Running with optimization
    ga = GeneticAlgorithm(population_size=200, max_generation=10, gene_type='digit', genes_size=3,
                          fitness_func=fitness_function_knn, train_x=train_x, train_y=train_y,
                          test_x=test_x, test_y=test_y)
    ga.run()
