import pandas as pd
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    headers = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    df = pd.read_csv('dataset/car.data', header=None, names=headers)
    classes = []

    for header in headers:
        le = LabelEncoder()
        df[header+'_le'] = le.fit_transform(df[header])
        if header is 'class':
            classes = le.classes_

    df = df.drop(headers, axis='columns')

    train_x, test_x, train_y, test_y = train_test_split(df[[item+'_le' for item in headers[:-1]]],
                                                        df['class_le'], random_state=None, test_size=0.1)
    # print(train_x, test_x, train_y, test_y)

    model = KNeighborsClassifier(n_neighbors=10)
    model.fit(train_x, train_y)

    predictions = model.predict(test_x)

    print(classification_report(test_y, predictions, target_names=classes))
