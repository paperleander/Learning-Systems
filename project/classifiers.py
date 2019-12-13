import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

x = []
y = []
encode_input = {'b': [1, 0, 0],
                'x': [0, 1, 0],
                'o': [0, 0, 1]}
encode_output = {'win': 0,
                 'draw': 1,
                 'loss': 2}

with open('data/connect-4.data') as f:
    for line in f.readlines():
        line_i = line.strip().split(',')
        mapped_line = [encode_input[i] if i in 'bxo' else encode_output[i] for i in line_i]
        x.append(mapped_line[:42])
        y.append(mapped_line[42])

x = np.array(x)
y = np.array(y)
x = x.reshape(x.shape[0], -1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)


class GridSearchHelper:

    def __init__(self, models_, params_):
        self.models = models_
        self.params = params_
        self.keys = models.keys()
        self.searches = {}

    def run(self, X, y):
        for key in self.keys:
            print("Running", key)
            model = self.models[key]
            params = self.params[key]
            search = GridSearchCV(model, params, cv=5)
            search.fit(X, y)
            self.searches[key] = search

    def score(self):
        frames = []
        for name, search in self.searches.items():
            frame = pd.DataFrame(search.cv_results_)
            frame = frame.filter(regex='^(?!.*param_).*$')
            frame['estimator'] = len(frame)*[name]
            frames.append(frame)
        df = pd.concat(frames)

        df = df.sort_values(['mean_test_score'], ascending=False)
        df = df.reset_index()
        df = df.drop(['rank_test_score', 'index'], 1)

        columns = df.columns.tolist()
        columns.remove('estimator')
        columns = ['estimator'] + columns
        df = df[columns]
        return df


models = {
    'DecisionTree': DecisionTreeClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier()
}

parameters = {
    'DecisionTree': {'max_depth': [1, 2, 3]},
    'AdaBoostClassifier': {'n_estimators': [16, 32]}
}

helper = GridSearchHelper(models, parameters)
helper.run(x_train, y_train)
print(helper.score())
