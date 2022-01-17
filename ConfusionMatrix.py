import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score, classification_report

import DataPrep

train_news = DataPrep.train_news

# User defined functon for K-Fold cross validation
def build_confusion_matrix(classifier, data, predicted):

    k_fold = KFold(n_splits = 5)
    scores = []
    confusion = np.array([[0, 0], [0, 0]])

    for train_index, test_index in k_fold.split(train_news):
        train_text = train_news.iloc[train_index]['Statement']
        train_y = train_news.iloc[train_index]['Label']

        test_text = train_news.iloc[test_index]['Statement']
        test_y = train_news.iloc[test_index]['Label']

        classifier.fit(train_text, train_y)
        predictions = classifier.predict(test_text)

        confusion += confusion_matrix(test_y, predictions)
        score = f1_score(test_y, predictions)
        scores.append(score)

    print('Total statements classified:', len(train_news))
    print('Score:', sum(scores) / len(scores))
    print('Score length', len(scores))
    print('Confusion matrix:\n', confusion)
    print('Classification Report:')
    print(classification_report(data, predicted))

