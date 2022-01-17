import numpy as np

import DataPrep
from ConfusionMatrix import build_confusion_matrix
from Curves import plot_learning_curve, plot_PR_curve

train_news = DataPrep.train_news
test_news = DataPrep.test_news

def model_statistics(pipeline, model):
    pipeline.fit(train_news['Statement'], train_news['Label'])
    predicted = pipeline.predict(test_news['Statement'])
    accuracy = np.mean(predicted == test_news['Label'])
    print()
    print(model)
    print('--------------------------------------------------------')
    print('Accuracy:', accuracy)
    #build_confusion_matrix(pipeline, test_news['Label'], predicted)
    #plot_learning_curve(pipeline, model)
    #plot_PR_curve(predicted)
    return pipeline
