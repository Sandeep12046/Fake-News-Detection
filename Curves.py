import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

import DataPrep

train_news = DataPrep.train_news
test_news = DataPrep.test_news

# Plotting Learing Curve
def plot_learning_curve(pipeline, title):
    size = 10
    cvg = KFold(size, shuffle = True)
    
    X = train_news['Statement']
    y = train_news['Label']
    
    pipeline.fit(X, y)
    
    train_sizes, train_scores, test_scores = learning_curve(pipeline, X, y, n_jobs = -1, cv = cvg, verbose = 0)
       
    train_scores_mean = np.mean(train_scores, axis = 1)
    train_scores_std = np.std(train_scores, axis = 1)
    test_scores_mean = np.mean(test_scores, axis = 1)
    test_scores_std = np.std(test_scores, axis = 1)
     
    plt.figure()
    plt.title(title)
    plt.legend(loc = "best")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.gca().invert_yaxis()
    plt.grid()

    # plot the std deviation as a transparent range at each training set size
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha = 0.1, color = "red")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha = 0.1, color = "green")

    # plot the average training and test score lines at each training set size
    plt.plot(train_sizes, train_scores_mean, 'o-', color = "red", label = "Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color = "green", label = "Cross-validation score")

    # sizes the window for readability and displays the plot
    # shows error from 0 to 1.1
    plt.ylim(-0.1, 1.1)
    plt.show()

# Plotting Precision-Recall Curve
def plot_PR_curve(classifier):

    precision, recall, thresholds = precision_recall_curve(test_news['Label'], classifier)
    average_precision = average_precision_score(test_news['Label'], classifier)

    plt.figure()
    plt.step(recall, precision, color='blue', alpha = 0.2, where = 'post')
    plt.fill_between(recall, precision, step = 'post', alpha = 0.2, color = 'blue')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Naive Bayes Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.show()
