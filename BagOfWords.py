import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import  LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

import DataPrep
import FeatureSelection
from ModelStatistics import model_statistics
import ModelFiles

train_news = DataPrep.train_news
test_news = DataPrep.test_news

countV = FeatureSelection.countV

nb_model_file = ModelFiles.nb_model_file_cv
logR_model_file = ModelFiles.logR_model_file_cv
svm_model_file = ModelFiles.svm_model_file_cv
sgd_model_file = ModelFiles.sgd_model_file_cv
rf_model_file = ModelFiles.rf_model_file_cv

# Using Bag of words technique
print()
print('------------------------------BAG Of WORDS------------------------------')
print()

# Building classifier using Naive Bayes
nb_pipeline = Pipeline(
    [
        ('nbCV', countV),
        ('nb_clf', MultinomialNB())
    ]
)
nb_pipeline = model_statistics(nb_pipeline, 'Naive Bayes Classifier')
pickle.dump(nb_pipeline, open(nb_model_file, 'wb'))

# Building classifier using Logistic Regression
logR_pipeline = Pipeline(
    [
        ('logRCV', countV),
        ('logR_clf', LogisticRegression(max_iter = 10000))
    ]
)
logR_pipeline = model_statistics(logR_pipeline, 'Logistic Regression Classifier')
pickle.dump(logR_pipeline, open(logR_model_file, 'wb'))

# Building classifier using Linear SVM
svm_pipeline = Pipeline(
    [
        ('svmCV', countV),
        ('svm_clf', svm.LinearSVC(max_iter = 10000))
    ]
)
svm_pipeline = model_statistics(svm_pipeline, 'Linear SVM Classifier')
pickle.dump(svm_pipeline, open(svm_model_file, 'wb'))

# Building classifier using Stochastic Gradient Descent on hinge loss
sgd_pipeline = Pipeline(
    [
        ('sgdCV', countV),
        ('sgd_clf', SGDClassifier(loss = 'hinge', penalty='l2', alpha = 1e-3, n_iter_no_change=5))
    ]
)
sgd_pipeline = model_statistics(sgd_pipeline, 'SGD Classifier')
pickle.dump(sgd_pipeline, open(sgd_model_file, 'wb'))

# Building classifier using Random Forest
rf_pipeline = Pipeline(
    [
        ('rfCV', countV),
        ('rf_clf', RandomForestClassifier(n_estimators = 200, n_jobs = 3))
    ]
)
rf_pipeline = model_statistics(rf_pipeline, 'Random Forest Classifier')
pickle.dump(rf_pipeline, open(rf_model_file, 'wb'))
