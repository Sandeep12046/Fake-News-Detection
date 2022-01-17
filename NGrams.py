import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import  LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

import DataPrep
import FeatureSelection
from ModelStatistics import model_statistics
import ModelFiles

train_news = DataPrep.train_news
test_news = DataPrep.test_news

tfidf_ngram = FeatureSelection.tfidf_ngram

nb_model_file = ModelFiles.nb_model_file_ngram
logR_model_file = ModelFiles.logR_model_file_ngram
svm_model_file = ModelFiles.svm_model_file_ngram
sgd_model_file = ModelFiles.sgd_model_file_ngram
rf_model_file = ModelFiles.rf_model_file_ngram
logR_model_file_final = ModelFiles.logR_model_file_final

# Using N-grams technique
print()
print('------------------------------N GRAMS------------------------------')
print()

# Building classifier using Naive Bayes
nb_pipeline = Pipeline(
    [
        ('nb_tfidf', tfidf_ngram),
        ('nb_clf', MultinomialNB())
    ]
)
nb_pipeline = model_statistics(nb_pipeline, 'Naive Bayes Classifier')
pickle.dump(nb_pipeline, open(nb_model_file, 'wb'))

# Building classifier using Logistic Regression
logR_pipeline = Pipeline(
    [
        ('logR_tfidf', tfidf_ngram),
        ('logR_clf', LogisticRegression(max_iter = 10000, dual = False, penalty = 'l2', C = 1))
    ]
)
logR_pipeline = model_statistics(logR_pipeline, 'Logistic Regression Classifier')
pickle.dump(logR_pipeline, open(logR_model_file, 'wb'))

# Building classifier using Linear SVM
svm_pipeline = Pipeline(
    [
        ('svm_tfidf', tfidf_ngram),
        ('svm_clf', svm.LinearSVC(max_iter = 10000))
    ]
)
svm_pipeline = model_statistics(svm_pipeline, 'Linear SVM Classifier')
svm_model_file = './classifiers/svm_model_ngram.sav'
pickle.dump(svm_pipeline, open(svm_model_file, 'wb'))

# Building classifier using Stochastic Gradient Descent on hinge loss
sgd_pipeline = Pipeline(
    [
         ('sgd_tfidf', tfidf_ngram),
         ('sgd_clf', SGDClassifier(loss = 'hinge', penalty = 'l2', alpha = 1e-3, n_iter_no_change = 5))
    ]
)
sgd_pipeline = model_statistics(sgd_pipeline, 'SGD Classifier')
pickle.dump(sgd_pipeline, open(sgd_model_file, 'wb'))

# Building classifier using Random Forest
rf_pipeline = Pipeline(
    [
        ('rf_tfidf', tfidf_ngram),
        ('rf_clf', RandomForestClassifier(n_estimators = 300, n_jobs = 3))
    ]
)
rf_pipeline = model_statistics(rf_pipeline, 'Random Forest Classifier')
pickle.dump(rf_pipeline, open(rf_model_file, 'wb'))
"""
# Grid Search Parameter Optimization

# Random Forest Classifier
parameters = {
    'rf_tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)],
    'rf_tfidf__use_idf': (True, False),
    'rf_clf__max_depth': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
}

gs_clf = GridSearchCV(rf_pipeline, parameters, n_jobs = -1)
gs_clf = gs_clf.fit(train_news['Statement'][:10000], train_news['Label'][:10000])

#print(gs_clf.best_score_)
#print(gs_clf.best_params_)
#print(gs_clf.cv_results_)

# Logistic Regression Classifier
parameters = {
    'logR_tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)],
    'logR_tfidf__use_idf': (True, False),
    'logR_tfidf__smooth_idf': (True, False)
}

gs_clf = GridSearchCV(logR_pipeline, parameters, n_jobs = -1)
gs_clf = gs_clf.fit(train_news['Statement'][:10000], train_news['Label'][:10000])

#print(gs_clf.best_score_)
#print(gs_clf.best_params_)
#gs_clf.cv_results_
"""
# Making final model with best parameter found during Grid Search
logR_pipeline_final = Pipeline(
    [
        ('logR_tfidf', TfidfVectorizer(stop_words = 'english', ngram_range = (1, 5), use_idf = True, smooth_idf = False)),
        ('logR_clf', LogisticRegression(max_iter = 10000, dual = False, penalty = 'l2', C = 1))
    ]
)
logR_pipeline_final = model_statistics(logR_pipeline_final, 'Logistic Regression Classifier[Fit On Best Grid Search Parameter]')
pickle.dump(logR_pipeline_final, open(logR_model_file_final, 'wb'))
