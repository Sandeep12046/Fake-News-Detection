import pickle
import re

from FromWeb import summarize
import DataPrep
import ModelFiles

test_news = DataPrep.test_news

# All Model Files
# Bag Of Words Models
nb_model_file_cv = ModelFiles.nb_model_file_cv
logR_model_file_cv = ModelFiles.logR_model_file_cv
svm_model_file_cv = ModelFiles.svm_model_file_cv
sgd_model_file_cv = ModelFiles.sgd_model_file_cv
rf_model_file_cv = ModelFiles.rf_model_file_cv

# NGram Models
nb_model_file_ngram = ModelFiles.nb_model_file_ngram
logR_model_file_ngram = ModelFiles.logR_model_file_ngram
svm_model_file_ngram = ModelFiles.svm_model_file_ngram
sgd_model_file_ngram = ModelFiles.sgd_model_file_ngram
rf_model_file_ngram = ModelFiles.rf_model_file_ngram

# Final Model [Logistic Regression - NGram]
logR_model_file_final = ModelFiles.logR_model_file_final

# Load all Models
# Bag Of Words Models
nb_load_model_cv = pickle.load(open(nb_model_file_cv, 'rb'))
logR_load_model_cv = pickle.load(open(logR_model_file_cv, 'rb'))
svm_load_model_cv = pickle.load(open(svm_model_file_cv, 'rb'))
sgd_load_model_cv = pickle.load(open(sgd_model_file_cv, 'rb'))
rf_load_model_cv = pickle.load(open(rf_model_file_cv, 'rb'))

# NGram Models
nb_load_model_ngram = pickle.load(open(nb_model_file_ngram, 'rb'))
logR_load_model_ngram = pickle.load(open(logR_model_file_ngram, 'rb'))
svm_load_model_ngram = pickle.load(open(svm_model_file_ngram, 'rb'))
sgd_load_model_ngram = pickle.load(open(sgd_model_file_ngram, 'rb'))
rf_load_model_ngram = pickle.load(open(rf_model_file_ngram, 'rb'))

# Final Model
logR_load_model_final = pickle.load(open(logR_model_file_final, 'rb'))

def prediction_summary(predictions, start, end):
    true_count = 0
    for i in range(start, end):
        if str(predictions[i]) == 'True':
            true_count += 1
    truth_probability = true_count / 5
    return truth_probability

# Voting Ensemble
def ensemble(input_text):
   
    predictions = []

    nb_prediction_cv = nb_load_model_cv.predict([input_text])[0]
    predictions.append(nb_prediction_cv)

    logR_prediction_cv = logR_load_model_cv.predict([input_text])[0]
    predictions.append(logR_prediction_cv)
    
    svm_prediction_cv = svm_load_model_cv.predict([input_text])[0]
    predictions.append(svm_prediction_cv)
    
    sgd_prediction_cv = sgd_load_model_cv.predict([input_text])[0]
    predictions.append(sgd_prediction_cv)
    
    rf_prediction_cv = rf_load_model_cv.predict([input_text])[0]
    predictions.append(rf_prediction_cv)
    
    nb_prediction_ngram = nb_load_model_ngram.predict([input_text])[0]
    predictions.append(nb_prediction_ngram)

    logR_prediction_ngram = logR_load_model_ngram.predict([input_text])[0]
    predictions.append(logR_prediction_ngram)
    
    svm_prediction_ngram = svm_load_model_ngram.predict([input_text])[0]
    predictions.append(svm_prediction_ngram)
    
    sgd_prediction_ngram = sgd_load_model_ngram.predict([input_text])[0]
    predictions.append(sgd_prediction_ngram)
    
    rf_prediction_ngram = rf_load_model_ngram.predict([input_text])[0]
    predictions.append(rf_prediction_ngram)
    
    logR_prediction_final = logR_load_model_final.predict([input_text])[0]
    predictions.append(logR_prediction_final)

    truth_probability_final = 1.0 if str(logR_prediction_final) == 'True' else 0.0

    prediction_cv = prediction_summary(predictions, 0, 5)
    prediction_ngram = prediction_summary(predictions, 5, 10)
    prediction_final = truth_probability_final

    prediction_weighted = 0.2 * prediction_cv + 0.4 * prediction_ngram + 0.4 * prediction_final

    result = True if prediction_weighted >= 0.5 else False

    return [result, round(prediction_weighted, 2)]

# Function to run for prediction
def predict_news(input_text):    
    if (re.search('^http', input_text)):
        input_text = summarize(input_text)
    result = ensemble(input_text)
    return result

# Get Accuracy of Voting Ensemble
def ensemble_accuracy():
    correct_prediction = 0
    test_news_size = test_news.size // 2
    for i in range(test_news_size):
        statement = test_news.iloc[i][0]
        label = test_news.iloc[i][1]
        predicted = ensemble(statement)[0]
        if str(predicted) == str(label):
            correct_prediction += 1
        #print(i, label, predicted, 'Correct' if str(label) == str(predicted) else 'Wrong')
    #print('Accuracy:', correct_prediction / test_news_size)

#ensemble_accuracy()
