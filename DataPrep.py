"""
This file contains all the pre-processing functions needed to process all input documents and texts.
First we read the train, test and validation data files then performed some pre-processing like tokenizing, stemming etc.
There is some exploratory data analysis performed like response variable distribution and data quality checks like null or missing values etc.
"""

# Import all packages required
import pandas as pd
import nltk
from nltk.stem import SnowballStemmer
import seaborn as sb
import matplotlib.pyplot as plt

from ExtractData import convert_dataset

#convert_dataset('./liar_dataset/train.tsv', r'./dataset/train.csv')
#convert_dataset('./liar_dataset/test.tsv', r'./dataset/test.csv')
#convert_dataset('./liar_dataset/valid.tsv', r'./dataset/valid.csv')

# Dataset
train_filename = "./dataset/train.csv"
test_filename = "./dataset/test.csv"
valid_filename = "./dataset/valid.csv"

# Read dataset
train_news = pd.read_csv(train_filename)
test_news = pd.read_csv(test_filename)
valid_news = pd.read_csv(valid_filename)

# Observing dataset
def data_obs():
    print("Training dataset size:", train_news.shape)
    print(train_news.head(10))
    print()

    print("Testing dataset size:", test_news.shape)
    print(test_news.head(10))
    print()
    
    print("Validation dataset size:", valid_news.shape)
    print(valid_news.head(10))
    print()

#data_obs()

# Check distribution of classes for prediction
def create_distribution(dataFile): 
    return sb.countplot(x = "Label", data = dataFile, palette = "hls")
    
# Shows that training, test and valid data seems to be fairly evenly distributed between the classes
create_distribution(train_news)
#plt.show()
create_distribution(test_news)
#plt.show()
create_distribution(valid_news)
#plt.show()

# Data integrity check (missing label values)
# None of the datasets contains missing values therefore no cleaning required
def data_qualityCheck():
    
    print("Checking training data qualitites:")
    train_news.isnull().sum()
    train_news.info()
    print("Training data check finished")
    print()

    print("Checking testing data qualitites:")
    test_news.isnull().sum()
    test_news.info()
    print("Testing data check finished")
    print()

    print("Checking validation data qualitites:")
    valid_news.isnull().sum()
    valid_news.info()
    print("Validation data check finished")
    print()

#data_qualityCheck()

# Sample stemming check
eng_stemmer = SnowballStemmer("english")
stopwords = set(nltk.corpus.stopwords.words("english"))
#print("Stopwords in ntlk corpus:")
#print(stopwords)
#print()

# Stemming
def stem_tokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    return stemmed

def process_data(data, exclude_stopword = True, stem = True):
    tokens = [w.lower() for w in data]
    tokens_stemmed = tokens
    tokens_stemmed = stem_tokens(tokens, eng_stemmer)
    tokens_stemmed = [w for w in tokens_stemmed if w not in stopwords ]
    return tokens_stemmed

doc = ['caresses', 'dies', 'mules', 'lied', 'died', 'sized', 'meeting', 'stating', 'seizing', 'itemization', 'reference', 'tokenizing', 'plotted']

#print('Sample stemming:')
#print(process_data(doc))
#print()

# Creating NGrams

# Unigrams 
def create_unigrams(words):
    assert type(words) == list
    return words

#print("Unigrams of passed words:")
#print(create_unigrams(doc))
#print()

# Bigram
def create_bigrams(words):
    assert type(words) == list
    skip = 0
    join_str = " "
    length = len(words)
    if length > 1:
        result = []
        for i in range(length - 1):
            for j in range(1, skip + 2):
                if i + j < length:
                    result.append(join_str.join([words[i], words[i + j]]))
    else:
        result = create_unigram(words)
    return result

#print("Bigrams of passed words:")
#print(create_bigrams(doc))
#print()

# Trigrams
def create_trigrams(words):
    assert type(words) == list
    skip = 0
    join_str = " "
    length = len(words)
    if length > 2:
        result = []
        for i in range(length - 1):
            for j in range(1, skip + 2):
                for k in range(1, skip + 2):
                    if i + j < length and i + j + k < length:
                        result.append(join_str.join([words[i], words[i + j], words[i + j + k]]))
    else:
        #set is as bigram
        result = create_bigrams(words)
    return result

#print("Trigrams of passed words:")
#print(create_trigrams(doc))
#print()
