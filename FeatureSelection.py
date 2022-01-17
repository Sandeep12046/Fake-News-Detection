"""
Before we can train an algorithm to classify fake news labels, we need to extract features from it. It means reducing the mass of unstructured data into some uniform set of attributes that an algorithm can understand. For fake news detection, it could be word counts (bag of words) and use of n-grams. 
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk.corpus 

import DataPrep

train_news = DataPrep.train_news

# Bag of Words Technique 
# Creating Feature Vector - Document Term Matrix
countV = CountVectorizer()

train_count = countV.fit_transform(train_news['Statement'].values)
#print('Train Count:')
#print(train_count)
#print()

# Print Training Doc Term Matrix
#we have matrix of size of (10240, 12196) by calling below
def get_countVectorizer_stats():
    
    # Vocabulary size
    print('Train Count Size:')
    print(train_count.shape)
    print()

    # Check vocabulary using below command
    print('Vocabulary From Dataset:')
    print(countV.vocabulary_)
    print()

    # Get feature names
    print('Feature Names:')
    print(countV.get_feature_names()[:25])
    print()

#get_countVectorizer_stats()

# Create TF-IDF Frequency Features
tfidfV = TfidfTransformer()
train_tfidf = tfidfV.fit_transform(train_count)

def get_tfidf_stats():

    print('TF-IDF Size:')
    print(train_tfidf.shape)
    print()

    # Train Data Feature Names 
    print('TF-IDF Feature Names:')
    print(train_tfidf[:10])
    print()

#get_tfidf_stats()

# Bag of Words - N-Grams

tfidf_ngram = TfidfVectorizer(stop_words = 'english', ngram_range = (1, 4), use_idf = True, smooth_idf = True)

# POS Tagging
tagged_sentences = nltk.corpus.treebank.tagged_sents()
#print(tagged_sentences)
cutoff = int(.75 * len(tagged_sentences))
training_sentences = train_news['Statement']
#print('Training sentences:')
#print(training_sentences)

# Training POS Tagger based on words
def features(sentence, index):
    # Sentence: [w1, w2, ...], index: the index of the word 
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }
    
#print(features(['It will take years to build a wall around USA'], 0)) 
