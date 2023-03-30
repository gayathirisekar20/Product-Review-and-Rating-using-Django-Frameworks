import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.pipeline import Pipeline
# import string
# from spacy.lang.en.stop_words import STOP_WORDS
import spacy
# from sklearn.svm import LinearSVC
# import pickle 

## Reading training data
# data = pd.read_csv('amazon_data.txt', sep='\t', header=None)
# data.columns = ["Review", "Sentiment"]

# ## Helpers for cleaning
# punctuation = string.punctuation
# stop_words = list(STOP_WORDS)
# nlp = spacy.load("en_core_web_sm")
# numbers = string.digits

# ## Function that cleans input text
# def cleaning_function(input_text):
#     text = nlp(input_text)
#     tokens = []
#     for token in text:
#         temp = token.lemma_.lower()
#         tokens.append(temp)
#     cleaned_tokens = []
#     for token in tokens:
#         if token not in stop_words and token not in punctuation and token not in numbers:
#             cleaned_tokens.append(token)
#     return cleaned_tokens

# ## X, y labels
# X = data["Review"]
# y = data["Sentiment"]

# ## SVC using tfidf (bag of words)
# tfidf = TfidfVectorizer(tokenizer = cleaning_function)
# classifier = LinearSVC()
# SVC_clf = Pipeline([('tfidf', tfidf), ('clf', classifier)])
# SVC_clf.fit(X, y)

# ## Saving trained model
# fname = 'saved_model.pickle'
# with open(fname, 'wb') as f:
#     pickle.dump(SVC_clf, f)


df = pd.read_csv('amazon_data.txt', names=['review', 'sentiment'], sep='\t') 
df.head()

from sklearn.model_selection import train_test_split
reviews = df['review'].values
labels = df['sentiment'].values
reviews_train, reviews_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=1000)

from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import string
punctuations = string.punctuation
parser = English()
stopwords = list(STOP_WORDS)
def spacy_tokenizer(utterance):
    tokens = parser(utterance)
    return [token.lemma_.lower().strip() for token in tokens if token.text.lower().strip() not in stopwords and token.text not in punctuations]
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
#By default, the vectorizer might be created as follows:
#vectorizer = CountVectorizer()
vectorizer.fit(reviews_train)

X_train = vectorizer.transform(reviews_train)
X_test = vectorizer.transform(reviews_test)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
accuracy = classifier.score(X_test, y_test)
print('Accuracy:', accuracy)
new_reviews = ['Old version of python useless', 'Very good effort, but not five stars', 'Clear and concise']
X_new = vectorizer.transform(new_reviews)
classifier.predict(X_new)