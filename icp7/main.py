from sklearn.datasets import fetch_20newsgroups
import requests
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import nltk as nl
from bs4 import BeautifulSoup
import urllib.request
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from nltk.chunk import conlltags2tree, tree2conlltags
#exercise 1
# twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
#
# tfidf_Vect = TfidfVectorizer(ngram_range=(1,2), stop_words='english') ##question 1 changes
# X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
# #print(tfidf_Vect.vocabulary_)
# clf = SVC() #change to SVC
# clf.fit(X_train_tfidf, twenty_train.target)
#
# twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
# X_test_tfidf = tfidf_Vect.transform(twenty_test.data)
#
# predicted = clf.predict(X_test_tfidf)
#
# score = metrics.accuracy_score(twenty_test.target, predicted)
# print(score)

#exercise 2
nl.download('punkt')
nl.download('averaged_perceptron_tagger')
url = "https://en.wikipedia.org/wiki/Google"
file = open("file.txt", "w")
html = urllib.request.urlopen(url).read()
soup = BeautifulSoup(html, "html.parser")
text = soup.text
text_tokens = nl.word_tokenize(text)
file.write(text_tokens)
print(text_tokens)
pos = nl.pos_tag(text_tokens)
print(pos)
pStemmer = PorterStemmer()
stem = pStemmer.stem(text_tokens)
print(stem)
lemmatizer = WordNetLemmatizer()
lemmatize = lemmatizer(text_tokens)
print(lemmatize)
trigrams = ngrams(text_tokens,3)
print(trigrams)
NamedEntities = nl.ne_chunk(pos)
print(NamedEntities)











