'''
Program to classify the data into similar and dissimilar bugs as well as features
refer the website: http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
refer http://scikit-learn.org/stable/modules/svm.html#svm-regression
'''
#from sklearn import svm
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import numpy as np
import random
import sys
import pandas as pd
import json
from collections import Counter
sys.path.append('/home/gouri/Ucalgary/672/GouriLibraries')

import LoadJson as ljson
from pprint import pprint
from nltk.stem import WordNetLemmatizer 
from nltk import word_tokenize 
import datetime
from nltk.stem.snowball import SnowballStemmer
from textblob import TextBlob, Word
from sklearn import svm
import numpy as np
from sklearn.model_selection import KFold

# define a function that accepts text and returns a list of lemmas
def split_into_lemmas(text):
    text = str(text)
    words = TextBlob(text).words
    #print "in here"
    #print [word.lemmatize() for word in words]
    #print "\n------------------------\n"
    return [word.lemmatize() for word in words]

fileIs = "/home/gouri/Ucalgary/672/RQ1/mergedSimilarityDropboxGoogleCastLinkedIn.csv"

#fileIs = "/home/gouri/Ucalgary/672/RQ1/SimilarityLinkedIn.csv"
#fileIs = "/home/gouri/Ucalgary/672/RQ1/SimilarityGoogleCast.csv"
#fileIs = "/home/gouri/Ucalgary/672/RQ1/SimilarityDropbox.csv"

def dataAnalysis(df):
	print len(df), list(df)
	print len(df.loc[df[df.columns[4]] == 1])
	print len(df.loc[df[df.columns[4]] == 0])
	positiveSamples = df.loc[df[df.columns[4]] == 1]

	#df['col'] = pd.to_datetime(df['col'])

	tweetDate = positiveSamples[positiveSamples.columns[1]]
	tweetDate = pd.to_datetime(tweetDate)
	reviewDate = positiveSamples[positiveSamples.columns[3]]
	reviewDate = pd.to_datetime(reviewDate)
	print "Total samples :", len(positiveSamples)
 	print "Where Tweets are earlier", len(positiveSamples[tweetDate >= reviewDate])
 	negativeSamples = df.loc[df[df.columns[4]] == 0]
	frames = [positiveSamples, negativeSamples.sample(920)]
	result = pd.concat(frames)
	return result


def LemmatizedCompute(train, test, train_lables, test_lables, typeOfcompute):
	count_vect = CountVectorizer(tokenizer=lambda doc: doc, analyzer=split_into_lemmas, lowercase=False, stop_words='english')
	X_train_counts = count_vect.fit_transform(train)
	
	#tokenize
	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

	#classify
	#clf = MultinomialNB(fit_prior=False).fit(X_train_tfidf, train_lables) 
	#clf = SGDClassifier().fit(X_train_tfidf, train_lables)
	clf = typeOfcompute.fit(X_train_tfidf, train_lables)
	X_new_counts = count_vect.transform(test)
	X_new_tfidf = tfidf_transformer.transform(X_new_counts)

	predicted = clf.predict(X_new_tfidf)
	print "Lemmatized:",np.mean(predicted == test_lables)
	#print type(test)

	#scikit-learn further provides utilities for more detailed performance analysis of the results:
	print metrics.classification_report(test_lables, predicted)
	print metrics.confusion_matrix(test_lables, predicted)
	#print metrics.mean_squared_error(test_lables, predicted), "\n"

	pass

def NonLemmatizedCompute(train, test, train_lables, test_lables, typeOfcompute):

	text_clf = Pipeline([('vect', CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)),
						('tfidf', TfidfTransformer()),
						('clf', typeOfcompute)])
	text_clf.fit(train, train_lables)  
	predicted = text_clf.predict(test)
	print "Non Lemmatized:", np.mean(predicted == test_lables)
	#print type(test)

	#scikit-learn further provides utilities for more detailed performance analysis of the results:
	#print metrics.classification_report(test_lables, predicted)
	#print metrics.confusion_matrix(test_lables, predicted)
	#print metrics.mean_squared_error(test_lables, predicted)
	pass	

def Naive_bayesCompute():
	pass


def process():
	data =  ljson.readCSVdf(fileIs, "#")
	#data =  ljson.readCSVdf(fileIs, ",")
	data= dataAnalysis(data)
	data = data.sample(frac=1).reset_index(drop=True)
	data = data.values #converted to 'numpy.ndarray'
	#print data[:,[4]]
	#data = data[:, [0,2,4]]
	processData = np.array(data[:, [0,2,4]]) #tweet, review and similar/dissimilar class]
	#shuffle the data

	#print processData
	kf = KFold(n_splits=10)
	i = 1
	for train_index, test_index in kf.split(processData):
	    #print trainD#, testD
	    print "Fold:", i
	    i = i+1

	    trainData, testData = np.array(processData[train_index]), np.array(processData[test_index])

	    #print trainData

	    train, train_lables = trainData[:, :-1], trainData[:, -1] 
	    train_lables = train_lables.astype('int')

	    #print train, train_lables

	    #test= testData
	    test, test_lables= testData[:, :-1], testData[:, -1]
	    test_lables=test_lables.astype('int')
	    
	    #LemmatizedCompute(train, test, train_lables, test_lables, SGDClassifier())
	    LemmatizedCompute(train, test, train_lables, test_lables, MultinomialNB(fit_prior=False))
	    #NonLemmatizedCompute(train, test, train_lables, test_lables, SGDClassifier())
	    #NonLemmatizedCompute(train, test, train_lables, test_lables, MultinomialNB(fit_prior=False))
	pass	
		

process()
