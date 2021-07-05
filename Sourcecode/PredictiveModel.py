'''
Program to do a line fitting and come up with a predictive model
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

fileIs = "/home/gouri/Ucalgary/672/RQ1/SimilarityDropbox.csv"
#fileIs = "/home/gouri/Ucalgary/672/RQ1/SimilarityGoogleCast.csv"
#fileIs = "/home/gouri/Ucalgary/672/RQ1/SimilarityLinkedIn.csv" #for this it is comma separated
#fileIs = "/home/gouri/Ucalgary/672/RQ1/mergedSimilarityDropboxGoogleCastLinkedIn.csv"
def dataAnalysis(df):
	print len(df)
	print len(df.loc[df[df.columns[4]] == 1]) #total positive samples
	print len(df.loc[df[df.columns[4]] == 0]) #total -ve samples
	positiveSamples = df.loc[df[df.columns[4]] == 1] #read them
	negativeSamples = df.loc[df[df.columns[4]] == 0] #read them
	print "Headers are:", list(positiveSamples)
	listpos, listposDates = positiveSamples[positiveSamples.columns[0]], positiveSamples[positiveSamples.columns[1]]
	listneg = negativeSamples[negativeSamples.columns[0]]
	print "Total +ves: ", len(listpos), "Total -ves: ", len(listneg) 
	listposTweets = set(positiveSamples[positiveSamples.columns[0]])
	listpostReviews = set(positiveSamples[positiveSamples.columns[2]])

	print "Total +ves Tweets: ", len(listposTweets), "Total +ves Review: ", len(listpostReviews) 

	#df['col'] = pd.to_datetime(df['col'])
	#print list(positiveSamples)
	#positiveSamples= positiveSamples.sort(positiveSamples.columns[1],ascending=1)

	tweetDate = positiveSamples[positiveSamples.columns[1]]
	tweetDate = pd.to_datetime(tweetDate)
	reviewDate = positiveSamples[positiveSamples.columns[3]]
	reviewDate = pd.to_datetime(reviewDate)
	print "Where Tweets are earlier", len(positiveSamples[tweetDate > reviewDate])
 	
	#frames = [positiveSamples, negativeSamples.sample(500)]
	#print positiveSamples
	sample1 = positiveSamples[tweetDate > reviewDate]
	sample2 = positiveSamples[tweetDate < reviewDate]
	sample3 = positiveSamples[tweetDate == reviewDate]
	print "Tweets: Unique and faster: ",len(set(sample1[sample1.columns[0]])), len(sample1[sample1.columns[0]]) 
	print "Reviews: Unique and faster: ",len(set(sample2[sample2.columns[2]]))
	print "Appeared at same time : ",len(set(sample3[sample3.columns[2]]))
	#print type(sample1)
	#print "percentage of faster spill", (len(set(sample1[sample1.columns[0]])) + len(set(sample3[sample3.columns[2]])))/(len(set(sample1[sample1.columns[0]])) + len(set(sample3[sample3.columns[2]]))+len(set(sample3[sample3.columns[2]])))

	print "Tweets are earlies", len(listpos)
	print "Reviews are earlies", len(listneg)

	listpos = set(listpos) 
	listneg = set(listneg)
	# Create a new column called df.diff where the value is +ve difference
	# if df.tweet time is greater than review time
	samplePos = positiveSamples #[tweetDate >= reviewDate]
	samplePos[samplePos.columns[1]] = pd.to_datetime(samplePos[samplePos.columns[1]])
	samplePos[samplePos.columns[3]] = pd.to_datetime(samplePos[samplePos.columns[3]])
	
	samplePos['diff'] = np.where(samplePos[samplePos.columns[1]>=samplePos.columns[3]], samplePos[samplePos.columns[1]]- samplePos[samplePos.columns[3]], '-1' )
	#print "Early tweets \n ", set(sample1[sample1.columns[0]]) 
	#print "Same time tweets \n ", set(sample3[sample3.columns[2]])
	#print "Late reviews \n ", set(sample2[sample2.columns[2]]) 
	
	#print len(listpos)
	#result = pd.concat(frames)
	#return result

def process():
	#file1 = "/home/gouri/Ucalgary/672/RQ1/SimilarityDropbox.csv"
	#file2 = "/home/gouri/Ucalgary/672/RQ1/SimilarityGoogleCast.csv"
	data1 =  ljson.readCSVdf(fileIs, "#")
	data1= dataAnalysis(data1)
	pass

process()
