# -*- coding: utf-8 -*-
#Inatalled keras and tensorflow using sudo pip install command
# code cited from https://github.com/hundredblocks/concrete_NLP_tutorial/blob/master/NLP_notebook.ipynb
# program to clean the file and process it

# Logic and flow
# There are two inputs: reviews xlsx and tweets xlsx for each application
# Process each file, retain the date, author and tweet/review information for each file
# get rid of unwanted characters
# create a new file with this data in a single file

import keras
import nltk
import pandas as pd
import numpy as np
import re
import codecs
import csv
import sys
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_similarity_score

name = "kendallkylie"
Reviewfile_name =  name + "_reviews.csv"
cleanReviewfile_name = name+"_reviews_cols_clean.csv"

Tweetfile_name =  name+"_tweets.csv"
cleanTweetfile_name =  name+"_tweets_cols_clean.csv"

TweetBugs = "bug_tweets.csv"
#TweetFeature = "feature_tweets.csv"

ReviewBugs = "bug_reviews.csv"
#ReviewFeature = "feature_reviews.csv"

#few regular expressions to clean up pour data, and save it back to disk for future use
def standardize_text(df, body, author= None):
    df[body] = df[body].str.replace(r"http\S+", "")
    df[body] = df[body].str.replace(r"http", "")
    df[body] = df[body].str.replace(r"@\S+", "")
    df[body] = df[body].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[body] = df[body].str.replace(r"@", "at")
    df[body] = df[body].str.replace(r".", "")
    df[body] = df[body].str.lower()
    
    if author is not None: 
    	df[author] = df[author].str.replace(r"http\S+", "")
    	df[author] = df[author].str.replace(r"http", "")
    	df[author] = df[author].str.replace(r"@\S+", "")
    	df[author] = df[author].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    	df[author] = df[author].str.replace(r"@", "at")
    	df[author] = df[author].str.lower()
    return df


#retain just the characters and get rid of all special chars
def sanitize_characters():    
	infile = codecs.open(cleanReviewfile_name, "r",encoding='utf-8', errors='replace')
	outfile = open(cleanReviewfile_name, "w")
	for line in infile:
		outfile.write(line)
     

#main begins here
#read the file and read the required data from file
def readReviewFile():
	ft = open(Reviewfile_name, "rb")
	reader = csv.DictReader(ft)
	#print reader.fieldnames
	reviewsdict = []
	for row in reader:
		Author = row['Author']
		Date = row['Date']
		body = row['body']
    	#mydict = {(Author,Date,body)  for in rows}
    	#print(mydict)
		#print row['Date'], row['body']
		dict1 = {'Author':Author, 'Date':Date, 'body':body}
		reviewsdict.append(dict1)
	ft.close() #close the file 
	Reviewinformation = pd.DataFrame(data=reviewsdict)	
	Reviewinformation = standardize_text(Reviewinformation, "body", "Author") #clean the data
	Reviewinformation.to_csv(cleanReviewfile_name) #hold on for now
	#print Reviewinformation
	return Reviewinformation


#function to prep Tweet file for analysis
def readTweetFile():
	df = pd.read_csv(Tweetfile_name, header=None, usecols=[1,2]) #tweet and the timestamp
	df[2] = pd.to_datetime(df[2])
	df[2] = df[2].dt.date
 	df.columns = ['tweet', 'Date'] # add column names here
 	df = standardize_text(df, 'tweet') # clean it
 	df.to_csv(cleanTweetfile_name) #hold on for now
 	TweetInformation = df
 	#print TweetInformation
 	return TweetInformation
	#return mydict

def mergeDataframe(df1, df2):
	#df = pd.merge(df1, df2, on=["Date","tweetDate"])
	df = pd.merge(df1, df2, on='Date', how='outer', indicator=True) #how='outer', on=["Date","tweetDate"])
	df.to_csv("out.csv") #hold on for now
	print df.groupby("Date").count()

	return df
	pass

#function to plot the distribution/histogram of the tweets/perday and reviews/day
def plotGraphs(tweetDict, reviewDict):
	#tweetDict.groupby(["Date"]).count()['tweet'].hist(bins=10)
	#plt.show()
	#reviewDict.groupby(["Date"]).count()['body'].plot.bar(stacked=True);
	#plt.show()		
	
	# Set up the axes and figure
	#fig, ax = plt.subplots()
	dataReview = []
	groupedReview= reviewDict.groupby(['Date'])
	for date, group in groupedReview:
		#print date, len(group)
		dataReview.append({'date':date, 'count':len(group)})
		#df.append([date, len(group)])
	df = pd.DataFrame(data=dataReview)
	df.to_csv('outReview.csv')
	print df

	dataTweet=[]
	grouped= tweetDict.groupby(['Date'])
	for date, group in grouped:
		#print date, len(group)
		dataTweet.append({'date':date, 'count':len(group)})
	df = pd.DataFrame(data=dataTweet)
	print df
	df.to_csv('outTweet.csv')

	#plt.bar(range(len(dataReview)), dataReview.values(), align="center")
	#plt.xticks(range(len(dataReview)), list(dataReview.keys()))
	#plt.show()
	#temp = {'data1':dataReview, 'data2': dataTweet}
	#plt.bar(range(len(temp)), list(D.values()), align='center')
	#plt.xticks(range(len(D)), list(D.keys()))

	#width = np.diff(tweetDict.groupby(["Date"]).count()['tweet']).min()
	#ax.bar(dates, counts, align='center', width=width)
	#displays the count of reviews for each day
	#print reviewDict.groupby("Date").count()

	#print tweetDict.groupby("Date").count()
	#gp = tweetDict.Date.unique()
	#df2 = tweetDict.groupby(["Date"]).count()
	#fig, ax = plt.subplots(figsize=(15,7))
	#plt.figure()

#read the already classified bugs and features with class labels and lookup using similarity measure and add to tweetDict dataframe
#for every entry in the bugs file and feature file
#	check similarity more than .8 and if mathed add the class lable to tweetDict
def readTwitterBugsFeature(tweetDict):
	#for bugs
	bugs = pd.read_csv(TweetBugs, header=None, usecols=[0,1]) #tweet and the class label for bugs
	#print bugs[0]
	bugs = standardize_text(bugs, 0)
	print len(bugs.index)
	tweetDict.loc[tweetDict['tweet'].isin(bugs[0]), 'Class'] = 1 #bug:1, review:2, rest:0  
	#print tweetDict.groupby(['Class']).count()
	#tweetDict.to_csv("out.csv") 
	
	tweetDict.to_csv("tweet_date_class.csv") 
	return tweetDict

def readReviewBugsFeature(reviewDict):
	bugs = pd.read_csv(ReviewBugs, header=None, usecols=[0,1]) #tweet and the class label for bugs
	#print bugs[0]
	bugs = standardize_text(bugs, 0)
	print len(bugs.index)
	reviewDict.loc[reviewDict['body'].isin(bugs[0]), 'Class'] = 1 #bug:1, review:2, rest:0  
	print reviewDict.groupby(['Class']).count()
	#tweetDict.to_csv("out.csv") 
	
	reviewDict.to_csv("review_date_class.csv") 
	return reviewDict
	

def findIntersection(dfReview, dfTweet):
    dfReview.loc[reviewDict['body'].isin(dfTweet['tweet']), 'flag'] = True 
    #dfTweet.reshape((9999,1))
    #dfReview.reshape((9999,1))
    print(jaccard_similarity_score(dfReview['body'], dfTweet['tweet']))
    #print dfReview.groupby(['flag']).count()
    #dfReview.to_csv("out.csv") 
	
	
	
reviewDict = readReviewFile()
tweetDict = readTweetFile()
#plotGraphs(tweetDict, reviewDict)
MasterTweetDict = readTwitterBugsFeature(tweetDict) # tweet:date:class
MasterReviewDict = readReviewBugsFeature(reviewDict) #review:date:class
#findIntersection(MasterReviewDict, MasterTweetDict) #check overlapping 'tweet' and 'body' column 


#print g.groups.keys()
#print df2["tweet"]
#tweetDict['groups'] = tweetDict.groupby("Date")
#tweetDict['count'] = tweetDict.groupby("Date").count()
#d = {'Date': [groups], 'count': [count]}
#print d
#df = pd.DataFrame(data=d)
#df.to_csv("out.csv") #hold on for now
#print df
#df.plot.bar(x='Date', y='count');

#tweetDict = tweetDict.pivot(columns='Date', values='tweet')

#df2 = pd.DataFrame(count, groups])
#tweetDict.plot.bar(x='groups', y='count');
#plt.figure(); 
#tweetDict.plot();

#tweetDict = tweetDict.pivot(columns='Date', values='tweet')
#print tweetDict.groupby("Date").count()
#tweetDict.to_csv("out.csv") #hold on for now
#print tweetDict
	
#mergeDataframe(reviewDict, tweetDict)


#Tokenizing sentences to a list of separate words and analyze further
#tokenizer = RegexpTokenizer(r'\w+')
#ReviewInformation["tokens"] = ReviewInformation["body"].apply(tokenizer.tokenize)
#print information.head()

#-----------Now process tweets and reviews separately and generate LDA for that day--------------#
#tweetforLDA = tweetDict.pivot(columns='Date', values='tweet')
#reviewforLDA = reviewsdict .pivot(columns='Date', values='body')


