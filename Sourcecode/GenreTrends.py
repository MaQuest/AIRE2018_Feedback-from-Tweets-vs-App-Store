'''
Program to process the paid tweet data and the paid review data files 
group them based on the Genre 
Compute the sum of tweets/review per day 
plot the data to visualize which kind of data is tweeted and the comparison
'''

import json
import pandas as pd
from pprint import pprint
import sys
import re
sys.path.append('/home/gouri/Ucalgary/672/GouriLibraries')
import LoadJson as ljson


def computeSum(df, fname):
	df = df.groupby(['GenreId']).sum()
	#df.to_csv("output"+fname)
	pass

def preProcess(df):
	df_new = df.drop(['GenreId', 'app'], axis=1)
	#df_new['max_value']=df_new.values.max(1)
	#print df_new['max_value']
	print "headers are \n", list(df_new)
	df_new.apply(lambda row: row.replace(0, max(row)), axis=0)
	df_new = df_new.sub(df_new.values.max(1), axis=0)
	df_new['GenreId'] = df['GenreId']
	df_new['app'] = df['app']
	return df_new

def readyFile(fileIs):
	df_ReviewData = ljson.readCSVdf(fileIs)
	df_new = df_ReviewData.drop(['GenreId', 'app'], axis=1)
	df_temp = df_ReviewData.drop(['GenreId', 'app', 'dummy'], axis=1)
	#df_new = df_new.sub(df_new.values.max(1), axis=0)
	column = list(df_temp)
	print "Review headers are \n", list(df_new)
	for index, row in df_new.iterrows():
		print row['dummy']
		for i in column:
			if row[i]!=0:
				row[i] = row['dummy'] - row[i]


	df_new['GenreId'] = df_ReviewData['GenreId']
	df_new['app'] = df_ReviewData['app']
	df_new.to_csv("Final"+fileIs)
	

	pass

def main():
	listOfFiles = ['FinalTopPaid.csvNEW.csv', 'TopPaidTweets.csvNEW.csv'] # Files with Genre and App name infomration
	
	fileIs = listOfFiles[0] #read the review data
	df_ReviewData = ljson.readCSVdf(fileIs)
	#remove duplicates
	df_ReviewData = df_ReviewData.drop_duplicates(subset=['app','GenreId'], keep='first')
	#f_ReviewData = preProcess(df_ReviewData)

	fileIs = listOfFiles[1] #read the tweets data
	df_TweetsData = ljson.readCSVdf(fileIs)
	#remove duplicates
	df_TweetsData = df_TweetsData.drop_duplicates(subset=['app','GenreId'], keep='first')

    #print the genre and the count under that Genre
	#print "Review headers are \n", list(df_ReviewData)
	print df_ReviewData.groupby('GenreId').GenreId.count().nlargest(20)
	#print "Twitter headers are \n", list(df_TweetsData)
	#print df_TweetsData.groupby('GenreId').GenreId.count().nlargest(20)

	#pass the df onto function to sum the rows which are not -ve for cumulative over a day
	df_SumReviewData = computeSum(df_ReviewData, listOfFiles[0] )
	#df_SumTweetData = computeSum(df_TweetsData, listOfFiles[1])

main()
#eadyFile('TopPaid.csvNEW.csv')
