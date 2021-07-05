import json
import pandas as pd
from pprint import pprint
import sys
import re
sys.path.append('/home/gouri/Ucalgary/672/GouriLibraries')
import LoadJson as ljson
import matplotlib.pyplot as plt
from matplotlib.pyplot import pie, axis, show
import numpy
#'TOOLS','PERSONALIZATION','GAME_ROLE_PLAYING','GAME_ADVENTURE','GAME_ARCADE','GAME_PUZZLE','GAME_ACTION','HEALTH_AND_FITNESS','GAME_STRATEGY','GAME_SIMULATION','MUSIC_AND_AUDIO','PRODUCTIVITY','EDUCATION','PHOTOGRAPHY','ENTERTAINMENT','GAME_CASUAL','BOOKS_AND_REFERENCE','GAME_BOARD','GAME_SPORTS','COMMUNICATION'
list_Genres = ['GAME_ACTION','HEALTH_AND_FITNESS','GAME_STRATEGY','GAME_SIMULATION','MUSIC_AND_AUDIO','PRODUCTIVITY','EDUCATION','PHOTOGRAPHY','ENTERTAINMENT','GAME_CASUAL','BOOKS_AND_REFERENCE','GAME_BOARD','GAME_SPORTS','COMMUNICATION']
COUNTGen = 10	

def getTopGenres(df):
	print len(df)
	for j in list_Genres:
		df = df[df.GenreId !=j]	
	print len(df)	

	return df	


def loadfiles():
	listOfFiles = ['All20FORPLOTGenrewiseReviewsData.csv', 'All20FORPLOTGenrewiseTWEETSData.csv'] # Files with Genre and App name infomration
	
	fileIs = listOfFiles[0] #read the review data
	df_ReviewData = ljson.readCSVdf(fileIs)

	fileIs = listOfFiles[1] #read the tweets data
	df_TweetsData = ljson.readCSVdf(fileIs)
	
	return (df_ReviewData,df_TweetsData)

def main():
	df_ReviewData,df_TweetsData=loadfiles()
	
	df_ReviewData = getTopGenres(df_ReviewData)
	df_ReviewData = df_ReviewData.set_index('GenreId').T

	#plt.figure(); 
	color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')

	#df_ReviewData.plot.bar(stacked=True)#.box(vert=False, color=color, sym='r+') #.plot();
	plt.show()
	
	df_TweetsData = getTopGenres(df_TweetsData)
	df_TweetsData = df_TweetsData.set_index('GenreId').T
	df_TweetsData.plot.bar(stacked=True)#.box(vert=False, color=color, sym='r+') #.plot();#.plot();
	#plt.show()
	
	pieChart(df_ReviewData)
	

def pieChart(df):
	axis('equal');
	#df= df[['GenreId','Total']]
	df=df.reindex(columns=['GenreId','Total'])
	#df= df[['GenreId','Total']]
	pie(df) #, labels=df.GenreId);
	show()
	pass

def computeCorrelation():
	df_ReviewData, df_TweetsData = loadfiles()
	df_ReviewData = df_ReviewData.sort('GenreId')
	df_TweetsData = df_TweetsData.sort('GenreId')
	print df_ReviewData['GenreId']
	print df_TweetsData['GenreId']
	df_ReviewData = df_ReviewData.drop(['GenreId'], axis=1)
	df_TweetsData =df_TweetsData.drop(['GenreId','08/09/16', '08/08/16', '08/07/16', '08/06/16'], axis=1)

	print "Review headers are \n", list(df_ReviewData),"\n-----\n", list(df_TweetsData)
	#print df_ReviewData
	
	for i in range(1,len(df_ReviewData),1):
		#print len(df_ReviewData.ix[i]),len(df_TweetsData.ix[i]), "\n ---------- \n"
		print numpy.corrcoef(df_ReviewData.ix[i],df_TweetsData.ix[i])[0, 1]
	pass
	

main()
#computeCorrelation()