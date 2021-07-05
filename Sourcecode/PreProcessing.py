''' 
Program to find the semantic similarity between the tweets as well as the features.
We have already created a file with 
File1: tweet:class:date and 
File2: review:class:date csv files.
Now I will create the matrix of m x n i.e for every occurence in File 1= m and File 2=n 

1) try just lemmatization
2) try Cosine similarity with 1
3) try basic stop word removal and lemmatization
4) try Cosine similarity with 3
3) try CMU' twitter for POS tagging
 
'''

#import LibTweeTokenize as TweetTok
#x = TweetTok.tokenizeRawTweetText("s mobile be terrible and the whole platform be too complicated. and i m not sure folk from microsoft will help here.")
#print x

import sys
import csv
import sklearn.metrics.pairwise as sklr
from requests import get
sys.path.append('/home/gouri/Ucalgary/672/GouriLibraries')
import LoadJson as ljson
from pprint import pprint

SSS_URL = "http://swoogle.umbc.edu/SimService/GetSimilarity"
#SSS_URL = "http://swoogle.umbc.edu/StsService/GetStsSim?"

pathIs = "/home/gouri/Ucalgary/672/Kendle and Kylie" #dropbox" #GoogleCast" #"LinkedIn"
tweetsFile = pathIs+"/tweet_date_class.csv"
reviewFile = pathIs+"/review_date_class.csv"

dropbox = [[41, 524],  [74, 1719], [1201,4159], [79,2848], [244,2325], [1143,2342], [244,2187 ],
		   [79, 2932], [79, 2947], [79, 2947], [79, 3003], [79, 3013], [79, 3061], [79, 3067], 
		   [1205, 1405], [2879,2129], [1241,190], [1201, 4159] ]
def lemmatization():
	pass

def removeStopwords():
	pass

def CosineSimilarity():
	pass

#tool implementation from website:  http://swoogle.umbc.edu/SimService/api.html
#it accept the POS tagged data as well
def Simtool(s1, s2, type='relation', corpus='webbase'):
	try:
		response = get(SSS_URL, params={'operation':'api','phrase1':s1,'phrase2':s2,'type':type,'corpus':corpus})
		#print float(response.text.strip())
		return float(response.text.strip())
	except:
		print 'Error in getting similarity for %s: %s' % ((s1,s2), response)
		return 0.0


def start():
	#read the file which has the pair with information.
	dfTwitterData = ljson.readCSVdf(tweetsFile)
	dfReviewData = ljson.readCSVdf(reviewFile)
	
	#in a loop pass pair to SimTool and display the output
	#for every entry in tweets file compare with review: mxn comparisons
	#fd = open('SimilarityGoogleCast.csv','a')
	print  "tweet", "#", "tweetDate", "#", "review", "#", "reviewDate","#", "similar", "#", "simScore","#", "class"
	for index, i in dfTwitterData.iterrows():
		#print str(i.Class), type(i.Class)
		#if str(i.Class) != "nan":
		for index, j in dfReviewData.iterrows():
			if j.Class == i.Class and str(j.Class)!= "nan":
				similarityIs = Simtool(i.tweet, j.body)
				
				if similarityIs >= 0.4:
					print  i.tweet, "#", i.Date, "#", j.body, "#", j.Date,"#", 1, "#", similarityIs,"#", j.Class
				else:
					print  i.tweet, "#", i.Date, "#", j.body, "#", j.Date,"#", 0, "#", similarityIs,"#",j.Class
					#print text
					#ljson.writeToCSV(fd,str(text))
			#print i.tweet, j.body
			
 			#print sklr.cosine_similarity(dfTwitterData['tweet'][i[0]], dfReviewData['body'][i[1]])
 	#fd.close()
	#lemmatize data and pass

	#POS tag data and pass


	pass

start()		