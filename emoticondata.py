from tweet import Tweet
from instance import Instance
import re
import cPickle
import csv
import subprocess
import string
		







def tag_content(content_file,tweets):
    # content_file: destination for pre-tag output
    # tweets : dict of tweets
	# this function writes <uid sid tweet\n> to content_file
	# this content_file is then tagged by the arc tagger

    script_path = "./ark-tweet-nlp-0.3.2/runTagger.sh --output-format conll"
    tagged_file = "{0}".format(content_file.replace("content","tagged"))
    
    outfile = open(content_file,"w")
    for key,tweet in tweets.items():

        uid,sid = tweet.key
        text = tweet.text
        if text:
            try:
                text = text.encode('ascii','ignore')
            except:
                print text
            outline ="{0} {1} {2}\n".format(uid,sid,text)
            outfile.write(outline)
        else:
            print "no text {0}\n".format(key)
    outfile.close()
    command = "{0} {1} > {2}".format(script_path,content_file,tagged_file)
    print "Calling {0}\n".format(command)
    subprocess.call([command],shell=True)
    return tagged_file

def load_parsed_tweets(taggedfile):
	# this function takes the name of a file containing arc tagged tweets.
	# the ARK tagger has different tags than the weib polarity set thus we need to create a mapping
	# mapping found --> https://github.com/brendano/ark-tweet-nlp/blob/master/docs/annot_guidelines.md

    print "loading tagged tweets from {0}".format(taggedfile)
    tag_map = dict([("N","noun"),("^","noun"),("Z","noun"),("S","noun"),("O","noun"),("L","noun"),
		("V","verb"),("R","adverb"),("A","adj"),("D","anypos"),("T","anypos"),("P","anypos"),("!","anypos")])
    tagger = lambda tag : tag_map[tag] if tag in tag_map else "anypos"

	#tweet_file = open("data/tagged_tweets.txt","r")	
    tuple_tweet = []
    count = 0
    tweet_dict = {}
    tweet_file = open(taggedfile,"r")
    for tweet in tweet_file:
        info = tweet.split()
        if info:
            count+=1
            word,tag,conf = info
            if count == 1:
                uid = word
            elif count == 2:
                sid = word
            else:
                tuple_tweet.append((word,tag))
        else:
            if (uid,sid) not in tweet_dict:
                tweet_dict[(uid,sid)] = tuple_tweet
            #print tweet_dict[(uid,sid)]
            count = 0
            tuple_tweet = []

    tweet_file.close()
    return tag_map, tagger, tweet_dict


keep = ["!","@","#"]

punct = string.punctuation
for each in keep:
	punct.replace(each,"")
uid = 0
sid = 0
tweet_dict = {}
instance_dict = {}
with open("emotdata.csv") as data:
	datareader = csv.reader(data, delimiter=",")
	for row in datareader:
		key = (str(uid),str(sid))
		try:
			polarity,tid,date,query,user,text = row
		except:
			print row
		if polarity == "4":
			label = "positive"
		elif polarity == "0":
			label = "negative"
		link = re.search(r'(https?://)?([-\w]+\.[-\w\.]+)+\w(:\d+)?((/)?([-\w/_\.]*(\?\S+)?)?)*',text)
		if link:
			text =re.sub(r'(https?://)?([-\w]+\.[-\w\.]+)+\w(:\d+)?((/)?([-\w/_\.]*(\?\S+)?)?)*',"URL",text)
		for letter in text:
			if letter in punct:
				text = text.replace(letter,"")
		tweet_dict[key] = Tweet(uid=uid,sid=sid,key=key,text=text)
		instance_dict[key] = Instance(uid=uid,sid=sid,task="A",key=key,startpos=0,endpos=0,label=label)
		uid+=1
		sid+=1

tag_content("content_emotdata.txt", tweet_dict)
t,tmap,tagged = load_parsed_tweets("tagged_emotdata.txt")
new_tweet_dict = {}
for key,tweet in tweet_dict.items():
	tagtweet = tagged[key]
	tweet.target = tagtweet
	tweet.tagged_tweet =tagtweet
	new_tweet_dict[key] = tweet
cPickle.dump(new_tweet_dict,open("tweet_emoticondata.pkl","wb"))
cPickle.dump(instance_dict,open("instance_emoticondata.pkl","wb"))




