from tweet import Tweet
from instance import Instance
import string
import cPickle
import sys
import re


# TASK A PARSING
tsvfile = sys.argv[1]
keep = ["!","@","#"]

punct = string.punctuation
for each in keep:
	punct.replace(each,"")
datafile = open(tsvfile,"rb")
tweet_dict = {}
instance_dict = {}
task = "B"
for each in datafile:
	try:
		if task == "A":
			sid,uid,startpos,endpos,label,text = each.split("\t")
		elif task == "B":
			sid,uid,label,text = each.split("\t")
		saved = False
		key = sid,uid
		if  uid not in tweet_dict:
			print "adding {0}\n".format(uid)
			#try:
			text = text.decode('utf-8-sig')
			link = re.search(r'(https?://)?([-\w]+\.[-\w\.]+)+\w(:\d+)?((/)?([-\w/_\.]*(\?\S+)?)?)*',text)
			if link:
				text =re.sub(r'(https?://)?([-\w]+\.[-\w\.]+)+\w(:\d+)?((/)?([-\w/_\.]*(\?\S+)?)?)*',"URL",text)
				print text
			#text = re.sub(r'/^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$/',"URL",text)
			#print text
			for each in text:
				if each in punct:
					text = text.replace(each,"")
			#text = text.encode("ascii",'ignore')
			tweet_dict[uid] = Tweet(uid=uid,sid=sid,key=key,text=text)
			#except:
				#print "failed encoding text for {0}\n".format(key)
		if uid not in instance_dict:
			if task == "A":
				instance_dict[uid] = Instance(uid=uid,sid=sid,task="A",key=key,startpos=startpos,endpos=endpos,label=label)
			elif task == "B":
				instance_dict[uid] = Instance(uid=uid,sid=sid,task="B",key=key,keyword=None,label=label)
	except Exception as e:
		print e.message
		print "failed to parse {0}\n".format(each)


tweet_file = "tweet_"+tsvfile.split("/")[1].replace(".tsv",".pkl")
instance_file = "instance_"+tsvfile.split("/")[1].replace(".tsv",".pkl")
cPickle.dump(tweet_dict,open(tweet_file,"wb"))
cPickle.dump(instance_dict,open(instance_file,"wb"))





			


