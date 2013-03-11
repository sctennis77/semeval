from classify import Classifier

# seperate classifier for pos|neg|neutral
		#self.tags = dict([("N","noun"),("^","noun"),("Z","noun"),("S","noun"),("O","noun"),("L","noun"),
		#("V","verb"),("R","adverb"),("A","adj"),("D","anypos"),("T","anypos"),("P","anypos"),("!","anypos")]).keys()
		#r
class PosTagClassifier(Classifier):
	def __init__(self,**kargs):
		Classifier.__init__(self,tweets=kargs["tweets"],instances=kargs["instances"],model=kargs["model"],keys=kargs["keys"],selection=kargs["selection"])
		self.id="tagcount{0},s:{1}".format(self.num_items,self.selection)




	def build_feature_vector(self,key):
		check_tags = ["A","R","!","E"]
		# checks for emoticon in tweet -> labels that emoticon:True
		# two ways to do this --> contains any emoticon True/False or specific ones
		# doing binary for now --> THIS SHOULD CHANGE
		# IMPROVMENTS
		# we can create a mapping from many emoticons to 3-4 central ones (as seen in other work)
		features = {}
		tweet = self.tweets[key].target
		tags = [t for w,t in tweet]
		for tag in check_tags:
			count = tags.count(tag)
			features["tcount(%s)"%tag]= count
		return features
