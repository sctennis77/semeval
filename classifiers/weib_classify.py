from classify import Classifier
class WeibClassifier(Classifier):
	def __init__(self,**kargs):
		Classifier.__init__(self,tweets=kargs["tweets"],instances=kargs["instances"],model=kargs["model"],keys=kargs["keys"],selection=kargs["selection"])
		self.polarity_dict = kargs["polarity_dict"]
		self.tag_map = kargs["tag_map"]
		self.id="weib{0},s:{1}".format(self.num_items,self.selection)
		self.prepare_features()





	def build_feature_vector(self,key):
		# checks for emoticon in tweet -> labels that emoticon:True
		# two ways to do this --> contains any emoticon True/False or specific ones
		# doing binary for now --> THIS SHOULD CHANGE
		# IMPROVMENTS
		# we can create a mapping from many emoticons to 3-4 central ones (as seen in other work)
		tagger = lambda tag : self.tag_map[tag] if tag in self.tag_map else "anypos"
		features = {}
		tweet = self.tweets[key].target
		for word,tag in tweet:
			each = (word,tagger(tag))
			if each in self.polarity_dict:
				features["lexicon_label({0})".format(self.polarity_dict[each].word)] = self.polarity_dict[each].polarity
				features["lexicon_strength({0})".format(self.polarity_dict[each].type)] = self.polarity_dict[each].polarity
		return features
