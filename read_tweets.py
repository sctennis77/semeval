#!/usr/bin/python

import sys

from prepare_data import prepare_tweet_data,prepare_test_data
#from parse_emot_tweets import emot_tagged_tweets,emot_instances
import random
import cPickle
from classifiers.ngram_classify import NgramClassifier
from classifiers.repeat_classify import RepeatClassifier
from classifiers.emoticon_classify import EmoticonClassifier
from classifiers.weib_classify import WeibClassifier
from classifiers.postag_classify import PosTagClassifier
from vote import Vote
from evaluate_classifiers import evaluate_classifiers,update_classifier_accuracy,get_baseline
from eval_classifiers import get_existing_classifiers
from dircheck import checkDir,createDir
from confidence_vote import ConfidenceVote
from polarity import parse_polarity_file




 
def write_classifier_dict(keys,classifier_dict,selection,mode):
    print "writing classifier dict"
    if not(checkDir(mode=mode,sub='pickles/target/',selection=selection)):
        createDir(mode=mode,sub="pickles/target/",selection=selection)
    for cid,classifier in classifier_dict.items():
        print "pickling cid={0}".format(cid)
        print classifier
        #if checkDir('/cresults/indiv')
        outpath = "cresults/pickles/target/{0}/{1}/{2}.pkl".format(mode,selection,cid)
        #try:
        with open(outpath,'wb') as f:
            cPickle.dump(classifier,f)
       # except:
        #    print "failed pickling classifier to {0}".format(outpath)





def get_test_data(test_keys):
    test_tweets = {}
    test_instances = {}
    for each in test_keys:
        test_tweets[each] = tweets[each]
        test_instances[each] = instances[each]
    return test_tweets,test_instances

def get_ngram_classifiers(keys,existing_class={},word=True,pos=True,selection="r",rank=2000):


    classifier_dict = {}
    unigram_classifier = NgramClassifier(tweets=tweets,instances=instances,keys=keys,mode="unigrams",word=word,pos=pos,merge=True,model=False,selection=selection,rank=rank)
    if unigram_classifier.id in existing_class:
        print unigram_classifier.id + "already evaluated\n"
        
    else:
        unigram_classifier.train_classifier()
        unigram_classifier.show_features(20)
        classifier_dict[unigram_classifier.id] = unigram_classifier
   
    bigram_classifier = NgramClassifier(tweets=tweets,instances=instances,keys=keys,mode="bigrams",word=word,pos=pos,merge=True,model=False,selection=selection,rank=rank)
    if bigram_classifier.id in existing_class:
        print bigram_classifier.id + "already evaluated"
    else:
        bigram_classifier.train_classifier()
        bigram_classifier.show_features(20)
        classifier_dict[bigram_classifier.id] = bigram_classifier
    
    """trigram_classifier = NgramClassifier(tweets=tweets,instances=instances,keys=keys,mode="trigrams",word=False,pos=True,merge=True,model=False,selection=selection,rank=rank)
    if trigram_classifier.id in existing_class:
        print trigram_classifier.id + "already evaluated"
    else:
        trigram_classifier.train_classifier()
        trigram_classifier.show_features(20)
        classifier_dict[trigram_classifier.id] = trigram_classifier"""

    #classifier_dict[trigram_classifier.id] = trigram_classifier
    #cout = "cresults/pickles/ngram/{0}ngram_pos_word_classifiers_{1}.pkl".format(len(keys),len(classifier_dict))
    # cPickle.dump(classifier_dict,open(cout,"w"))
    return classifier_dict






def train_ngram_classifiers(mode="unigram",selection="ngramrank",word=True,pos=False,rank=1500):
    existing_classifiers = get_existing_classifiers(sub="pickles/target",selection=selection,mode=mode)
    ngram_classifiers = get_ngram_classifiers(keys, existing_classifiers,word=word,pos=pos,selection=selection,rank=rank)
    classifier_dict = ngram_classifiers
    if classifier_dict:
        print "evaluating classifier alpha_results for {0}\n".format(classifier_dict.keys())
        write_classifier_dict(keys=keys,classifier_dict=classifier_dict,selection=selection,mode=mode)
        test_keys = classifier_dict.values()[0].test_keys
        test_tweets,test_instances = get_test_data(test_keys) 
        v = Vote(tweets=test_tweets,instances=test_instances,classifiers=classifier_dict,selection=selection)
        evaluate_classifiers(v, test_keys,classifier_dict,mode=mode,selection=selection)

    # AT THIS POINT CLASSIFIERS ARE TRAINED
    # need some logic going in --> are we using already classified stuff or making new mode?
    else:
        print "already trained {0}".format(existing_classifiers)
    return existing_classifiers



def train_ngrams_byrank():

    # word
    existing = train_ngram_classifiers(mode="unigram",selection="ngramrank",word=True,pos=True,rank=2000)
    
    # word + pos
    #train_ngram_classifiers(selection="all",word=True,pos=True)


def get_misc_classifiers(keys,existing_class = {},selection="default"):     

    #weib_classifier = WeibClassfier(tagged_tweets=tagged_tweets,instances=instances,keys=keys,polarity_dict=polarity_dict,tag_map=tag_map,model=False)
    #weib_classifier.train_classifier()
    #classifier_dict[weib_classifier.id] = weib_classifier

    classifier_dict = {}
    """emot_classifier = EmoticonClassifier(tweets=tweets,instances=instances,keys=keys,model=False,selection=selection)
    if emot_classifier.id in existing_class:
        print emot_classifier.id + " already evaluated"
    else:
        emot_classifier.train_classifier()
        emot_classifier.show_features()
        classifier_dict[emot_classifier.id]=emot_classifier
    repeat_classifier = RepeatClassifier(tweets=tweets,instances=instances,keys=keys,model=False,selection=selection)
    if repeat_classifier.id in existing_class:
        print repeat_classifier.id + " already evaluated"
    else:
        repeat_classifier.train_classifier()
        repeat_classifier.show_features()
        classifier_dict[repeat_classifier.id]=repeat_classifier"""

    weib_classifier = WeibClassifier(tweets=tweets,instances=instances,keys=keys,model=False,polarity_dict=polarity_dict,selection=selection,tag_map=tag_map)
    if weib_classifier.id in existing_class:
        print weib_classifier.id + " already evaluated"
    else:
        weib_classifier.train_classifier()
        weib_classifier.show_features()
        classifier_dict[weib_classifier.id] = weib_classifier
    tag_count_classifier = PosTagClassifier(tweets=tweets,instances=instances,keys=keys,model=False,selection=selection)
    if tag_count_classifier.id in existing_class:
        print tag_count_classifier.id + " already evaluated"
    else:
        tag_count_classifier.train_classifier()
        tag_count_classifier.show_features()
        classifier_dict[tag_count_classifier.id] = tag_count_classifier

   # tags = ["E","!","A","!"]

   # tagcount_classifer = PosTagClassifier(tweets=tweets,instances=instances,keys=keys,model=False,tags=tags,selection=selection)

    return classifier_dict


def show_confusion_matrix(classifier,vtest):
# build confusion matrix over test set
 #test_truth   = [s for (t,s) in v_test]
#test_predict = [classifier.classify(t) for (t,s) in v_test]

 print 'Confusion Matrix'
#print nltk.ConfusionMatrix( test_truth, test_predict )

def train_misc_classifiers(selection="default",mode="misc"):
    # ask rich about evaluating votes on training keys instead of testing keys
    existing_classifiers = get_existing_classifiers(sub="pickles/target",selection=selection,mode=mode)
    print "existing {0}".format(existing_classifiers)
    misc_classifiers = get_misc_classifiers(keys,existing_class=existing_classifiers,selection=selection)
    classifier_dict = misc_classifiers
    if classifier_dict:
        print "evaluating classifier alpha_results for {0}\n".format(classifier_dict.keys())
        write_classifier_dict(keys=keys,classifier_dict=classifier_dict,selection=selection,mode=mode)
        test_keys = classifier_dict.values()[0].test_keys
        train_keys = classifier_dict.values()[0].train_keys
        test_tweets,test_instances = get_test_data(test_keys) 
        train_tweets,train_instances = get_test_data(train_keys)
        v = Vote(tweets=test_tweets,instances=test_instances,classifiers=classifier_dict,selection=selection)
        evaluate_classifiers(v, test_keys,classifier_dict,mode=mode,selection=selection)
    else:
        print  "already tained {0}".format(existing_classifiers)
    return existing_classifiers
    # need some logic going in --> are we using already classified stuff or making new mode?
    #  AT THIS POINT WE HAVE TRAINED THE CLASSIFIERS
    # update classifier_accuracy loads classifiers from pickles with their corresponding accuracy



def train_all_misc():

    # emoticon, repeat classifiers
    train_misc_classifiers()

def use_trained_classifiers(selection="ngramrank",mode="unigram",test_tweets={},test_instances={},cid="all",descrip="rank"):
    total_vote_dict = {}
    ud = update_classifier_accuracy(selection=selection,mode=mode)
    if cid=="all":
        print "confidence voting combined dict {0}".format(ud,keys())
        cv = ConfidenceVote(tweets= test_tweets, instances=test_instances,classifiers=ud,selection=selection)
        cv.score_all_classifiers()
        for each in test_tweets.keys():
            cv.alpha_vote(each)
        alpha_vote_dict = cv.evaluate_results()
        result_dict = score_evaluated_classifier(alpha_vote_dict, test_tweets.keys(),testset_instances,selection=selection,mode=mode,cid=cid,descrip=descrip)
        total_vote_dict["all"] = result_dict
    else:
        for key in ud:
            print "confidence voting {0}".format(key)
            cdict = {key:ud[key]}
            cv = ConfidenceVote(tweets= test_tweets, instances=test_instances,classifiers=cdict,selection=selection)
            cv.score_all_classifiers()
            for each in test_tweets.keys():
                cv.alpha_vote(each)
            alpha_vote_dict = cv.evaluate_results()
            result_dict = score_evaluated_classifier(alpha_vote_dict, test_tweets.keys(),testset_instances,selection=selection,mode=mode,cid=key,descrip=descrip)
            total_vote_dict[key]= result_dict
    return total_vote_dict



def score_evaluated_classifier(target_alpha_vote_dict,tweet_keys,testset_instances,selection="ngramrank",mode="unigram",cid="all",descrip="rank"):
    ta= target_alpha_vote_dict
    num_correct = 0
    num_wrong =0
    neg = 0
    pos = 0
    voted_dict = {}
    for key in tweet_keys:
        choice = ""
        conf = 0
        result = ta[key]
        for label,value in result.items():
            if value > conf:
                choice = label
                conf = value
            voted_dict[key] = (choice,conf)
        actual = testset_instances[key].label
        if actual:
            if choice == actual:
                num_correct+=1
            else:
                num_wrong+=1
            if actual == "negative":
                neg+=1
            if actual =="positive":
                pos+=1
        #if choice =="negative" or actual == "negative":
          #  print "vote: {0} ({1})\tactual: {2}\n".format(choice,conf,actual)

    total = num_correct + num_wrong
    percent = float(num_correct)/total
    result_file = "{0}/{1}/{3}{2}.txt".format("cEvaluations",mode,len(tweet_keys),descrip)

    with open(result_file,"a") as f:
        # total,correct,percent,numneg,numpos
        outstr = "{5}\t{0}\t{1}\t{2}\t{3}\t{4}\n".format(total,num_correct,percent,neg,pos,cid)
        f.write(outstr)
    print "num_neg = ",neg
    print "num_pos = ",pos
    print "c: {0} w: {1} acc: {2}".format(num_correct,num_wrong,float(num_correct)/total)
    return voted_dict


if __name__=='__main__':
    # so this will eventually be python read_tweets.py <tsvfile> <task> <training> <pickle files if training false>
    # or we can hardcode the best word_prob and length_prob files (i.e the biggest)
    # should we eventually combine multiple word probs into a master ???
    #- task2-GROUP-SUBTASK-DATASET-CONDITION.output
#- task2-GROUP-SUBTASK-DATASET-CONDITION.description
    try:
        train = lambda x: True if x == "True" else False
        tsvfile = sys.argv[1]
        testfile = sys.argv[2]
        task = sys.argv[3]


        if task not in ['A', 'B']:
            sys.stderr.write("Must provide task as A or B\n")            
            sys.exit(1)
     
    except IndexError:
        sys.stderr.write("read_tweets.py <tsvfile> <task> <training>")
        sys.exit(1)
    #if dataset == "emot":
        # emoticon datset
     ##  instances = emot_instances

        # normal dataset
    tweets,instances,tag_map = prepare_tweet_data(tsvfile,task)
    print tag_map
    testset_tweets,testset_instances,test_tag_map = prepare_test_data(testfile,task)        
    # lazy cleaning of objective and neutral
    objectives = [key for key in tweets if instances[key].label == "objective" or instances[key].label == "neutral"]
    popped = 0
    tpopped=0
    tneu = 0
    neu_count=0
    pred_file = open("task2-swatcs-A-twitter-constrained.output","wb")
    for key in objectives:
        if instances[key].label == "neutral":
            neu_count+=1
        if task == "A":
            instances.pop(key)
            tweets.pop(key)
            popped+=1
        elif task == "B":
            instances[key].label = "neutral"
    """test_obj = [key for key in testset_tweets if testset_instances[key].label =="objective" or testset_instances[key].label == "neutral"]
    for key in test_obj:
        if testset_instances[key] == "neutral":
            tneu +=1
        if task == "A":
            testset_instances.pop(key)
            testset_tweets.pop(key)
            tpopped+=1

    print "removed {0} total {1} neutral from training dataset\n".format(popped,neu_count)
    #print "removed {0} total {1} neutral from testing dataset\n".format(tpopped,tneu)"""

   

    polarity_dict = parse_polarity_file("subclues.tff")
    keys = tweets.keys()
    random.seed(0)
    random.shuffle(keys)
    dist = get_baseline(instances)

    # todo 
    # Negation
    # Lexicon polarity
    # rank for ngram
    # bigram
    # train with twitter160 library from stanford.

    

    # NGRAM(unigram)EVALUATIONS BY keepfeature (r=[0.1-1])

    train_ngrams_byrank()
    train_all_misc()
    buf = 80 * "*"
    ngram_res_dict = use_trained_classifiers(selection="ngramrank", mode="unigram", test_tweets=testset_tweets, test_instances = testset_instances,cid="indiv",descrip="ngramrank")
    misc_res_dict = use_trained_classifiers(selection="default", mode="misc", test_tweets=testset_tweets, test_instances = testset_instances, cid="indiv", descrip="emotorepeat")
    accum_votes = {}
    res_dict = dict(ngram_res_dict.items() + misc_res_dict.items())
    num_class= len(res_dict)
    resfname = "fulloutput-testset:{0}_class:{1}class.txt".format(len(testset_instances),num_class)
    resfile = open(resfname,"wb")
    num_neg = 0
    num_pos = 0 
    correct = 0
    wrong = 0
    act_neg =0
    act_pos = 0
    # vote from top 2 then take the tiebreaker from second two
    for key in testset_instances.keys():
        result_dict = {"positive":[],"negative":[],"neutral":[],"novote":[]}
        if key not in accum_votes:
            accum_votes[key] = {}
        actual = testset_instances[key].label
        if actual == "positive":
            act_pos+=1
        elif actual =="negative":
            act_neg +=1
        head="{0}\tact:{1}\n".format(key,actual)
        resfile.write(head)
        for class_key,result in res_dict.items():
                if key in result:
                  vote,conf = result[key]
                else:
                  vote,conf = ("novote",0.0)

                accum_votes[key][class_key] = (vote,conf)
        tmp = accum_votes[key]
        ranked = sorted(tmp,key = lambda x: tmp[x][1],reverse=True)
        for val,ckey in enumerate(ranked):
            vt,conf = tmp[ckey]
            result_dict[vt].append(conf)
            row = "{0},{1},{2}\n".format(val,ckey.split(",")[0],tmp[ckey])
            resfile.write(row)
        resfile.write("{0}\n".format(buf))
        if result_dict["positive"]:
            pavg = float(sum(result_dict["positive"]))/len(result_dict["positive"])
        else:
            pavg = 0.0
        if result_dict["negative"]:
            negavg = float(sum(result_dict["negative"]))/len(result_dict["negative"])
        else:
            negavg = 0.0
        diff = pavg - negavg
        """if abs(diff) <.2:
            print "{0} close defaulting to weib".format(diff)
            for each in ranked:
                if each.startswith("w"):
                    vt,conf = tmp[each]
                    final_vote = vt"""

        if diff>.1:
            final_vote = "positive"
            num_pos+=1
        else:
            num_neg+=1
            final_vote = "negative"
        
        if final_vote == actual:
            correct+=1
        else:

            #print key,pavg-negavg,final_vote,actual
            wrong+=1
        id1,id2 = testset_tweets[key].key
        text = " ".join([w for w,t in testset_tweets[key].target])
        startpos = testset_instances[key].startpos
        endpos = testset_instances[key].endpos
        out = [id1,id2,startpos,endpos,final_vote,text]
        outstrs = [str(i) for i in out]
        answer = "\t".join(outstrs) +"\n"
        pred_file.write(answer)



    percent = float(correct)/(correct + wrong)
    print "c: {0} w: {1}\tgp: {2} ap: {3}\tgn: {4} an: {5}\tpercent: {6}".format(correct,wrong,num_pos,act_pos,num_neg,act_neg,percent)
    resfile.close
    pred_file.close


    """train_all_misc()
    use_trained_classifiers(selection="default",mode="misc",test_tweets=testset_tweets,test_instances=testset_instances,cid="indiv",descrip="emotorepeat")"""



  #  combined_ngrams = dict(allgram.items() + allposgram.items() +targgram.items() + targposgram.items() + congram.items() + conposgram.items())



    # this code evaluates classifiers based on alpha / beta = .3







