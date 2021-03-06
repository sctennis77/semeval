# file to do alpha evaluation from the classifiers using vote dicts
from dircheck import createDir, checkDir
from eval_classifiers import get_classifier_accuracy
import cPickle
def get_baseline(instances):
	dist = {}
	for key in instances:
		dist[instances[key].label] = dist.get(instances[key].label,0) + 1

	return dist
	# could return pos/total

def evaluate(results,n=100,fname=None):
    # calculates results
    header = "alpha total correct pecent-correct\n"
    if fname:
        print "evaluating to {0}\n".format(fname)
        outf = open(fname,'wb')
    else:
        print header
    for alpha in range(n):
        if alpha%10 == 0:
            alpha2 = alpha/100.0
            res = [a for a in results.values() if abs(a[3])>alpha2]
            if res:
                num_tot = len(res)
                correct = len([a for a in res if a[0] == a[4]])
                percent = correct/float(num_tot)
                if num_tot:
                    out= "{0} {1} {2} {3}\n".format(alpha2,num_tot,correct,percent)
                else:
                    out ="alpha={0} no results".format(alpha2)
                if fname:
                    outf.write(out)
                else:
                    print out
            else:
                continue
                #print "no res :(....\n"
    if fname:
        outf.close()



def evaluate_classifiers(v,test_keys,classifier_dict,selection="r",mode="unigram"):
    print "creating alpha results from classifiers..."
    results = {}
    if not(checkDir(sub="alpha_results/target/",selection=selection,mode=mode)):
        createDir(sub="alpha_results/target/",selection=selection,mode=mode)
    for cid in classifier_dict:
        print "evaulating cid={0}".format(cid)
        #if checkDir('/cresults/indiv')
     
        outpath = "cresults/alpha_results/target/{0}/{1}/{2}.txt".format(mode,selection,cid)

        v.score_tweets_bycid(cid)
        # just need to change here to do combinations of classifiers
        v.build_vote_dicts()
        basic = v.basic_result_dict
        weighted = v.weighted_result_dict
        summarized = v.summarize_weighted_results()
        for key in test_keys:
            pos_votes = basic[key].count("positive")
            neg_votes = basic[key].count("negative")
            actual = v.instances[key].label
            pos_score = summarized[key]["positive"]
            neg_score = summarized[key]["negative"]
            diff = pos_score-neg_score
            beta=0
            # this should be programable to optimize beta! 
            score_vote = "positive" if (pos_score > neg_score) else "negative"
            count_vote = "positive" if pos_votes > neg_votes else "negative" # doesnt work for ties # if certain num negvotes?
            line = (actual,pos_score,neg_score,diff,score_vote,pos_votes,neg_votes,count_vote)
            results[key] = line 

        evaluate(results,fname=outpath)
        v.reset()


def score_evaluated_classifier(target_alpha_vote_dict,tweet_keys,testset_instances):
    ta= target_alpha_vote_dict
    num_correct = 0
    num_wrong =0
    neg = 0
    for key in tweet_keys:
        choice = ""
        conf = 0
        result = ta[key]
        actual = testset_instances[key].label
        for label,value in result.items():
            if value > conf:
                choice = label
                conf = value
        if choice == actual:
            num_correct+=1
        else:
            num_wrong+=1
        if actual == "negative":
            neg+=1
        #if choice =="negative" or actual == "negative":
          #  print "vote: {0} ({1})\tactual: {2}\n".format(choice,conf,actual)
    total = num_correct + num_wrong
    print "num_neg = ",neg
    print "c: {0} w: {1} acc: {2}".format(num_correct,num_wrong,float(num_correct)/total)

def update_classifier_accuracy(selection="r",mode="unigram",baseline=0.7):

    # this should be mode 
    updated_dict = {}
    pic_path = "cresults/pickles/target/{0}/{1}/".format(mode,selection)
    outpath = "cresults/alpha_results/target/{0}/{1}/".format(mode,selection)

    print "updating classifier results from {0}\tbaseline:{1}\n".format(outpath,baseline)
    a = get_classifier_accuracy(outpath, baseline)
    for class_key,result in a.items():
        if result:
            pic_file = pic_path + class_key +".pkl"
            with open(pic_file,'rb') as f:
                print pic_file
                #try:
                cl = cPickle.load(f)
                cl.alpha_acc = result
                cl.baseline = baseline
                updated_dict[class_key] = cl
                print "updated --> {0}\n".format(pic_file)
               # except:
                   # print "error pickling classifier --> {0}\n".format(class_key)
        else:
            print "skipping classifier --> {0}\n".format(class_key)
    return updated_dict
