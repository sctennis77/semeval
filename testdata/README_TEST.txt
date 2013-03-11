TEST datas for SemEval-2013 Task #2: Sentiment Analysis on Twitter

Task organizers:
Theresa Wilson   Johns Hopkins University, HLTCOE
Zornitsa Kozareva  University of Southern California, ISI
Preslav Nakov  Qatar Computing Research Institute, Qatar Foundation
Sara Rosenthal  Columbia University
Veselin Stoyanov Johns Hopkins University 
Alan Ritter  University of Washington

The accompanying dataset is released under a Creative Commons Attribution 3.0 Unported License (http://creativecommons.org/licenses/by/3.0/).

Please note that by downloading the Twitter data you agree to abide by the Twitter terms of service (https://twitter.com/tos), and in particular you agree not to redistribute the data and to delete tweets that are marked deleted in the future. You MUST NOT re-distribute the tweets, the annotations or the corpus obtained, as this violates the Twitter Terms of Use.


Version 1.0: March 6, 2013


SUMMARY

TASK A:
twitter-test-input-A.tsv -- test input for task A, Twitter messages
sms-test-input-A.tsv     -- test input for task A, SMS messages

TASK B:
twitter-test-input-B.tsv -- test input for task B, Twitter messages
sms-test-input-B.tsv     -- test input for task B, SMS messages


IMPORTANT

To use this test dataset, the participants should download (1), and most likely also (2) and (3):

1. the official scorer and format checker: from the FTP server at /home/task2
2. the training dataset: http://www.cs.york.ac.uk/semeval-2013/task2/index.php?id=data
3. the dev dataset: from the FTP server (also available from the above link; the version on the FTP server comes in format that is suitable for the scorer)

The format checker released together with the scorer should be used to check the output before submitting the results.


INPUT DATA FORMAT

-----------------------TASK A-----------------------------------------
--Test Data--

The format for the test file is as follows:
id1<TAB>id2<TAB>start_token<TAB>end_token<TAB>unknwn<TAB>tweet_text

for example:
418381654813081609      15115101        2       2       unknwn  amoure wins oscar
418381654813081610      15115101        3       4       unknwn  who's a master brogramer now?

--System Output--
We expect the following format for the prediction file is:
id1<TAB>id2<TAB>start_token<TAB>end_token<TAB>pred_class<TAB>tweet_text

where the text field is optional. For example:
418381654813081609      15115101        2       2       positive  amoure wins oscar
418381654813081610      15115101        3       4       neutral  who's a master brogramer now?


--Gold Standard--
The gold standard following the same format as the system output above 

-----------------------TASK B-----------------------------------------
(Task B uses the same format as Task A, but it excludes the start and end token fields.)
--Test Data--

The format for the test file is as follows:
id1<TAB>id2<TAB>unknwn<TAB>tweet_text

for example:
418381654813081609      15115101       unknwn  amoure wins oscar
418381654813081610      15115101       unknwn  who's a master brogramer now?

--System Output--
We expect the following format for the prediction file is:
id1<TAB>id2<TAB>pred_class<TAB>tweet_text

where the text field is optional. For example:
418381654813081609      15115101        positive  amoure wins oscar
418381654813081610      15115101        neutral  who's a master brogramer now?


--Gold Standard--
The gold standard following the same format as the system output above 


EVALUATION

The metric for evaluating the participants' systems will be average F-measure (averaged F-positive and F-negative, and ignoring F-neutral; note that this does not make the task binary!), as well as F-measure for each class (positive, negative, neutral), which can be illuminating when comparing the performance of different systems.
For each subtask, the systems will be ranked based on their average F-measure.  Separate rankings for each test dataset will be produced.

See also the scorer for details on scoring the output.


DATASET USE

The development dataset is intended to be used as a development-time evaluation dataset as the participants develop their systems. However, the participants are free to use the dataset in any way they like, e.g., they can add it to their training dataset as well.

Participants should note that there are two test datasets, one composed of Twitter messages and another one composed of SMS messages, for which they have not received any explicit training data. The purpose of having a separate test set of SMS messages is to see how well systems trained on Twitter data can generalize to other types of message data.


ABOUT "OBJECTIVE"

Some of the datasets (e.g., the training dataset) contain "objective" labels, which the participants are free to use on training as they wish. However, we recommend that for task A these labels be ignored since there will be no "objective" labels in the testing dataset. For task B, "objective" and "neutral" labels should be merged into "neutral"; the two labels will also be merged likewise for the test dataset. So, at test time, for both task A and task B, the systems will have to predict just three labels: positive, negative and neutral. However, while for task A neutral means just "neutral", for task B, neutral means "neutral OR objective".


TEST PROCEDURE

Task participants must submit their runs by the final deadline of March 15, 2013 (23:59 at Midway, Midway Islands, United States: see http://www.timeanddate.com/worldclock/city.html?n=1890). Late submissions will not be counted.
Note that you can overwrite your own submissions on the FTP server multiple times, but only before the deadline.
Thus, we advise that you submit your runs early, and possibly overwrite them later if there is time for that.

For *each task* AND for *each test dataset*, each team may submit two runs: one constrained and one unconstrained:

(1) Constrained - using the provided training AND development data only; other resources, such as lexicons are allowed; however, it is not allowed to use additional tweets/SMS messages or additional sentences with sentiment annotations; and

(2) Unconstrained - using additional data, e.g., additional tweets/SMS messages or additional sentences annotated for sentiment.


SUBMISSION PROCEDURE

1. Each participating team should have registered on http://www.cs.york.ac.uk/semeval-2013. Only one registration is necessary per team even participating in multiple tasks. Registered participants should have received instructions on how to access the FTP server. If not, please tell us immediately: semevaltweet-2013@googlegroups.com
The results should be uploaded to the FTP server!

2. Each team can make one submission for each combination of: subtask (A or B), dataset (Twitter or SMS) and training condition (constrained or unconstrained). This makes a total of 8 submissions allowed.

3. Name of the submissions:  "task2-GROUP-SUBTASK-DATASET-CONDITION.zip".
Where GROUP is the group name, SUBTASK is "A" or "B", DATASET is "Twitter" or "SMS", and CONDITION is "constrained" or "unconstrained"
For example: "task2-QCRI-A-twitter-constrained.zip", "task2-JHU-B-SMS-unconstrained.zip".

4. Format of the submission: Each ZIP file should contain two files that have the same name as the ZIP file but different extensions:

- task2-GROUP-SUBTASK-DATASET-CONDITION.output
- task2-GROUP-SUBTASK-DATASET-CONDITION.description

For example, "task2-QCRI-A-twitter-constrained.zip" should contain "task2-QCRI-A-twitter-constrained.output" and "task2-QCRI-A-twitter-constrained.description".

The first file should follow the output format specified above.

The second file should have the format of the SUBMISSION_DESCRIPTION_TEMPLATE.txt.
It contains the following information:

	1. Site ID

	2. Site affiliation

	3. Contact information

	4. Submission, i.e., ZIP file name

	5. System specs

	- 5.1 Core approach

	- 5.2 Supervised or unsupervised

	- 5.3 Critical features used

	- 5.4 Critical tools used

	- 5.5 Significant data pre/post-processing

	- 5.6 Other data used (outside of the provided)

	6 References (if applicable)



USEFUL LINKS:

Google group: semevaltweet-2013@googlegroups.com
Task website: http://www.cs.york.ac.uk/semeval-2013/task2/
SemEval-2013 website: http://www.cs.york.ac.uk/semeval-2013/


TASK SCHEDULE

August 1, 2012   Trial data has been released
September 12, 2012  First Call for participation
January 10, 2013  Training Data (batch 1) released
February 15, 2013   Registration Deadline for Task Participants
February 19, 2013  Full Training Data released
February 28, 2013  Development Data released (can be used also for training)
March 6, 2013 *Test* data released to the participants
March 15, 2013   Participants predictions due (end of evaluation period)
April 9, 2013   Paper submission deadline [TBC]
April 23, 2013 Reviews Due [TBC]
May 4, 2013   Camera ready Due [TBC]
June 13-14, 2013 [TBC], SemEval Workshop, co-located with NAACL & *Sem, Atlanta Georgia
