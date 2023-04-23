# Identifying-political-persuasion-on-Reddit

## Introduction

This project will give you experience with a social media corpus (i.e., a collection of posts from Reddit), Python programming, part-of-speech (PoS) tags, sentiment analysis, and machine learning with scikit-learn. 

Your task is to split posts into sentences, tag them with a PoS tagger that we will provide, gather some feature information from each post, learn models, and use these to classify political persuasion. 

Sentiment analysis is an important topic in computational linguistics in which we quantify subjective aspects of language. These aspects can range from biases in social media for marketing, to a spectrum of cognitive behaviours for disease diagnosis.

## Reddit Corpus

We have curated data from Reddit by scraping subreddits, using Pushshift, by perceived political affiliation. Subreddits assigned to each of four categories: left-leaning, right-leaning, center/neutral, and ‘alternative facts’. 

Although the first three (at least) are often viewed as ordinal segments on a unidimensional spectrum, here we treat these categories as nominal classes. 

Each datum has several fields of interest, including:

- **ups**: the integer number of upvotes.
- **downs**: the integer number of downvotes.
- **score**: [ups − downs]
- **controversiality**: a combination of the popularity of a post and the ratio between ups and downs.
- **subreddit**: the subreddit from which the post was sampled.
- **author**: the author ID.
- **body**: the main textual message of the post, and our primary interest.
- **id**: the unique identifier of the comment.

## Task 1: Pre-processing, tokenizing, and tagging

The comments, as given, are not in a form amenable to feature extraction for classification – there is too much ‘noise’. Therefore, the first step is to complete a Python program named preproc.py, in accordance with Section 5, that will read subsets of JSON files, and for each comment perform the following steps, in order, on the ‘body’ field of each selected comment:

1. Replace all non-space whitespace characters, including newlines, tabs, and carriage returns, with spaces.

2. Replace HTML character codes (i.e., &...;) with their ASCII equivalent (see http://www.asciitable.com).
 
3. Remove all URLs (i.e., tokens beginning with http or www).
 
4. Remove duplicate spaces between tokens.
 
5. Apply the following steps using spaCy (see below):

     -- **Tagging**: 
       Tag each token with its part-of-speech. A tagged token consists of a word, the ‘/’ symbol, and the tag (e.g., dog/NN). See below for information on how to use the tagging module. The tagger can make mistakes.

     -- **Lemmatization**: 
       Replace the token itself with the token.lemma . E.g., words/NNS becomes word/NNS. If the lemma begins with a dash (‘-’) when the token doesn’t (e.g., -PRON- for I, just keep the token.). Retain the case of the original token when you perform this replacement. We make two distinctions here: if the original token is entirely in uppercase, the so is the lemma; otherwise, keep the lemma in lowercase.
     
     -- **Sentence segmentation**: 
       Add a newline between each sentence. For this assignment, we will use spaCy’s sentencizer component to segment sentences in a post. Remember to also mark the end of the post with a newline (watch out for duplicates!).

## Task 2: Feature extraction

The second step is to complete a Python program named extractFeatures.py, in accordance with Section 5, that takes the preprocessed comments from Task 1, extracts features that are relevant to bias detection, and builds an npz datafile that will be used to train models and classify comments in Task 3.

For each comment, you need to extract 173 features and write these, along with the category, to a single NumPy array. 
These features are listed below. Several of these features involve counting tokens based on their tags. 
For example, counting the number of adverbs in a comment involves counting the number of tokens that have been tagged as RB, RBR, or RBS. 
You may copy and modify these files, but do not change their filenames.

Features you need to extract are listed below:

1. Number of tokens in uppercase (≥ 3 letters long) 
2. Number of first-person pronouns
3. Number of second-person pronouns
4. Number of third-person pronouns
5. Number of coordinating conjunctions 
6. Number of past-tense verbs
7. Number of future-tense verbs
8. Number of commas
9. Number of multi-character punctuation tokens 
10. Number of common nouns
11. Number of proper nouns
12. Number of adverbs
13. Number of wh- words
14. Number of slang acronyms
15. Average length of sentences, in tokens
16. Average length of tokens, excluding punctuation-only tokens, in characters 
17. Number of sentences.

Lexical norms are aggregate subjective scores given to words by a large group of individuals. 
Each type of norm assigns a numerical value to each word. Here, we use two sets of norms: Bristol+GilhoolyLogie Norms and Warringer Norms

You need to extract some important features from these norms as well. They are listed below:

18. Average of AoA (100-700) from Bristol, Gilhooly, and Logie norms
19. Average of IMG from Bristol, Gilhooly, and Logie norms
20. Average of FAM from Bristol, Gilhooly, and Logie norms
21. Standard deviation of AoA (100-700) from Bristol, Gilhooly, and Logie norms 
22. Standard deviation of IMG from Bristol, Gilhooly, and Logie norms
23. Standard deviation of FAM from Bristol, Gilhooly, and Logie norms
24. Average of V.Mean.Sum from Warringer norms
25. Average of A.Mean.Sum from Warringer norms
26. Average of D.Mean.Sum from Warringer norms
27. Standard deviation of V.Mean.Sum from Warringer norms
28. Standard deviation of A.Mean.Sum from Warringer norms
29. Standard deviation of D.Mean.Sum from Warringer norms

For features 30-173, you will use **LIWC/Receptiviti** tools to extract features.

The Linguistic Inquiry & Word Count (LIWC) tool has been a standard in a variety of NLP research, especially around authorship and sentiment analysis. 
This tool provides 85 measures mostly related to word choice; more information can be found in this link (https://www.liwc.app/help/howitworks). 
The company Receptiviti provides a superset of these features, which also includes 59 measures of personality derived from text.

## Task 3: Experiments and classification

The third step is to use the features extracted in Task 2 to classify comments using the scikit-learn machine learning package. 
Here, you will modify various hyper-parameters and interpret the results analytically. 
As everyone has different slices of the data, there are no expectations on overall accuracy, but you are expected to discuss your findings with scientific rigour.


### 3.1 Classifiers ###

Train the following 5 classifiers (see hyperlinks for API) with fit(X train, y train):
1. **SGDClassifier**: support vector machine with a linear kernel.

2. **GaussianNB**: a Gaussian naive Bayes classifier.

3. **RandomForestClassifier**: with a maximum depth of 5, and 10 estimators.

4. **MLPClassifier**: A feed-forward neural network, with α = 0.05.

5. **AdaBoostClassifier**: with the default hyper-parameters.

Here, X train is the first 173 columns of your training data, and y train is the last column. Obtain
predicted labels with these classifiers using predict (X test), where X test is the first 173 columns of
your testing data. 

Obtain the 4 × 4 confusion matrix C using confusion matrix. Given that the element
at row i, column j in C is the number of instances belonging to class i that were classified as
class j, compute the following manually, using the associated function templates:

- Accuracy : the total number of correctly classified instances over all classifications.
- Recall : for each class κ, the fraction of cases that are truly class κ that were classified as κ.
- Precision : for each class κ, the fraction of cases classified as κ that truly are κ.

Write the results to the text file a1_3.1.txt in the output directory. 
For each classifier, you will print the accuracy, recall, precision, and confusion matrix.

### 3.2 Amount of training data ###

Many researchers attribute the success of modern machine learning to the sheer volume of data that is now available.
Modify the amount of data that is used to train your preferred classifier from above in five increments: 1K, 5K, 10K, 15K, and 20K. 
These can be sampled arbitrarily from the training set in Section 3.1. 

Using only the classification algorithm with the highest accuracy from Section 3.1, report the accuracies of the classifier to the file a1_3.2.txt using the format string provided in the template. 
On one or more lines following the reported accuracies, comment on the changes to accuracy as the number of
training samples increases, including at least two sentences on a possible explanation.

### 3.3 Feature analysis ###

Certain features may be more or less useful for classification, and too many can lead to overfitting or other problems.

In the example above, pp stores the p-value associated with doing a χ2 statistical test on each feature. A smaller value means the associated feature better separates the classes. Do this:

1. For the 32k training set and each number of features k = {5, 50}, find the best k features according to this approach. Write the associated p-values to a1_3.3.txt using the format strings provided.

2. Train the best classifier from section 3.1 for each of the 1K training set and the 32K training set, using only the best k = 5 features. Write the accuracies on the full test set of both classifiers to a1_3.3.txt using the format strings provided.

3. Extract the indices of the top k = 5 features using the 1K training set and take the intersection with the k = 5 features using the 32K training set.

4. Format the top k = 5 feature indices extracted from the 32K training set to file.

### 3.4 Cross-validation ###
Many papers in machine learning stick with a single subset of data for training and another for testing (occasionally with a third for validation). 
This may not be the most honest approach. Is the best classifier from Section 3.1 really the best? 
For each of the classifiers in Section 3.1, run 5-fold cross-validation given all the initially available data. 
Specifically, use KFold. Set the shuffle argument to true.

For each fold, obtain accuracy on the test partition after training on the rest for each classifier. Report the mean accuracy of each classifier for each of the 5 folds in the order specified in 3.1 to a1_3.4.txt using the format strings provided. Next, determine whether the accuracy of your best classifier, across the 5 folds, is significantly better than any others.

## Task 4: Bonus

You may decide to pursue any number of tasks of your own design related to this project.

Some ideas:

- Identify words that the PoS tagger tags incorrectly and add code that fixes those mistakes. Does this code introduce new errors elsewhere? E.g., if you always tag dog as a noun to correct a mistake, you will encounter errors when dog should be a verb. How can you mitigate such errors?

- Explore alternative features to those extracted in Task 2. What other kinds of variables would be useful in distinguishing affect? Consider, for example, the Stanford Deep Learning for Sentiment Analysis. Test your features empirically as you did in Task 3 and discuss your findings.

- Explore alternative classification methods to those used in Task 3. Explore different hyper- parameters. Which hyper-parameters give the best empirical performance, and why?

- Learn about topic modelling as in latent Dirichlet allocation. Are there topics that have an effect on the accuracy of the system? E.g., is it easier to tell how someone feels about politicians or about events? People or companies? As there may be class imbalances in the groups, how would you go about evaluating this? Go about evaluating this.

## Acknowledgment

The started code of this project is from CSC401: Natural Language Computing (Winter 2023), University of Toronto

Instructors: Annie En-Shiun Lee, Raeid Saqur, and Zining Zhu.

[Course Website](https://www.cs.toronto.edu/~raeidsaqur/csc401/)


