# Identifying-political-persuasion-on-Reddit

CSC401 - Natural Language Computing

Instructor: Annie En-Shiun Lee, Raeid Saqur, and Zining Zhu.

# Introduction

This project will give you experience with a social media corpus (i.e., a collection of posts from Reddit), Python programming, part-of-speech (PoS) tags, sentiment analysis, and machine learning with scikit-learn. Your task is to split posts into sentences, tag them with a PoS tagger that we will provide, gather some feature information from each post, learn models, and use these to classify political persuasion. Sentiment analysis is an important topic in computational linguistics in which we quantify subjective aspects of language. These aspects can range from biases in social media for marketing, to a spectrum of cognitive behaviours for disease diagnosis.

# Reddit Corpus

We have curated data from Reddit by scraping subreddits, using Pushshift, by perceived political affiliation. Table 1 shows the subreddits assigned to each of four categories: left-leaning, right-leaning, center/neutral, and ‘alternative facts’. Although the first three (at least) are often viewed as ordinal segments on a unidimensional spectrum, here we treat these categories as nominal classes. 

Each datum has several fields of interest, including:

**ups**: the integer number of upvotes.

**downs**: the integer number of downvotes.

**score**: [ups − downs]

**controversiality**: a combination of the popularity of a post and the ratio between ups and downs.

**subreddit**: the subreddit from which the post was sampled.

**author**: the author ID.

**body**: the main textual message of the post, and our primary interest.

**id**: the unique identifier of the comment.

# Task 1: Pre-processing, tokenizing, and tagging

The comments, as given, are not in a form amenable to feature extraction for classification – there is too much ‘noise’. Therefore, the first step is to complete a Python program named a1 preproc.py, in accordance with Section 5, that will read subsets of JSON files, and for each comment perform the following steps, in order, on the ‘body’ field of each selected comment:

1. Replace all non-space whitespace characters, including newlines, tabs, and carriage returns, with spaces.

2. Replace HTML character codes (i.e., &...;) with their ASCII equivalent (see http://www.asciitable.com).
 
3. Remove all URLs (i.e., tokens beginning with http or www).
 
4. Remove duplicate spaces between tokens.
 
5. Apply the following steps using spaCy (see below):

• Tagging: Tag each token with its part-of-speech. A tagged token consists of a word, the ‘/’ symbol, and the tag (e.g., dog/NN). See below for information on how to use the tagging module. The tagger can make mistakes.

• Lemmatization: Replace the token itself with the token.lemma . E.g., words/NNS becomes word/NNS. If the lemma begins with a dash (‘-’) when the token doesn’t (e.g., -PRON- for I, just keep the token.). Retain the case of the original token when you perform this replacement. We make two distinctions here: if the original token is entirely in uppercase, the so is the lemma; otherwise, keep the lemma in lowercase.
• Sentence segmentation: Add a newline between each sentence. For this assignment, we will use spaCy’s sentencizer component to segment sentences in a post. Remember to also mark the end of the post with a newline (watch out for duplicates!).
