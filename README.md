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
