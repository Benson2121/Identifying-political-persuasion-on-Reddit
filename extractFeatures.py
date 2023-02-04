#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import re
import numpy as np
import argparse
import json
import string

# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}


def extract1(comment):
    """ This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 174-length vector of floating point features (only the first 29 are expected to be filled, here)
    """
    # Array with 173-length vector of floating point features
    feature_array = np.zeros(174)

    # For feature 16, initialization
    number_of_characters = 0
    number_of_char_tokens = 0

    # For feature 18-23, initialization
    bristol_norms = []

    # For feature 24-29, initialization
    warringer_norms = []

    for token in comment.split():

        if '/' not in token:
            continue

        # Extratc lemma and tag from the token
        lemma = token[:token.rindex('/')]
        tag = token[token.rindex('/') + 1:]

        # feature 1: Number of tokens in uppercase (>= 3 letters long)
        if lemma.isupper() and len(lemma) >= 3:
            feature_array[0] += 1

        lower_lemma = lemma.lower()

        # feature 2: Number of first-person pronouns
        if lower_lemma in FIRST_PERSON_PRONOUNS:
            feature_array[1] += 1
        # feature 3: Number of second-person pronouns
        if lower_lemma in SECOND_PERSON_PRONOUNS:
            feature_array[2] += 1
        # feature 4: Number of third-person pronouns
        if lower_lemma in THIRD_PERSON_PRONOUNS:
            feature_array[3] += 1
        # feature 5: Number of coordinating conjunctions
        if tag == 'CC':
            feature_array[4] += 1
        # feature 6: Number of past-tense verbs
        if tag == 'VBD':
            feature_array[5] += 1
        # feature 7: Number of future-tense verbs
        if lower_lemma == 'will' and tag == 'MD':
            feature_array[6] += 1
        # feature 8: Number of commas
        if lower_lemma == ',':
            feature_array[7] += 1
        # feature 9: Number of multi-character punctuation tokens
        if tag in string.punctuation and len(lower_lemma) > 1:
            feature_array[8] += 1
        # feature 10: Number of common nouns
        if tag in ('NN', 'NNS'):
            feature_array[9] += 1
        # feature 11: Number of proper nouns
        if tag in ('NNP', 'NNPS'):
            feature_array[10] += 1
        # feature 12: Number of adverbs
        if tag in ('RB', 'RBR', 'RBS'):
            feature_array[11] += 1
        # feature 13: Number of wh- words
        if tag in ('WDT', 'WP', 'WRB', 'WP$'):
            feature_array[12] += 1
        # feature 14: Number of slang acronyms
        if lower_lemma in SLANG:
            feature_array[13] += 1
        if tag not in string.punctuation:
            number_of_characters += len(lemma)
            number_of_char_tokens += 1

            if lemma in bristol_words:
                bristol_norms.append(bristol_words.index(lemma))
            if lemma in warringer_words:
                warringer_norms.append(warringer_words.index(lemma))

    # feature 7: Number of future-tense verbs (case of 'go/going to do')
    feature_array[6] += len([re.finditer(r"(GO|Going|go)/VBG (TO|to)/TO .*(/VB)", comment)])
    # feature 7: Number of future-tense verbs (case of 'gonna do')
    feature_array[6] += len([re.finditer(r"(GON|gon)/VBG (NA|na)/TO .*(/VB)", comment)])

    # feature 15: Average length of sentences, in tokens
    number_of_sentences = comment.count('./.') + 1
    feature_array[14] = (len(comment.split()) / number_of_sentences)

    # feature 16: Average length of tokens, excluding punctuation-only tokens, in characters
    feature_array[15] = (number_of_characters / number_of_char_tokens) if number_of_char_tokens != 0 else 0

    # feature 17: Number of sentences.
    feature_array[16] = number_of_sentences

    if len(bristol_norms) >= 1:
        # feature 18-20: Average of AoA, IMG, FAM from Bristol, Gilhooly, and Logie norms
        feature_array[17:20] = np.mean(bristol_scores[:, bristol_norms], axis=1)
        # feature 21-23: Standard deviation of AoA, IMG, FAM from Bristol, Gilhooly, and Logie norms
        feature_array[20:23] = np.std(bristol_scores[:, bristol_norms], axis=1)

    if len(warringer_norms) >= 1:
        # feature 24-26: Average of V.Mean.Sum, A.Mean.Sum, D.Mean.Sum from Warringer norms
        feature_array[24:27] = np.mean(warringer_scores[:, warringer_norms], axis=1)
        # feature 27-29: Standard deviation of V.Mean.Sum, A.Mean.Sum, D.Mean.Sum from Warringer norms
        feature_array[24:27] = np.std(warringer_scores[:, warringer_norms], axis=1)

    return feature_array


def extract2(feat, comment_class, comment_id):
    """ This function adds features 30-173 for a single comment.

    Parameters:
        feat: np.array of length 174
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feat : numpy Array, a 174-length vector of floating point features (this
        function adds feature 30-173). This should be a modified version of
        the parameter feats.
    """

    # Find the index i of the ID in the appropriate ID text file, for the category
    index_IDs = comment_id_index[comment_class][comment_id]

    # Extract LIWC data to the feat
    feat[29:173] = LIWC_data[comment_class][index_IDs, :]

    return feat


def get_label(feat, comment_class):
    """
    Get labels for each sample ('Left', 'Center', 'Right', 'Alt').

    :param feat: np.array of length 174
    :param comment_class: list of classes ['Left', 'Center', 'Right', 'Alt']
    :return: np.array of length 174
    """
    feat[173] = cat_list.index(comment_class)
    return feat


def get_data(path, columns):
    """
    Get data for bristol and warringer.

    :param path: path to the designated file
    :param columns: useful columns in the file
    :return: data of bristol and warringer scores
    """
    data = [[] for i in range(0, 4)]
    with open(path, mode='r') as file:
        file.readline()
        for line in file:
            line_content = line.split(',')
            for i, column in enumerate(columns):
                if line_content[0] == '':
                    continue
                if i == 0:
                    data[i].append(line_content[column - 1])
                else:
                    data[i].append(float(line_content[column - 1]))
    return data


def main(args):
    # Declare necessary global variables here.
    global cat_list, comment_id_index
    global bristol_path, warringer_path
    global bristol_columns, warringer_columns
    global bristol_data, warringer_data
    global bristol_words, bristol_scores, warringer_words, warringer_scores
    global LIWC_data

    # Declare labels and collect words for each label
    cat_list = ['Left', 'Center', 'Right', 'Alt']
    comment_id_index = {'Left': {}, 'Center': {}, 'Right': {}, 'Alt': {}}

    # Declare the paths to get useful bristol and warringer scores and useful columns
    bristol_path = '/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv'
    warringer_path = '/u/cs401/Wordlists/Ratings_Warriner_et_al.csv'
    bristol_columns = [2, 4, 5, 6]
    warringer_columns = [2, 3, 6, 9]
    LIWC_data = {}

    # Get data and split data in to words and scores.
    bristol_data = get_data(bristol_path, bristol_columns)
    warringer_data = get_data(warringer_path, warringer_columns)
    bristol_words = bristol_data[0]
    warringer_words = warringer_data[0]
    bristol_scores = np.array(bristol_data[1:])
    warringer_scores = np.array(warringer_data[1:])

    # Load npy_data and fill in the index_of_words for each label
    for cat in cat_list:
        npy_path = f'/u/cs401/A1/feats/{cat}_feats.dat.npy'
        LIWC_data[cat] = np.load(npy_path)
        txt_path = f'/u/cs401/A1/feats/{cat}_IDs.txt'
        with open(txt_path, mode='r') as f_file:
            f_file.readline()
            for index, line in enumerate(f_file):
                dp_list = line.split()
                comment_id_index[cat][dp_list[0]] = index

    # Load data
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173 + 1))

    # Fill in the extracted features into the array
    loop_i = 0
    for loaded_line in data:
        feats[loop_i] = extract1(loaded_line['body'])
        feats[loop_i] = extract2(feats[loop_i, :], loaded_line["cat"], loaded_line["id"])
        feats[loop_i] = get_label(feats[loop_i, :], loaded_line["cat"])
        loop_i += 1

    np.savez_compressed(args.output, feats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir",
                        help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.",
                        default="/u/cs401/A1/")
    args = parser.parse_args()

    main(args)