#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz
import html
import sys
import argparse
import os
import json
import re
import spacy

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
nlp.add_pipe('sentencizer')


def preproc1(comment, steps=range(1, 6)):
    """ This function pre-processes a single comment

    Parameters:
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step

    Returns:
        modComm : string, the modified comment
    """
    modComm = comment
    if 1 in steps:
        # Replace all non-space whitespace characters, including newlines, tabs, and carriage returns, with spaces.
        modComm = re.sub(r"\n+", " ", modComm)
        modComm = re.sub(r"\t+", " ", modComm)
        modComm = re.sub(r"\r+", " ", modComm)

    if 2 in steps:
        # Replace HTML character codes with their ASCII equivalent.
        modComm = html.unescape(modComm)

    if 3 in steps:
        # Remove URLs
        modComm = re.sub(r"(http|www)\S+", "", modComm)

    if 4 in steps:
        # Remove duplicate spaces between tokens.
        modComm = re.sub(' +', ' ', modComm)

    if 5 in steps:
        modComm_doc = nlp(modComm)

        modComm_sentences = []
        for sentence in modComm_doc.sents:
            new_tokens = []
            for token in sentence:

                # Lemmatization
                if (not token.text.startswith('-')) and token.lemma_.startswith('-'):
                    new_text = token.text
                else:
                    new_text = token.lemma_

                if token.text.isupper():
                    new_text = new_text.upper()
                else:
                    new_text = new_text.lower()

                # Tagging
                new_tokens.append(new_text + '/' + token.tag_)

            modComm_sentences.append(' '.join(new_tokens))

        # Sentence segmentation
        modComm = '\n'.join(modComm_sentences)

        # Make sure to:
        #    * Insert "\n" between sentences.
        #    * Split tokens with spaces.
        #    * Write "/POS" after each token.

    return modComm


def main(args):
    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print("Processing " + fullFile)

            data = json.load(open(fullFile))

            # Subsampling
            sample = args.ID[0] % len(data)

            # Select appropriate args.max lines
            for line in data[sample: sample + args.max]:

                # Read those lines with something like `j = json.loads(line)`
                loaded_line = json.loads(line)

                # Choose to retain fields from those lines that are relevant to you
                datum = {}
                datum['id'] = loaded_line['id']
                datum['body'] = loaded_line['body']

                # Add a field to each selected line called 'cat' with the value of 'file'
                datum['cat'] = file

                # Process and replace the body field (j['body']) with preproc1(...) using default for `steps` argument
                datum['body'] = preproc1(datum['body'])

                # Append the result to 'allOutput'
                allOutput.append(datum)

    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", type=int, help="The maximum number of comments to read from each file", default=10000)
    parser.add_argument("--a1_dir",
                        help="The directory for A1. Should contain subdir data. Defaults to the directory for A1 on cdf.",
                        default='/u/cs401/A1')

    args = parser.parse_args()

    if (args.max > 200272):
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)

    indir = os.path.join(args.a1_dir, 'data')
    main(args)
