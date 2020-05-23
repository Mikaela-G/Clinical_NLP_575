"""
Goals:
- WE ALSO HAVE TO LOAD TESTING DATA INTO IOB FORMAT
- Load training data (two folders: original text & corresponding annotations)
- Tokenize on whitespace
- Convert tokens to IOB sequences
- Store each token as row in Pandas dataframe with these columns:
    - token, IOB, doc_ID (filename), sent_ID (sentence number in document), word_ID (word number in document)
- Insert Pandas dataframe into SQL database? or into pickle?
"""

import pandas as pd
###import pymysql
###import argparse???
import os
import sys

class DataLoader:

    def __init__(self, txt_folder, concept_folder):
        """
        Initialize class variables.

        :param txt_folder: str containing filepath to unstructured txt folder
        :param concept_folder: str containing filepath to folder
                                with corresponding concept annotation files
        """
        self.txt_folder = txt_folder
        self.concept_folder = concept_folder
        
        self.data = self.process_txt_folder()
        self.process_concept_folder()
        
    def process_txt_folder(self):
        """
        Process all files in folder with raw training data.
        """
        df = {'token':[], 'IOB':[],
                'doc_ID':[], 'sent_ID':[], 'word_ID':[]}

        for filename in os.listdir(self.txt_folder):
            filepath = os.path.join(self.txt_folder, filename)
            self.parse_txt_file(filename, filepath, df)
        
        df = pd.DataFrame.from_dict(df)
        return df

    def parse_txt_file(self, filename, filepath, df):
        """
        Process training data file.
        Store tokens, doc_ID, sent_ID, and word_ID in dataframe.

        :param filename: str containing filename sans .txt
        :param filepath: str containing absolute path
        :param df: dict with (key, val) as (str, list)
        """
        doc_ID = filename.split('.')[0]
        with open(filepath, 'r') as f:
            sent_ID = 0
            for sent in f:
                sent = sent.rstrip().split()
                for idx, word in enumerate(sent):
                    df['token'].append(word)
                    df['IOB'].append('<NONE>') ###
                    df['doc_ID'].append(doc_ID)
                    df['sent_ID'].append(sent_ID)
                    df['word_ID'].append(idx)
                sent_ID += 1

    def process_concept_folder(self):
        """
        Process all files in folder with training data annotations.
        """
        ### iterate through each file in self.concept_folder
            ### call parse_concept_file() on each file
    
    def parse_concept_file(self, filepath):
        """
        Process training data annotation file.
        Generate IOB tags for concepts and insert in dataframe.
        """

def main():
    ### load small example training data (just to test out code on small subset)
    example = DataLoader(sys.argv[1], sys.argv[2])
    train_df = example.data
    train_df['type'] = 'train'
    ### print(train_df)

    # # load beth training data
    # beth = DataLoader(sys.argv[3], sys.argv[4])
    # beth_df = beth.data
    # beth_df['type'] = 'train'
    # # load partners training data
    # partners = DataLoader(sys.argv[5], sys.argv[6])
    # partners_df = partners.data
    # partners_df['type'] = 'train'
    # # run pipeline on test data
    # test = DataLoader(sys.argv[7], sys.argv[8])
    # test_df = test.data
    # test_df['type'] = 'test'
    # # merge beth, partners, and test dataframes into one

    ### pickle or SQL???

if __name__ == "__main__":
    main()
