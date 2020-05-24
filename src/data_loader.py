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

    def __init__(self, txt_folder, con_folder, data_type):
        """
        Initialize class variables.

        :param txt_folder: str containing filepath to unstructured txt folder
        :param con_folder: str containing filepath to folder
                                with corresponding concept annotation files
        :param data_type: str containing whether data is train or test
        """
        self.txt_folder = txt_folder
        self.con_folder = con_folder
        self.data_type = data_type
        
        self.data = self.process_txt_folder()
        self.process_con_folder()
        
    def process_txt_folder(self):
        """
        Process all files in folder with raw training data.
        """
        print('Started process_txt_folder()') ###

        df = {'token':[], 'IOB':[],
                'doc_ID':[], 'sent_ID':[], 'word_ID':[],
                'data_type':[]}

        for filename in os.listdir(self.txt_folder):
            filepath = os.path.join(self.txt_folder, filename)
            self.parse_txt_file(filename, filepath, df)
        
        df = pd.DataFrame.from_dict(df)

        print('Completed process_txt_folder()') ###
        
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
                    df['IOB'].append('O') # to be updated during process_con_folder()
                    df['doc_ID'].append(doc_ID)
                    df['sent_ID'].append(sent_ID)
                    df['word_ID'].append(idx)
                    df['data_type'].append(self.data_type)
                sent_ID += 1

    def process_con_folder(self):
        """
        Process all files in folder with training data annotations.
        """
        print('Started process_con_folder()') ###

        for filename in os.listdir(self.con_folder):
            filepath = os.path.join(self.con_folder, filename)
            self.parse_con_file(filename, filepath)
            print('|||PARSED CON FILE') ###

        print('Completed process_con_folder()') ###
    
    def parse_con_file(self, filename, filepath):
        """
        Process training data annotation file.
        Generate IOB tags for concepts and insert in dataframe.
        """
        doc_ID = filename.split('.')[0]
        with open(filepath, 'r') as f:
            for annotation in f:
                annotation = annotation.rstrip().split('||t=')
                # create list of tokens in concept
                concept = annotation[0].split('"')[1].split()
                # get concept span
                span = annotation[0].split()[-2:]
                span = [tag.split(':') for tag in span]
                sent_ID, word_ID_start, word_ID_end = int(span[0][0])-1, int(span[0][1]), int(span[1][1])
                # get concept type
                concept_type = annotation[1].replace('"', '')
                # create list of IOB tags
                IOB_tags = ['I-'+concept_type for token in concept]
                IOB_tags[0] = 'B-'+concept_type
                # update self.data for each token in concept
                IOB_idx = 0
                for i in range(word_ID_start, word_ID_end+1):
                    self.data.loc[(self.data['doc_ID']==doc_ID)&
                                (self.data['sent_ID']==sent_ID)&
                                (self.data['word_ID']==i), 'IOB'] = IOB_tags[IOB_idx]
                    IOB_idx += 1

def main():

    # ### load small example training data (just to test out code on small subset)
    # example = DataLoader(sys.argv[1], sys.argv[2], 'train')
    # df = example.data
    # ###print(df.loc[train_df['doc_ID']=='doc1'])

    # load training data
    beth = DataLoader(sys.argv[1], sys.argv[2], 'train')
    partners = DataLoader(sys.argv[3], sys.argv[4], 'train')
    # loda test data
    test = DataLoader(sys.argv[5], sys.argv[6], 'test')
    # merge training and test into one dataframe
    df = pd.concat([beth.data, partners.data, test.data]).reset_index(drop=True)

    print('---COMPLETED LOADING DATA FROM ALL SOURCES---') ###
    print(df.head(20)) ###

    ### EXPORTING/STORING DF: pickle or SQL???

if __name__ == "__main__":
    main()
