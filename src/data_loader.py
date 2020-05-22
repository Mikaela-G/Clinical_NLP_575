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
        df = pd.DataFrame(columns=['token', 'IOB',
                                'doc_ID', 'sent_ID', 'word_ID'])
        ### iterate through each file in self.txt_folder
            ### call parse_txt_file() on each file
            
        return df

    def parse_txt_file(self, filepath):
        """
        Process training data file.
        Store tokens, doc_ID, sent_ID, and word_ID in dataframe.
        """
        doc_ID = filepath
        ### open file
            ### iterate through each sentence
                ### iterate through each word
                    ### store entry in dataframe

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
    train_df = example.data()
    train_df['type'] = 'train'

    # # load beth training data
    # beth = DataLoader(sys.argv[3], sys.argv[4])
    # # load partners training data
    # partners = DataLoader(sys.argv[5], sys.argv[6])
    # # run pipeline on test data
    # test = DataLoader(sys.argv[7], sys.argv[8])
    # # merge beth, partners, and test dataframes into one

    ### pickle or SQL???

if __name__ == "__main__":
    main()
