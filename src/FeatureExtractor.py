"""
Methods:
- Extract morphological features
- Extract grammatical features
- Extract context-based features
- Extract MetaMap-based features
- Extract sentence-level features
- Extract skip-gram features
"""
import nltk
import re

class FeatureExtractor:

    def __init__(self, data):
        """
        Initialize class variables.

        :param data: Pandas dataframe
        """
        self.data = data
        # convert tokens column to list
        self.tokens = list(self.data['token'])        
        ### extract features & update df with results of each function
        self.morph_feats()
        ###self.gram_feats() ### temporarily commented out bc slow
        self.cont_feats()
        self.metamap_feats()
        self.sent_feats()
        self.skipgram_feats()

    def morph_feats(self):
        tokens = list(self.data['token'])
        prefix_list = []
        suffix_list = []
        capitalized = []
        contains_special = []

        for token in tokens:
            if token[0].isupper():
                capitalized.append('True')
            else:
                capitalized.append('False')
            if not re.match(r'^\w+$', token):
                contains_special.append('True')
            else: 
                contains_special.append('False')

        self.data['Capitalizations'] = capitalized
        self.data['Special Characters'] = contains_special 

    
    def gram_feats(self):
        # tag tokens for POS
        pos_list = nltk.pos_tag(self.tokens)
        pos_list = [tup[1] for tup in pos_list]
        # add POS column to dataframe
        self.data['POS'] = pos_list
    
    def cont_feats(self):
        # get tokens from dataframe as list
        tokens = list(self.data['token'])
        # set up list for context features
        prev2_list = []
        prev_list = []
        next_list = []
        next2_list = []
        
        for i in range(len(tokens)):
            if i > 0:
                prev_list.append(tokens[i-1])
            else:
                prev_list.append('')
            if i > 1:
                prev2_list.append(tokens[i-2])
            else:
                prev2_list.append('')
            if i < len(tokens) - 1:
                next_list.append(tokens[i+1])
            else:
                next_list.append('')
            if i < len(tokens) - 2:
                next2_list.append(tokens[i+2])
            else:
                next2_list.append('')

        self.data['Prev2'] = prev2_list
        self.data['Prev'] = prev_list
        self.data['Next'] = next_list
        self.data['Next2'] = next2_list

    
    def metamap_feats(self):
        pass ###
    
    def sent_feats(self):
        pass ###
    
    def skipgram_feats(self):
        pass ###