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

class FeatureExtractor:

    def __init__(self, data):
        """
        Initialize class variables.

        :param data: Pandas dataframe
        """
        self.data = data
        ### extract features & update df with results of each function
        
        ###self.gram_feats()

    def morph_feats(self):
        pass ###
    
    def gram_feats(self):
        # convert tokens column to list
        tokens = list(self.data['token'])
        # tag tokens for POS
        pos_list = nltk.pos_tag(tokens)
        pos_list = [tup[1] for tup in pos_list]
        # add POS column to dataframe
        self.data['POS'] = pos_list
    
    def cont_feats(self):
        pass ###
    
    def metamap_feats(self):
        pass ###
    
    def sent_feats(self):
        pass ###
    
    def skipgram_feats(self):
        pass ###