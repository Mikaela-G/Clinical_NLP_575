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
        # get suffixes/prefixes
        # check if token is capitalized (True/False)
        # check if special characters in token (True/False)
        ## WordAsClass??
        pass ###
    
    def gram_feats(self):
        # tag tokens for POS
        pos_list = nltk.pos_tag(self.tokens)
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