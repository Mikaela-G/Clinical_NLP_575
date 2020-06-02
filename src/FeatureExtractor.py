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
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
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
        # extract features & update df with results of each function
        self.morph_feats()
        self.gram_feats()
        self.cont_feats()
        self.metamap_feats()
        self.sent_feats()
        self.skipgram_feats()

    def morph_feats(self):
        tokens = list(self.data['token'])
        prefix_lists = [[] for i in range(4)] 
        suffix_lists = [[] for i in range(4)] 
        lemma_list = []
        capitalized = []
        contains_special = []

        # for lemmas
        lemmatizer = WordNetLemmatizer()

        #map pos tags to format for lemmatizer
        def get_wordnet_pos(word): 
            tag = nltk.pos_tag([word])[0][1][0].upper()
            tag_dict = {"J": wordnet.ADJ,
                        "N": wordnet.NOUN,
                        "V": wordnet.VERB,
                        "R": wordnet.ADV}

            return tag_dict.get(tag, wordnet.NOUN)

        i = 0
        for token in tokens:
            # get lemmas for each word
            lemma = lemmatizer.lemmatize(token, get_wordnet_pos(token))
            i+=1
            lemma_list.append(lemma)

            # get prefix/suffix features for each word
            prefs = []
            sufs = []
            if len(token) > 1 and re.match(r'^[a-zA-Z]+$', token):
                prefs = [token[:j] for j in range(1, 5) if j <= len(token)]
                sufs = [token[-j:] for j in range(1, 5) if j <= len(token)]

            for k in range(len(prefix_lists)):
                if k < len(prefs):
                    prefix_lists[k].append(prefs[k])
                else:
                    prefix_lists[k].append('')
            for k in range(len(suffix_lists)):
                if k < len(sufs):
                    suffix_lists[k].append(sufs[k])
                else:
                    suffix_lists[k].append('')

            # get capitalization and special character features
            if token[0].isupper():
                capitalized.append('True')
            else:
                capitalized.append('False')
            if not re.match(r'^\w+$', token):
                contains_special.append('True')
            else: 
                contains_special.append('False')

        # add all morphological features to dataframe
        self.data['Prefixes1'] = prefix_lists[0]
        self.data['Prefixes2'] = prefix_lists[1]
        self.data['Prefixes3'] = prefix_lists[2]
        self.data['Prefixes4'] = prefix_lists[3]
        self.data['Suffixes1'] = suffix_lists[0]
        self.data['Suffixes2'] = suffix_lists[1]
        self.data['Suffixes3'] = suffix_lists[2]
        self.data['Suffixes4'] = suffix_lists[3]
        self.data['Lemmas'] = lemma_list
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
        
        # get context features for each token
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

        #add all context features to dataframe
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