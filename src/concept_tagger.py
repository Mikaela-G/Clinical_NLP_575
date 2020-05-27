"""
Goals:
- Read in dataframe
- Extract features
- Convert features and IOB labels to expected python-crfsuite format
- Train CRF
- Evaluate CRF with 10-fold cross validation on training data
"""

import sklearn_crfsuite
from FeatureExtractor import *

### list of lists of dicts
### each outer list represents a sentence
### each inner list represents a word
### each word is represented by a feature dictionary