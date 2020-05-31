#!/usr/bin/env python3
"""
Goals:
- Read in dataframe
- Extract features
- Convert features and IOB labels to expected python-crfsuite format
- Train CRF
- Evaluate CRF with 10-fold cross validation on training data
"""

from FeatureExtractor import *
import pandas as pd
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

def format_df(data, feature_set):
    """
    Convert features and IOB labels to expected python-crfsuite format.

    :param data: Pandas dataframe
    :param feature_set: str that declares which feature set(s) to use
    """
    # drop feature columns depending on feature set(s) chosen
    ### INSERT CODE FOR DOING THIS

    # group data
    grouped = data.groupby(['doc_ID', 'sent_ID'])

    # create list with one dataframe per sentence
    sent_dfs = [sent for _, sent in grouped]

    # create list of lists of IOB tags
    IOB_tags = [list(sent_df['IOB']) for sent_df in sent_dfs]

    # drop columns which do not represent features
    # (token, IOB, doc_ID, sent_ID, word_ID, data_type)
    ### INSERT CODE FOR DOING THIS THAT IS NOT SUPER SLOW
    # sent_dfs = [sent_df.drop(['token','IOB',
    #                         'doc_ID','sent_ID','word_ID',
    #                         'data_type'], axis=1, inplace=True)
    #                         for sent_df in sent_dfs]

    # create list of lists of feature dicts
    feat_dicts = [sent_df.to_dict('records') for sent_df in sent_dfs]

    return feat_dicts, IOB_tags

def main():
    # read in dataframe
    df = pd.read_pickle('df.pkl')
    # extract features and store in original dataframe
    features = FeatureExtractor(df)
    df = features.data 

    ##print(df.head(5))

    # convert features and IOB labels to expected python-crfsuite format
    train = df.loc[df['data_type']=='train']
    test = df.loc[df['data_type']=='test']
    ### format_df creates memory errors right now because dictionaries are too big
    ### instead of current structure, load dictionaries using generator?
    X_train, y_train = format_df(train, 'grammatical')
    X_test, y_test = format_df(test, 'grammatical')
    ##print(X_train[:10])
    ##print(y_train[:10])

    # train crf with default algorithm
    crf = sklearn_crfsuite.CRF(algorithm='lbfgs')
    crf.fit(X_train, y_train)

    # generate predictions
    labels = list(crf.classes_)
    y_pred = crf.predict(X_test)
    print(metrics.flat_f1_score(y_test, y_pred,
                      average='weighted', labels=labels))

if __name__ == "__main__":
    main()