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
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

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
    #print(len(sent_dfs))

    # create list of lists of IOB tags
    IOB_tags = [list(sent_df['IOB']) for sent_df in sent_dfs]

    # drop columns which do not represent features
    # (token, IOB, doc_ID, sent_ID, word_ID, data_type)
    ### INSERT CODE FOR DOING THIS THAT IS NOT SUPER SLOW
    columns_to_keep = []
    if 'grammatical' in feature_set:
        columns_to_keep.append('POS')
    if 'morphological' in feature_set:
        columns_to_keep.extend(['Capitalizations', 'Special Characters'])
    if 'context-based' in feature_set:
        columns_to_keep.extend(['Prev2', 'Prev', 'Next', 'Next2'])


    drop_list = []
    for col in data.columns:
        if col not in columns_to_keep:
            drop_list.append(col)
    #print(drop_list)

    sent_dfs = [sent_df.drop(drop_list, axis=1)
                             for sent_df in sent_dfs]

    #sent_dfs = [sent_df.drop(['token','IOB',
    #                         'doc_ID','sent_ID','word_ID',
    #                         'data_type'], axis=1)
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
    
    #pd.set_option('display.max_columns', None)
    #print(df.head(5))

    # convert features and IOB labels to expected python-crfsuite format
    train = df.loc[df['data_type']=='train']
    test = df.loc[df['data_type']=='test']
   
    #baseline
    X_train_base, y_train_base = format_df(train, ['context-based', 'morphological'])
    X_test, y_test = format_df(test, ['context-based', 'morphological'])
    ##print(X_train[:10])
    ##print(y_train[:10])

    #model 2
    X_train, y_train = format_df(train, ['grammatical', 'context-based', 'morphological'])
    #X_test, y_test = format_df(test, ['grammatical', 'context-based', 'morphological'])


    X_train_sets = [X_train_base, X_train]
    y_train_sets = [y_train_base, y_train]
    cross_validation_scores = [] #for multiple models with different feature sets

    
    # run 10-fold cross validation with crf
    for i in range(len(X_train_sets)):
         # train crf with default algorithm
        crf = sklearn_crfsuite.CRF(algorithm='lbfgs')
        f_scorer = make_scorer(metrics.flat_f1_score, average='weighted')
        cv_results = cross_validate(crf, X_train_sets[i], y_train_sets[i], scoring=f_scorer, cv=10, return_train_score=True)
        
        test_scores = cv_results['test_score']
        average_score = sum(test_scores)/len(test_scores)
        print('CRF ' + str(i) + ' cross validation scores: ' + str(test_scores))
        print('CRF ' + str(i) + ' average score: ' + str(average_score))
        cross_validation_scores.append(average_score)

    best_idx = cross_validation_scores.index(max(cross_validation_scores))

    #these should go in crf.fit() below
    best_model_X = X_train_sets[best_idx]
    best_model_y = y_train_sets[best_idx]

    # train crf with features that optimized cross validation f-score?
    crf = sklearn_crfsuite.CRF(algorithm='lbfgs')
    crf.fit(best_model_X, best_model_y)
    #crf.fit(X_train, y_train) 

    # generate predictions on test set
    labels = list(crf.classes_)
    y_pred = crf.predict(X_test)
    print('final eval f-score: ' + str(metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)))


if __name__ == "__main__":
    main()

