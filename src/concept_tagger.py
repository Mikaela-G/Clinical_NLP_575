#!/usr/bin/env python3
"""
Goals:
- Read in dataframe
- Extract features
- Convert features and IOB labels to expected python-crfsuite format
- Train CRF
- Evaluate CRF with 10-fold cross validation on training data
- Use best model to generate predictions and confusion matrices for train and test
"""

from FeatureExtractor import *
import pandas as pd
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, confusion_matrix

def format_df(data, feature_set):
    """
    Convert features and IOB labels to expected python-crfsuite format.

    :param data: Pandas dataframe
    :param feature_set: str that declares which feature set(s) to use
    """
    # group data
    grouped = data.groupby(['doc_ID', 'sent_ID'])
    
    # create list with one dataframe per sentence
    sent_dfs = [sent for _, sent in grouped]

    # create list of lists of IOB tags
    IOB_tags = [list(sent_df['IOB']) for sent_df in sent_dfs]

    # drop columns which do not represent features
    # (token, IOB, doc_ID, sent_ID, word_ID, data_type)
    # and keep columns depending on feature set selected
    columns_to_keep = []
    if 'grammatical' in feature_set:
        columns_to_keep.append('POS')
    if 'morphological' in feature_set:
        columns_to_keep.extend(['Prefixes1', 'Prefixes2', 'Prefixes3', 'Prefixes4', 'Suffixes1', 'Suffixes2', 'Suffixes3', 'Suffixes4', 'Capitalizations', 'Special Characters'])
    if 'context-based' in feature_set:
        columns_to_keep.extend(['Prev2', 'Prev', 'Next', 'Next2'])
    if 'lemma' in feature_set:
        columns_to_keep.append('Lemmas')
    if 'metamap' in feature_set:
        columns_to_keep.append('MetaMap')

    drop_list = []
    for col in data.columns:
        if col not in columns_to_keep:
            drop_list.append(col)

    sent_dfs = [sent_df.drop(drop_list, axis=1)
                             for sent_df in sent_dfs]

    # create list of lists of feature dicts
    feat_dicts = [sent_df.to_dict('records') for sent_df in sent_dfs]

    return feat_dicts, IOB_tags

def main():
    # read in dataframe
    df = pd.read_pickle('df.pkl')
    # extract features and store in original dataframe
    features = FeatureExtractor(df)
    df = features.data

    # convert features and IOB labels to expected python-crfsuite format
    train = df.loc[df['data_type']=='train']
    test = df.loc[df['data_type']=='test']
   
    # baseline -> preliminary features
    X_train_base, y_train_base = format_df(train, ['context-based', 'morphological'])
    X_test, y_test = format_df(test, ['context-based', 'morphological'])

    # model 2 -> preliminary features + POS
    X_train2, y_train2 = format_df(train, ['grammatical', 'context-based', 'morphological'])

    # model 3 -> preliminary features + POS + Lemma
    X_train3, y_train3 = format_df(train, ['grammatical', 'context-based', 'morphological', 'lemma'])

    # model 4 -> preliminary features + POS + Lemma + MetaMap-based
    X_train4, y_train4 = format_df(train, ['grammatical', 'context-based', 'morphological', 'lemma', 'metamap'])
    
    # all models
    X_train_sets = [X_train_base, X_train2, X_train3, X_train4]
    y_train_sets = [y_train_base, y_train2, y_train3, y_train4]
    cross_validation_scores = [] #for multiple models with different feature sets

    # run 10-fold cross validation with crf
    for i in range(len(X_train_sets)):

         # train crf with default algorithm
        crf = sklearn_crfsuite.CRF(algorithm='lbfgs')
        f_scorer = make_scorer(metrics.flat_f1_score, average='weighted')
        cv_results = cross_validate(crf, X_train_sets[i], y_train_sets[i], scoring=f_scorer, cv=10, return_train_score=True)
        
        # print cross validation scores
        test_scores = cv_results['test_score']
        average_score = sum(test_scores)/len(test_scores)
        print('CRF ' + str(i) + ' cross validation scores: ' + str(test_scores))
        print('CRF ' + str(i) + ' average score: ' + str(average_score))
        cross_validation_scores.append(average_score)

    best_idx = cross_validation_scores.index(max(cross_validation_scores))

    # select model with best feature combination to go in crf.fit() below
    best_model_X = X_train_sets[best_idx]
    best_model_y = y_train_sets[best_idx]

    # train crf with features that optimized cross validation f-score
    crf = sklearn_crfsuite.CRF(algorithm='lbfgs')
    crf.fit(best_model_X, best_model_y)

    # get ordered list of classes to be predicted by CRF
    labels = list(crf.classes_)
    print('\n')
    print('ORDER OF LABELS IN CONFUSION MATRIX: ' + ' '.join(labels) + '\n')

    # generate predictions and confusion matrix on train set (using best model features)
    y_pred_train = crf.predict(best_model_X)
    print('CONFUSION MATRIX FOR BEST MODEL ON TRAIN SET:')
    print(confusion_matrix(best_model_y, y_pred_train, labels=labels))
    print('\n')

    # generate predictions and confusion matrix on test set
    ##labels = list(crf.classes_) ## temporarily commented out, moved upwards, delete this one?
    y_pred = crf.predict(X_test)
    print('final eval f-score: ' + str(metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)))
    print('\n')
    print('CONFUSION MATRIX FOR BEST MODEL ON TEST SET:')
    print(confusion_matrix(y_test, y_pred, labels=labels))
    print('\n')

    # evaluate best model performance on all tags
    print('ALL CLASSES:')
    print(metrics.flat_classification_report(y_test, y_pred, labels=labels))
    print('\n')

    # evaluate best model performance on problem, treatment, and test (B and I tags combined)
    combined_labels = ['problem', 'treatment', 'test']
    combined_y_test = [[tag.replace('B-', '') for tag in sent] for sent in y_test]
    combined_y_test = [[tag.replace('I-', '') for tag in sent] for sent in combined_y_test]
    combined_y_pred = [[tag.replace('B-', '') for tag in sent] for sent in y_pred]
    combined_y_pred = [[tag.replace('I-', '') for tag in sent] for sent in combined_y_pred]    
    print('COMBINED CLASSES:')
    print(metrics.flat_classification_report(combined_y_test, combined_y_pred, labels=combined_labels))

if __name__ == "__main__":
    main()

