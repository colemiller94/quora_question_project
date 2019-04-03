import pandas as pd
import numpy as np
import spacy
import time
from fuzzywuzzy import fuzz
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import linear_model

import matplotlib.pyplot as plt
import seaborn as sns
#__________________intermediary/helper_functions_______________________#

def get_pos_tags(doc1,doc2,pos,as_vecs=False):
    '''
    doc1,doc2 : Spacy Document Objects
    pos: string Part-of-speech tag (ie 'noun','verb')

    returns: tuple of token lists (pseudo-document) filtered by pos tag
    '''

    if as_vecs:
        return [token.vector for token in doc1 if token.pos_ == pos.upper()],[token.vector for token in doc2 if token.pos_ == pos.upper()]
    return [token for token in doc1 if token.pos_ == pos.upper()],[token for token in doc2 if token.pos_ == pos.upper()]


def get_ents(doc_pair, as_vecs=False):
    '''
    doc_pair : tuple of lists of spacy tokens

    returns: filters both docs for tokens with entity tags
    '''
    return [token for token in doc_pair[0] if token.ent_type],[token for token in doc_pair[1] if token.ent_type]


def get_ent_types(doc_pair):
    '''
    doc_pair : tuple of lists of spacy tokens

    returns: entity types for both docs as tup of lists
    '''

    return [token.ent_type_ for token in doc_pair[0] if token.ent_type],[token.ent_type_ for token in doc_pair[1] if token.ent_type]


def count_toks(doc_pair,unique=False,doc2=False):
    '''
    doc_pair : tuple of lists of spacy tokens

    returns : total number of tokens in both docs. If unique, returns the
    number of unique tokens (as determined by lemma values)
    '''
    flatten = lambda l: [item.lemma_ for sublist in l for item in sublist]
    flat=flatten(doc_pair)

    if unique:
        return len(set(flat))
    return len(flat)

def count_txt_toks(doc_pair,unique=False):
    '''
    doc_pair : tuple of lists of spacy tokens
    Works the same as count_toks but for non-spacy tokens

    returns : total number of tokens in both docs. If unique, returns the
    number of unique tokens (as determined by lemma values)

    '''
    flatten = lambda l: [item for sublist in l for item in sublist]
    flat=flatten(doc_pair)

    if unique:
        return len(set(flat))
    return len(flat)

def get_common_words(doc_pair,as_vecs=False,use_toks=True):
    '''
    doc_pair : tuple of lists of spacy tokens

    returns: new list of common tokens, compared by lemma values if use_toks
    is true

    '''
    if use_toks:
        return [token for token in doc_pair[0] if token.lemma_ in [token.lemma_ for token in doc_pair[1]]]
    return [token for token in doc_pair[0] if token in [token for token in doc_pair[1]]]

def get_diff_words(doc_pair,as_vecs=False,use_toks=True):
    '''
    doc_pair : tuple of lists of spacy tokens

    returns: new doc pair of different tokens, compared by lemma values if use_toks
    is true

    '''
    if use_toks:
        in_1_not_2 = [token for token in doc_pair[0] if token.lemma not in [token.lemma for token in doc_pair[1]]]
        in_2_not_1 = [token for token in doc_pair[1] if token.lemma not in [token.lemma for token in doc_pair[0]]]
    else:
        in_1_not_2 = [token for token in doc_pair[0] if token not in [token for token in doc_pair[1]]]
        in_2_not_1 = [token for token in doc_pair[1] if token not in [token for token in doc_pair[0]]]

    if not as_vecs:
        return in_1_not_2,in_2_not_1
    else:
        return [token.vector for token in in_1_not_2],[token.vector for token in in_2_not_1]


def get_sims(doc_pair):
    '''
    doc_pair : tuple of lists of spacy tokens
    averages word vecs in each doc

    returns: cosine similarity of two docs
    '''
    if len(doc_pair[0])==0 or  len(doc_pair[1])==0:
        return 0
    mean_vecs = []
    for each in doc_pair:
        if len(each)<=1:
            mean_vecs.append(each)
        else:
            mean_vecs.append(mean_vec(each))

    return similarity(mean_vecs[0],mean_vecs[1])




def mean_vec(list_of_vecs,use_toks=False):
    '''extends functionality of spacy.doc.vector by averaging the vectors of component tokens'''
    if use_toks:
        return np.mean(np.array([x.vector for x in list_of_vecs]), axis=0)
    return np.mean(np.array([*list_of_vecs]), axis=0)

##__________________________________FEATURE_ENGINEERING_________________________________##

#___________________________________statistical semantic features
def ent_match_ratio(doc1,doc2,unique=False):
    '''Returns a basic fuzz.ratio of the entities in each document'''
    ents1,ents2=get_ents((doc1,doc2))
    weight = 1

    return .01*fuzz.ratio(ents1,ents2)*weight

def ent_type_match_ratio(doc1,doc2):
    '''Returns a basic fuzz.ratio of entity types (ie USA=>GPE for geopolitical entity)'''
    enttypes1,enttypes2=get_ent_types((doc1,doc2))

    return .01*fuzz.ratio(enttypes1,enttypes2)

def pos_match_ratio(doc1,doc2,pos):
    '''Returns a basic fuzz.ratio of given part of speech (ie verb) in each document'''

    pos=get_pos_tags(doc1,doc2,pos)

    return .01*fuzz.ratio(*pos)

#___________________________________purely semantic features

def similarity(vec1, vec2):
    '''Mimics spacy similarity for general vectors. Returns cosine similarity of two vecs'''
    vec12 = np.squeeze(vec1)
    vec22 = np.squeeze(vec2)
    if vec12.all() and vec22.all():
        return np.dot(vec12, vec22) / (np.linalg.norm(vec12) * np.linalg.norm(vec22))
    else:
        return 0

def sim_by_pos(doc1,doc2,pos):
    return get_sims(get_pos_tags(doc1,doc2,pos,as_vecs=True))

def sim_of_diffs(doc1,doc2,pos=None):
    if pos:
        return get_sims(get_diff_words(get_pos_tags(doc1,doc2,pos),as_vecs=True))
    return get_sims(get_diff_words((doc1,doc2),as_vecs=True))

lemmastr = lambda l: "".join([item.lemma_+" " for item in l]).strip()

##PARSE PIPELINE:
def parse(tup_of_docgens,keep_docs=True,keep_text=False):
    feat_dict = defaultdict(list)


    for q1,q2 in tqdm(tup_of_docgens):
        if keep_docs:
            feat_dict['q1_docs'].append(q1)
            feat_dict['q2_docs'].append(q2)
        if keept_text:
            feat_dict['q1_txt'].append(q1.text)
            feat_dict['q2_txt'].append(q2.text)


        feat_dict['sim'].append(q1.similarity(q2))
        feat_dict['sim_of_diffs'].append(sim_of_diffs(q1,q2))

        feat_dict['sim_of_nouns'].append(sim_by_pos(q1,q2,'noun'))
        feat_dict['sim_of_diffs_nouns'].append(sim_of_diffs(q1,q2,pos='noun'))
        feat_dict['noun_mratio'].append(pos_match_ratio(q1,q2,'noun'))

        feat_dict['sim_of_verbs'].append(sim_by_pos(q1,q2,'verb'))
        feat_dict['sim_of_diffs_verbs'].append(sim_of_diffs(q1,q2,pos='verb'))
        feat_dict['verb_mratio'].append(pos_match_ratio(q1,q2,'verb'))

        feat_dict['sim_of_adjectives'].append(sim_by_pos(q1,q2,'adj'))
        feat_dict['sim_of_diffs_adjectives'].append(sim_of_diffs(q1,q2,pos='adj'))
        feat_dict['adj_mratio'].append(pos_match_ratio(q1,q2,'adj'))

        feat_dict['sim_of_adverbs'].append(sim_by_pos(q1,q2,'adv'))
        feat_dict['sim_of_diffs_adverbs'].append(sim_of_diffs(q1,q2,pos='adj'))
        feat_dict['adv_mratio'].append(pos_match_ratio(q1,q2,'adv'))

        feat_dict['propn_mratio'].append(pos_match_ratio(q1,q2,'propn'))
        feat_dict['ent_ratio'].append(ent_match_ratio(q1,q2))
        feat_dict['ent_type_match_ratio'].append(ent_type_match_ratio(q1,q2))



    return pd.DataFrame(data=feat_dict)


##TESTING/CHECKING

def feature_sampler(index,df=df,y=y):
    test1,test2 = (df.q1_docs.loc[index],df.q2_docs.loc[index])
    print(test1)
    print(test2)
    print(y.loc[index])
    print('\n')
    print('sim: ', test1.similarity(test2))
    print('sods: ',sim_of_diffs(test1,test2))

    print('\n nouns:')
    print(get_pos_tags(test1,test2,'noun'))
    print('posmatchratio:',pos_match_ratio(test1,test2,'noun'))
    print('sim: ',sim_by_pos(test1,test2,'noun'))
    print('sods: ',sim_of_diffs(test1,test2,pos='noun'))

    print('\n verbs:')
    print(get_pos_tags(test1,test2,'verb'))
    print('posmatchratio:',pos_match_ratio(test1,test2,'verb'))
    print('sim:',sim_by_pos(test1,test2,'verb'))
    print('sods: ',sim_of_diffs(test1,test2,pos='verb'))

    print('\n adjectives:')
    print(get_pos_tags(test1,test2,'adj'))
    #print('has_adjs: ',both_have_pos((test1,test2),'adj'))
    print('posmatchratio:',pos_match_ratio(test1,test2,'adj'))
    print('sim:',sim_by_pos(test1,test2,'adj'))
    print('sods: ',sim_of_diffs(test1,test2,pos='adj'))

    print('\n adverbs:')
    print(get_pos_tags(test1,test2,'adv'))
    #print('has_adverbs: ',both_have_pos((test1,test2),'adv'))
    print('posmatchratio:',pos_match_ratio(test1,test2,'adv'))
    print('sim:',sim_by_pos(test1,test2,'adv'))
    print('sods: ',sim_of_diffs(test1,test2,pos='adv'))

    print('\n propns')
    print(get_pos_tags(test1,test2,'propn'))
   # print('has_propns: ',both_have_pos((test1,test2),'propn'))
    print('posmatchscore:',pos_match_ratio(test1,test2,'propn'))
    print('sim: ',sim_by_pos(test1,test2,'propn'))
    print('sods: ',sim_of_diffs(test1,test2,pos='propn'))

    print('\n ents')
    print(get_ents((test1,test2)))
    #print('has_ents: ',both_have_ents((test1,test2)))
    print(get_ent_types((test1,test2)))
    print('ent match ratio: ',ent_match_ratio(test1,test2))
    print('ent type match ratio: ',ent_type_match_ratio(test1,test2))
