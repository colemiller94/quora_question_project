import pandas as pd
import numpy as np
import spacy

import time
from fuzzywuzzy import fuzz
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
#import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import linear_model
from collections import defaultdict
from tqdm import tqdm

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

##PREPARE PARSE PIPELINE:
def prepare(traindf):
    q1 = traindf['question1']
    q2 = traindf['question2']

    q1_it = iter(q1)
    q2_it = iter(q2)

    q1_docs_ =  nlp.pipe(q1_it) #nlp.pipe takes an iterator and returns a generator that preforms spacy pipeline

    q2_docs_ = nlp.pipe(q2_it)
    return zip(q1_docs_,q2_docs_)

def parse(tup_of_docgens,keep_docs=True,keep_text=False):
    feat_dict = defaultdict(list)
    pos_list = ['noun','verb','adj','adv']

    for q1,q2 in tqdm(tup_of_docgens):
        if keep_docs:
            feat_dict['q1_docs'].append(q1)
            feat_dict['q2_docs'].append(q2)
        if keep_text:
            feat_dict['q1_txt'].append(q1.text)
            feat_dict['q2_txt'].append(q2.text)


        feat_dict['sim'].append(q1.similarity(q2))
        feat_dict['sim_of_diffs'].append(sim_of_diffs(q1,q2))
        for pos in pos_list:

            feat_dict[f'sim_of_{pos}s'].append(sim_by_pos(q1,q2,pos))
            feat_dict[f'sim_of_diffs_{pos}s'].append(sim_of_diffs(q1,q2,pos=pos))
            feat_dict[f'{pos}_mratio'].append(pos_match_ratio(q1,q2,pos))


        feat_dict['propn_mratio'].append(pos_match_ratio(q1,q2,'propn'))
        feat_dict['ent_ratio'].append(ent_match_ratio(q1,q2))
        feat_dict['ent_type_match_ratio'].append(ent_type_match_ratio(q1,q2))



    return pd.DataFrame(data=feat_dict)


##TESTING/CHECKING

def feature_sampler(index,df=None,y=None):
    test1,test2 = (df.q1_docs.loc[index],df.q2_docs.loc[index])
    pos_list = ['noun','verb','adj','adv']
    print(test1)
    print(test2)
    print(y.loc[index])
    print('\n')
    print('similarities: ', test1.similarity(test2))
    print('similarity of differences: ',sim_of_diffs(test1,test2))

    for pos in pos_list:

        print('\n _______f{pos}s_______')
        print(get_pos_tags(test1,test2,pos))
        print('___________________________')
        print(f'{pos}_matchratio: ',pos_match_ratio(test1,test2,pos))
        print('sim: ',sim_by_pos(test1,test2,pos))
        print('s.o.d.s: ',sim_of_diffs(test1,test2,pos=pos))

    print('\n _______Proper_Nouns_______')
    print(get_pos_tags(test1,test2,'propn'))
    print('___________________________')
   # print('has_propns: ',both_have_pos((test1,test2),'propn'))
    print('propn_matchratio:',pos_match_ratio(test1,test2,'propn'))
    print('sim: ',sim_by_pos(test1,test2,'propn'))
    print('s.o.d.s: ',sim_of_diffs(test1,test2,pos='propn'))

    print('\n _______Entities_______')
    print(get_ents((test1,test2)))
    print('___________________________')
    print('ent_matchratio: ',ent_match_ratio(test1,test2))
    #print('has_ents: ',both_have_ents((test1,test2)))


    print('\n _______Entity_Types_______')
    print(get_ent_types((test1,test2)))
    print('___________________________')
    print('ent_type_match ratio: ',ent_type_match_ratio(test1,test2))
