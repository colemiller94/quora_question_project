import pandas as pd
import numpy as np
import spacy




def get_pos_tags(doc1,doc2,pos,as_vecs=False):
    if as_vecs:
        return [token.vector for token in doc1 if token.pos_ == pos.upper()],[token.vector for token in doc2 if token.pos_ == pos.upper()]
    return [token for token in doc1 if token.pos_ == pos.upper()],[token for token in doc2 if token.pos_ == pos.upper()]


def get_ents(doc1,doc2=False,as_vecs=False):
    if type(doc1)==tuple:
        doc1,doc2 = doc1

    return [token for token in doc1 if token.ent_type],[token for token in doc2 if token.ent_type]


def get_ent_types(tup_of_docs):
    return [token.ent_type_ for token in tup_of_docs[0] if token.ent_type],[token.ent_type_ for token in tup_of_docs[1] if token.ent_type]


def count_toks(tup_of_docs,unique=False,doc2=False):

    flatten = lambda l: [item.lemma_ for sublist in l for item in sublist]
    flat=flatten(tup_of_docs)

    if unique:
        return len(set(flat))
    return len(flat)

def count_txt_toks(tup_of_docs,unique=False):
    flatten = lambda l: [item for sublist in l for item in sublist]
    flat=flatten(tup_of_docs)

    if unique:
        return len(set(flat))
    return len(flat)




def get_common_words(doc1,doc2=False,as_vecs=False,use_toks=True):
    if type(doc1)==tuple:
        doc1,doc2 = doc1
    if use_toks:
        return [token for token in doc1 if token.lemma in [token.lemma for token in doc2]]
    return [token for token in doc1 if token in [token for token in doc2]]



def get_diff_words(doc1,doc2=False,as_vecs=False):
    if type(doc1)==tuple:
        doc1,doc2 = doc1

    in_1_not_2 = [token for token in doc1 if token.lemma not in [token.lemma for token in doc2]]
    in_2_not_1 = [token for token in doc2 if token.lemma not in [token.lemma for token in doc1]]

    if not as_vecs:
        return in_1_not_2,in_2_not_1
    else:
        output1,output2 =  [token.vector for token in in_1_not_2],[token.vector for token in in_2_not_1]

        return output1,output2

def get_sims(tup_of_doc):

    if len(tup_of_doc[0])==0 or  len(tup_of_doc[1])==0:

        return 0

    mean_vecs = []
    for each in tup_of_doc:
        if len(each)<=1:
            mean_vecs.append(each)
        else:
            mean_vecs.append(mean_vec2(each))


    return similarity2(mean_vecs[0],mean_vecs[1])


def mean_vec(toks):
    return np.mean(np.array([x.vector for x in toks]), axis=0)

def mean_vec2(list_of_vecs):
    if type(list_of_vecs[1])=='spacy.tokens.token.Token':
        return mean_vec(list_of_vecs)
    return np.mean(np.array([*list_of_vecs]), axis=0)


def similarity2(vec1, vec2):
    vec12 = np.squeeze(vec1)
    vec22 = np.squeeze(vec2)
    if vec12.all() and vec22.all():
        return np.dot(vec12, vec22) / (np.linalg.norm(vec12) * np.linalg.norm(vec22))
    else:
        return 0




def ent_match_ratio(doc1,doc2,unique=False):
    num = len(get_common_words(get_ents((doc1,doc2))))
    denom = count_toks(get_ents((doc1,doc2)),unique=unique)
    if denom>0:
        return 2*num/denom
    return 0

def ent_type_match_ratio(doc1,doc2,unique=False):
    num= len(get_common_words(get_ent_types((test1,test2)),use_toks=False))
    denom = count_txt_toks(get_ent_types((doc1,doc2)),unique=unique)
    if denom>0:
        return 2*num/denom
    return 0

def pos_match_ratio(doc1,doc2,pos,unique=False):
    num = len(get_common_words(get_pos_tags(q1_docs[index],q2_docs[index],pos=pos)))
    denom = count_toks(get_pos_tags(q1_docs[index],q2_docs[index],pos=pos))

    if denom>0:
        return 2*num/denom
    return 0

def sim_by_pos(doc1,doc2,pos):
    return get_sims(get_pos_tags(doc1,doc2,pos,as_vecs=True))

def sim_of_diffs_by_pos(doc1,doc2,pos):
    return get_sims(get_diff_words(get_pos_tags(doc1,doc2,pos),as_vecs=True))
