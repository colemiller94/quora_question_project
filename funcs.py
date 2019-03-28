import pandas as pd
import numpy as np
import spacy







def get_pos_tags(doc1,doc2,pos,as_vecs=False):
    if as_vecs:
        return [token.vector for token in doc1 if token.pos_ == pos.upper()],[token.vector for token in doc2 if token.pos_ == pos.upper()]
    return [token for token in doc1 if token.pos_ == pos.upper()],[token for token in doc2 if token.pos_ == pos.upper()]

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

        return np.nan

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

    return np.dot(vec12, vec22) / (np.linalg.norm(vec12) * np.linalg.norm(vec22))


def similarity(self, other):
    """
    !!direct from spacy!!
    Compute a semantic similarity estimate. Defaults to cosine over vectors.
    Arguments:
        other:
            The object to compare with. By default, accepts Doc, Span,
            Token and Lexeme objects.
    Returns:
        score (float): A scalar similarity score. Higher is more similar.
    """
    if 'similarity' in self.doc.user_token_hooks:
            return self.doc.user_token_hooks['similarity'](self)
    if self.vector_norm == 0 or other.vector_norm == 0:
        return 0.0
    return numpy.dot(self.vector, other.vector) / (self.vector_norm * other.vector_norm)


def sim_by_pos(doc1,doc2,pos):
    return get_sims(get_pos_tags(doc1,doc2,pos,as_vecs=True))

def sim_of_diffs_by_pos(doc1,doc2,pos):
    return get_sims(get_diff_words(get_pos_tags(doc1,doc2,pos),as_vecs=True))
