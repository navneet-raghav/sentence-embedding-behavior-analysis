import numpy as np
import re
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity

# Load pretrained embeddings
w2v_model = api.load("glove-wiki-gigaword-100")

def clean_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)
    return sentence

def sentence_vector(sentence):
    words = clean_sentence(sentence).split()
    word_vectors = []

    for word in words:
        if word in w2v_model.key_to_index:
            word_vectors.append(w2v_model[word])

    if len(word_vectors) == 0:
        return np.zeros(100)

    return np.mean(word_vectors, axis=0)

def word2vec_similarity(s1, s2):
    vec1 = sentence_vector(s1)
    vec2 = sentence_vector(s2)

    sim = cosine_similarity([vec1], [vec2])[0][0]
    return sim
