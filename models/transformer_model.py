from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def transformer_similarity(s1, s2):
    embeddings = model.encode([s1, s2])
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return sim
