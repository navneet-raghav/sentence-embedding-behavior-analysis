from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


vectorizer = TfidfVectorizer()

def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)

def preprocess(text):
    text = text.lower()
    words = word_tokenize(text)
    
    cleaned_words = []
    for word in words:
        if word.isalpha() and word not in stop_words:
            pos = get_wordnet_pos(word)
            lemma = lemmatizer.lemmatize(word, pos)
            cleaned_words.append(lemma)
    
    return " ".join(cleaned_words)

def initialize_tfidf(corpus):
    processed_corpus = [preprocess(sentence) for sentence in corpus]
    vectorizer.fit(processed_corpus)

def tfidf_similarity(s1, s2):
    p1 = preprocess(s1)
    p2 = preprocess(s2)
    
    tfidf_matrix = vectorizer.transform([p1, p2])
    sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    return sim
