# Sentence Similarity Behaviour Study

# Overview
This project compares three different approaches to sentence similarity: 
1. TF-IDF
2. Word2Vec 
3. transformer-based MiniLM model

Instead of focusing only on similarity scores, the aim here is to understand how these models behave when sentences are slightly modified. 

I created a controlled dataset of 100 sentence pairs across different linguistic categories (such as synonyms, negation, antonyms, and word order changes) to observe where each model performs well and where it fails.

The goal is to study representation behavior and not just accuracy.



# Motivation
Sentence embeddings are widely used in applications like semantic search, question answering, and text matching. However, high similarity between two sentences does not necessarily mean they express the same meaning.

For example, sentences that differ only by negation eg- “I like this movie” vs “I do not like this movie” may still receive very high similarity scores.

This project is built to examine such cases systematically and compare how lexical, static embedding, and contextual embedding models respond to controlled linguistic changes.



# Models Compared

## 1. TF-IDF (Lexical Baseline)
This model represents sentences using a bag-of-words approach with TF-IDF weighting. 
Preprocessing includes tokenization, stopword removal, and lemmatization.

It mainly captures surface-level word overlap.

## 2. Word2Vec (Static Embeddings)
Pretrained GloVe embeddings (100-dimensional) are used to obtain word vectors. 
Sentence vectors are computed by taking the average of the word embeddings.

This approach captures distributional similarity, meaning words that appear in similar contexts are placed close together in vector space.

## 3. Transformer (MiniLM)
The transformer based model ('all-MiniLM-L6-v2') generates contextual sentence embeddings. 
Unlike static embeddings, it considers the entire sentence while generating representations.

Similarity is computed using cosine similarity across all models.

# Dataset

To evaluate model behaviour in a controlled way, I created a custom dataset of 100 sentence pairs.  
Each pair is manually written to isolate a specific linguistic phenomenon.

The dataset is divided into the following categories:

- Exact lexical match (10 pairs)
- Synonym substitution (15 pairs)
- Antonym presence (15 pairs)
- Negation handling (15 pairs)
- Word order variation (10 pairs)
- Abbreviations and expansions (10 pairs)
- Domain shift (10 pairs)
- Same topic, different stance (15 pairs)

Each category tests a different aspect of sentence representation.

For example:

- Synonyms: test whether models can recognize meaning beyond exact word overlap.
- Antonyms: test whether models incorrectly group opposite meanings due to contextual     similarity.
- Negation: examines whether models capture logical polarity.
- Word order variation: checks if role reversal affects similarity.
- Same topic, different stance: evaluates whether models confuse topic similarity with agreement.

The intention is not to create a benchmark dataset, but to design a small, controlled experimental dataset that reveal specific strengths and weaknesses of each embedding approach across different categories.

# Experimental Setup

All three models are evaluated on the same dataset.

For TF-IDF, a global vocabulary is first built using all sentences in the dataset to ensure consistency across categories. 
Then, similarity is computed using cosine similarity between the TF-IDF vectors.

For Word2Vec, pretrained 100-dimensional GloVe embeddings are used. 
Each sentence is represented by averaging the embeddings of its words. 
Cosine similarity is then computed between sentence vectors.

For the transformer model, the "all-MiniLM-L6-v2" sentence-transformer is used to generate contextual sentence embeddings. 
Similarity is again computed using cosine similarity.

Cosine similarity between two vectors A and B is defined as:

(A·B)/(||A||×||B||)

For each category, similarity scores are computed for all sentence pairs, and then the average similarity is calculated to analyze overall behaviour.

# Results Summary

The results show clear behavioural differences between the three models.

## Exact Match

- All models produce similarity scores very close to 1.0
- This confirms that the implementations are functioning correctly.
- It also verifies that cosine similarity behaves as expected when two sentences are identical.

## Synonyms

- TF-IDF often assigns very low similarity because it relies on direct word overlap.
- Word2Vec generally assigns high similarity scores.
- The transformer model also assigns high similarity in most cases.

This indicates that embedding-based models capture semantic similarity beyond surface form, whereas TF-IDF remains strictly lexical.

## Antonyms


The antonym category produces one of the most interesting observations.

- Even when the meaning is opposite (for example, 'good' vs 'bad'), Word2Vec still assigns very high similarity scores, often above 0.95.

- The transformer model also gives high similarity in most cases.

- TF-IDF sometimes produces moderate similarity due to shared words in the sentence.

This suggests that embedding similarity reflects contextual relatedness rather than logical opposition.

## Negation

Negation turns out to be particularly challenging.

- TF-IDF often gives a similarity score of 1.0 because the word "not" is removed during preprocessing. As a result, the negated sentence looks identical to the original in vector space.

- Word2Vec also assigns very high similarity, and although the transformer model gives slightly lower scores in some cases, they are still relatively high overall.

This shows that none of the models truly understand logical polarity; they mainly measure semantic closeness.

## Word Order Variation


When the subject and object are swapped (eg- 'The cat chased the mouse' vs 'The mouse chased the cat'), all models still produce very high similarity scores.

- TF-IDF and Word2Vec are almost unaffected because the same words are present.

- The transformer model shows a small drop in similarity but still remains high.

This indicates that sentence embedding similarity does not strongly capture role reversal, rather it focuses more on the contextual similarity.


## Domain Shift

- In the domain shift category, TF-IDF consistently produces low similarity scores (mostly below 0.35), which is expected because there is little word overlap between sentences.

- Word2Vec assigns noticeably higher similarity scores in many cases (between 0.62 and 0.90). Even when the topics are different, shared verbs seem to keep the sentence vectors relatively close.

- The transformer model generally assigns lower similarity scores than Word2Vec in this category (sometimes as low as 0.07 and often below 0.40), although a few pairs still reach moderate values.

Overall, the transformer appears more sensitive to contextual differences across domains compared to static embeddings.


## Same Topic, Different Stance

In this category, sentences clearly discuss the same topic but express opposing opinions.

- TF-IDF produces moderate similarity scores (roughly between 0.40 and 0.72), largely depending on how many words overlap.

- Word2Vec consistently assigns high similarity (often above 0.90), even when the stance is opposite.

- The transformer model also assigns relatively high similarity in most cases (commonly between 0.68 and 0.96), though it shows slightly more variation than Word2Vec.

These results suggest that embedding-based similarity primarily captures topical closeness rather than agreement in meaning. Opposing viewpoints on the same subject still remain close in embedding space.

# Project Structure

- `models/` – similarity model implementations  
- `data/` – dataset and evaluation results  
- `experiments/` – evaluation notebook   
- `README.md` – project overview  


# How to Run

1. Clone the repository.

2. Install the required dependencies:
   - scikit-learn  
   - nltk  
   - gensim  
   - sentence-transformers  
   - tqdm  

3. Download required NLTK resources:
   - punkt
   - stopwords
   - wordnet
   - averaged_perceptron_tagger

4. Open and run:

   experiments/run_evaluation.ipynb

The notebook:

- Initializes the TF-IDF vocabulary
- Computes similarity scores for all three models
- Saves results to `data/evaluation_results.json`
- Computes category-wise averages


# Key Insights

- TF-IDF measures lexical overlap, not meaning.
- Word2Vec captures semantic relatedness but fails on logical opposition.
- Transformer embeddings improve contextual sensitivity but still do not model reasoning.
- Negation and stance remain difficult for all embedding-based similarity models.
- High cosine similarity does not imply agreement or logical equivalence.

# Future Work

This study can be extended in several directions:

- Expanding the dataset with a larger number of controlled perturbations to enable more robust statistical analysis.
- Evaluating larger and more recent transformer-based sentence embedding models.
- Incorporating logical reasoning benchmarks to examine polarity and contradiction handling.
- Comparing cosine similarity with alternative similarity measures or learned scoring  functions.

These extensions would help better understand the gap between semantic proximity and logical equivalence in embedding-based similarity models.
