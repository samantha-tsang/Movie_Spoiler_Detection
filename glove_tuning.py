import pandas as pd
import numpy as np
import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from collections import Counter
from mittens import Mittens
from sklearn.feature_extraction.text import CountVectorizer
import pickle

# import training data
X_train = pd.read_csv("X_train.csv")

###### Fine tuning GloVe
# source: https://towardsdatascience.com/fine-tune-glove-embeddings-using-mittens-89b5f3fe4c39

# import pretained embeddings. We will use the GloVe modeel trained on 400K vocab size with 6M tokens. We will use the 100-dim one
def glove2dict(glove_filename):
    with open(glove_filename, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=' ',quoting=csv.QUOTE_NONE)
        embed = {line[0]: np.array(list(map(float, line[1:])))
                for line in reader}
    return embed
glove_path = "glove.6B.100d.txt"
pre_glove = glove2dict(glove_path)

# next we preprocess the training text

# first we tokenize entire corpus
def tokenize_dataframe_column(df, column):
    """
    Tokenize a specified column of a pandas DataFrame.

    Args:
    df (pd.DataFrame): The DataFrame containing the text data.
    column (str): The name of the column to be tokenized.

    Returns:
    List[str]: A list of tokens.
    """
    # Concatenate all text from the column
    text = " ".join(df[column].dropna())

    # remove symbols in text
    def remove_symbols(text):
        return ''.join(char for char in text if char not in string.punctuation)

    # Tokenize the text
    text = remove_symbols(text)
    tokens = word_tokenize(text)

    return tokens

def tokenize_dataframe_column(df, column):
    """
    Tokenize a specified column of a pandas DataFrame.

    Args:
    df (pd.DataFrame): The DataFrame containing the text data.
    column (str): The name of the column to be tokenized.

    Returns:
    List[str]: A list of tokens.
    """
    # Concatenate all text from the column
    text = " ".join(df[column].dropna())

    # remove symbols in text
    def remove_symbols(text):
        return ''.join(char for char in text if char not in string.punctuation)

    # Tokenize the text
    text = remove_symbols(text)
    tokens = word_tokenize(text)

    return tokens
review_tokens = tokenize_dataframe_column(X_train, "combined_text")

# then we remove stop words from the corpus
sw = set(stopwords.words("english"))
review_nonstop = [token.lower() for token in review_tokens if (token.lower() not in sw)]

# define words that are not already in the GloVe pretrained
oov = [token for token in review_nonstop if token not in pre_glove.keys()]

# next, we remove the words that only appeared once in the entire corpus. This will help save space in the next steps
def get_rareoov(xdict, val):
    return [k for (k,v) in Counter(xdict).items() if v<=val]
oov_rare = get_rareoov(oov, 1)

# define the set of vocab unique to our corpus, and the entire document
corp_vocab = list(set(oov) - set(oov_rare))
review_doc = [' '.join(review_nonstop)]

# NOTE: stemming and lemmatization will not be applied to the preprocessing. Since the GloVe odel is originally trained on
# words that did not go through this preprocessing, we will continue with this format to prevent mismatch

# next we create the word-word co-occurence matrix
cv = CountVectorizer(ngram_range=(1,1), vocabulary=corp_vocab)
X = cv.fit_transform(review_doc)
Xc = (X.T * X)
Xc.setdiag(0)
coocc_ar = Xc.toarray()

mittens_model = Mittens(n=50, max_iter=1000)

new_embeddings = mittens_model.fit(
    coocc_ar,
    vocab=corp_vocab,
    initial_embedding_dict= pre_glove)

# combine the original glove dict and the one with the unseen words, then save it
newglove = dict(zip(corp_vocab, new_embeddings))
review_glove = pre_glove.update(newglove)




##### Calculating DF-IIF for each word
# source: https://cseweb.ucsd.edu/~jmcauley/pdfs/acl19.pdf
# DF-IIF is similar to TF-IDF, which calculates the frequency of a word in terms of a document and the inverse frequency of that term in the whole corpus
# DF-IIF slighly changes the calculation, where instead of calculating the frequency of a word in one document, we calculate the frequency of the word in all documents relating to item i (ie a movie
# This is due to the fact that not all documents are independent in context here (ie, there are multiple reviews of one mocie) so it is more suitable to calculate word specificity in relation to item instead of document

import re
import nltk
from collections import Counter
from collections import defaultdict
import scipy.sparse as sp
from numpy.linalg import norm


def get_df_iif(review_df, item_list, epsilon):
    # split up entire corpus into reviews
    corpus = review_df['combined_text']
    words_array = [review.split() for review in corpus]

    # further split corpus up into list of unique qords
    words = list(set([word for words in words_array for word in words]))

    # for each word in words, set up the dictionary
    features_dict = {w: 0 for w in words}

    ##### CALCULATE DF #####
    # for each movie, set up dict to count how many reviews for each movie
    movie_dict = {m: 0 for m in item_list}

    df = defaultdict(dict)
    # count how many reviews for each movie
    for movie in item_list:
        num_reviews = len(review_df[review_df['movie_id'] == movie])
        movie_dict[movie] = num_reviews

    # count how many reviews contain each word
    for word in words:
        features_dict[word] = sum(word in review for review in corpus)

    for movie in item_list:
        # this is the denominator of the df ter
        den = movie_dict[movie]

        # subset rows that are reviews of movie, and create its own corpus
        df = review_df[review_df["movie_id"] == movie]['combined_text']
        movie_reviews = [review.split() for review in corpus]

        # for each word that is in any reviews on movie i, calculate the df metric (for that movie)
        for word in words:
            if word in movie_reviews:
                df[movie][word] = features_dict[word] / movie_dict[movie]

    ##### CALCULATE IIF #####

    # create default empty dict to store iif values for each word
    # calculate number of movies total
    iif = defaultdict(dict)
    num_movies = len(set(review_df['movie_id']))

    # calculate number of movies with reviews containing each word
    word_dict = {w:0 for w in words}
    for movie in item_list:
        # list of unique words in reviews for this movie
        seen_words = set()
        df = review_df[review_df['movie_id'] == movie]['combined_text']
        for review in df:
            # update the seen words set to any new words that appeared in this review
            seen_words.update(review.split())
        for word in seen_words:
            # update the counter for how many movies has reviews containing word
            word_dict[word] += 1
    for word in words:
        iif[word] = np.log((epsilon + num_movies)/(epsilon + word_dict[word]))

    ##### Finally, calculate the df-iif metric for each word for each review
    df_iif = defaultdict(dict)
    for movie in item_list:
        for word in words:
            try:
                df_iif[movie][word] = df[movie][word] * iif[word]

            # if the movie doesnt have that word, then skip to next word
            except KeyError:
                continue
    return pd.DataFrame(df_iif)

##### CREATE EMBEDDING DICTIONARY #####
# each movie gets a key, and for each word in that movie, it will have an embedding that it the word embedding from the fine tuned GloVe
# plus the item specificity term from the dict

# first we get the item specificity score from the above function
movies = set(X_train['movie_id'])
df_iif = get_df_iif(X_train, movies, 1)

embedding_dict = defaultdict(dict)
for movie in movies:
    for word in review_glove.keys():
        try:
            embedding_dict[movie][word] = review_glove[word] + df_iif[movie][word]
        # if word not in any reviews for this movie, continue to loop through next word
        except KeyError:
            continue

##### embedding dict is the final dictionary we will use to get the embedding input for each word


f = open("repo_glove.pkl","wb")
pickle.dump(review_glove, f)
f.close()