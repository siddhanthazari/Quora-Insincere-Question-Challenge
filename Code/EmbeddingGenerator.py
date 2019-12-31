import numpy as np
import pandas as pd
import nltk
from tqdm import tqdm
import pickle
from nltk.stem.wordnet import WordNetLemmatizer
import unidecode
import contractions
import re
import os
import gensim


def clean_input(data):
    if os.path.isfile('clean_data.pickle'):
        with open('clean_data.pickle', 'rb') as handle:
            cleaned = pickle.load(handle)
        return cleaned

    print("Cleaning data...")
    print()
    cleaned = []
    for unclean_sentence in tqdm(data):
        clean = unidecode.unidecode(unclean_sentence)
        clean = contractions.fix(clean)
        clean = clean.lower()
        clean = re.sub('\W+', ' ', clean)
        cleaned.append(clean)
    with open('clean_data.pickle', 'wb') as handle:
        pickle.dump(cleaned, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return cleaned


def tokenize_sentence(data):
    if os.path.isfile('sent_tokens.pickle'):
        with open('sent_tokens.pickle', 'rb') as handle:
            sent_tokens = pickle.load(handle)
        return sent_tokens

    print("Tokenizing sentences...")
    print()
    sent_tokens = []
    for sentence in tqdm(data):
        sent_tokens.append(nltk.word_tokenize(sentence))
    with open('sent_tokens.pickle', 'wb') as handle:
        pickle.dump(sent_tokens, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return sent_tokens


def generate_embeddings(data):
    lmtzr = WordNetLemmatizer()
    ps = nltk.stem.PorterStemmer()

    print("Loading model...")
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format('lexvec.commoncrawl.300d.W.pos.vectors.gz', binary=False)

    print("Generating embeddings...")
    embeddings = []
    unknowns = []
    for sentence in tqdm(data):
        mean_vector = np.zeros(300)
        counts = 0
        for word in sentence:
            token = word
            try:
                embedd = w2v_model[token]
                mean_vector = mean_vector + embedd
                counts += 1
                continue
            except:
                abx = 1

            token = lmtzr.lemmatize(token, pos="n")
            try:
                embedd = w2v_model[token]
                mean_vector = mean_vector + embedd
                counts += 1
                continue
            except:
                abx = 1

            token = lmtzr.lemmatize(token, pos="v")
            try:
                embedd = w2v_model[token]
                mean_vector = mean_vector + embedd
                counts += 1
                continue
            except:
                abx = 1

            token = lmtzr.lemmatize(token, pos="a")
            try:
                embedd = w2v_model[token]
                mean_vector = mean_vector + embedd
                counts += 1
                continue
            except:
                abx = 1

            token = ps.stem(token)
            try:
                embedd = w2v_model[token]
                mean_vector = mean_vector + embedd
                counts += 1
                continue
            except:
                abx = 1

            unknowns.append(word)
        if counts > 0:
            mean_vector = mean_vector / counts
        embeddings.append(mean_vector)

    embeddings = np.array(embeddings)
    unknowns = list(set(unknowns))
    print("Not found words:")
    print(unknowns)
    print("counts = ", len(unknowns))
    return embeddings


def get_matrices():
    if os.path.isfile('embedding_matrix.pickle') and os.path.isfile('data_label.pickle'):
        with open('embedding_matrix.pickle', 'rb') as handle:
            data_matrix = pickle.load(handle)
        with open('data_label.pickle', 'rb') as handle:
            y = pickle.load(handle)
        return data_matrix, y

    train = pd.read_csv('train.csv').values
    x = train[:, 1]
    y = train[:, 2].astype(np.int32)

    clean_x = clean_input(x)
    token_sentences = tokenize_sentence(clean_x)
    embedding_matrix = generate_embeddings(token_sentences)

    with open('embedding_matrix.pickle', 'wb') as handle:
        pickle.dump(embedding_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('data_label.pickle', 'wb') as handle:
        pickle.dump(y, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return embedding_matrix, y


def get_embedding():
    if os.path.isfile('embedding_matrix.pickle') and os.path.isfile('data_label.pickle'):
        with open('embedding_matrix.pickle', 'rb') as handle:
            data_matrix = pickle.load(handle)
        with open('data_label.pickle', 'rb') as handle:
            y = pickle.load(handle)
        return data_matrix, y


if __name__ == '__main__':
    get_matrices()
