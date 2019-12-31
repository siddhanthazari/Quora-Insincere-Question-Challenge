import numpy as np
import pandas as pd
import nltk
from tqdm import tqdm
import pickle
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from scipy.sparse import lil_matrix
import unidecode
import contractions
import re
import os


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


def lemmatize_sentence(data):
    if os.path.isfile('base_sentences.pickle'):
        with open('base_sentences.pickle', 'rb') as handle:
            base_sentences = pickle.load(handle)
        return base_sentences

    lmtzr = WordNetLemmatizer()
    ps = nltk.stem.PorterStemmer()

    lem_dict = {}

    print("lemmatizing sentences...")
    print()
    base_sentences = []
    for sentence in tqdm(data):
        base = []
        for token in sentence:
            base_token = token
            try_count = 0
            while True:
                try_count += 1
                new_token = lmtzr.lemmatize(base_token, pos="n")
                if new_token != base_token:
                    base_token = new_token
                    continue
                new_token = lmtzr.lemmatize(base_token, pos="v")
                if new_token != base_token:
                    base_token = new_token
                    continue
                new_token = lmtzr.lemmatize(base_token, pos="a")
                if new_token != base_token:
                    base_token = new_token
                    continue
                if try_count > 10:
                    break
                new_token = ps.stem(base_token)
                if new_token != base_token:
                    base_token = new_token
                    continue
                break
            lem_dict[token] = base_token
            base.append(base_token)
        base_sentences.append(base)
    with open('base_sentences.pickle', 'wb') as handle:
        pickle.dump(base_sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('lem_dict.pickle', 'wb') as handle:
        pickle.dump(lem_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return base_sentences


def words_list(data):
    if os.path.isfile('tokens.pickle'):
        with open('tokens.pickle', 'rb') as handle:
            tokens = pickle.load(handle)
        return tokens

    print("Extracting words list...")
    print()
    tokens = [j for sub in tqdm(data) for j in sub]
    with open('tokens.pickle', 'wb') as handle:
        pickle.dump(tokens, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return tokens


def filter_tokens(data):
    if os.path.isfile('tokens_distribution.pickle'):
        with open('tokens_distribution.pickle', 'rb') as handle:
            tokens_distribution = pickle.load(handle)
        return tokens_distribution

    tokens_distribution = Counter(data)

    unq_tokens = list(set(data))
    print("Vocab size = " + str(len(unq_tokens)))

    for token in unq_tokens:
        if tokens_distribution[token] <= 2:
            del tokens_distribution[token]

    filtered_words = tokens_distribution.keys()
    print("Vocab size after removing uncommon words= " + str(len(filtered_words)))

    sw = set(stopwords.words('english'))

    for word in sw:
        if word in tokens_distribution.keys():
            del tokens_distribution[word]

    filtered_words = list(tokens_distribution.keys())
    print("Vocab size after removing stop words = " + str(len(filtered_words)))

    print("Vocabulary:")
    print(filtered_words)

    with open('tokens_distribution.pickle', 'wb') as handle:
        pickle.dump(tokens_distribution, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return tokens_distribution


def filter_sentences(data, vocab_distribution):
    if os.path.isfile('filtered_sentences.pickle'):
        with open('filtered_sentences.pickle', 'rb') as handle:
            filtered_sentences = pickle.load(handle)
        return filtered_sentences

    print("Filtering sentences...")
    print()
    filtered_sentences = []

    for sentence in tqdm(data):
        filtered_sentence = []
        for word in sentence:
            if vocab_distribution[word] > 0:
                filtered_sentence.append(word)
        filtered_sentences.append(filtered_sentence)

    with open('filtered_sentences.pickle', 'wb') as handle:
        pickle.dump(filtered_sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return filtered_sentences


def get_filtered_words():
    if os.path.isfile('filtered_words.pickle'):
        with open('filtered_words.pickle', 'rb') as handle:
            filtered_words = pickle.load(handle)
        return filtered_words


def get_lem_dict():
    if os.path.isfile('lem_dict.pickle'):
        with open('lem_dict.pickle', 'rb') as handle:
            lem_dict = pickle.load(handle)
        return lem_dict


def get_filtered_sentences():
    if os.path.isfile('filtered_sentences.pickle'):
        with open('filtered_sentences.pickle', 'rb') as handle:
            filtered_sentences = pickle.load(handle)
        return filtered_sentences


def get_clean_sentences():
    if os.path.isfile('sent_tokens.pickle'):
        with open('sent_tokens.pickle', 'rb') as handle:
            clean_sentences = pickle.load(handle)
        return clean_sentences


def get_matrices():
    if os.path.isfile('data_matrix.pickle') and os.path.isfile('data_label.pickle'):
        with open('data_matrix.pickle', 'rb') as handle:
            data_matrix = pickle.load(handle)
        with open('data_label.pickle', 'rb') as handle:
            y = pickle.load(handle)
        return data_matrix, y

    train = pd.read_csv('train.csv').values
    x = train[:, 1]
    y = train[:, 2].astype(np.int32)

    clean_x = clean_input(x)
    token_sentences = tokenize_sentence(clean_x)
    base_sentences = lemmatize_sentence(token_sentences)
    all_words = words_list(base_sentences)
    tokens_distribution = filter_tokens(all_words)

    filtered_words = list(tokens_distribution.keys())
    filtered_words.sort()

    filtered_sentences = filter_sentences(base_sentences, tokens_distribution)

    d = len(filtered_words)
    m = len(base_sentences)

    print("Required matrix size = " + str(m) + " x " + str(d))

    data_matrix = lil_matrix((m, d), dtype=np.int8)

    mapping_dict = {}
    idx = 0
    for word in filtered_words:
        mapping_dict[word] = idx
        idx += 1

    idx = 0
    for sentence in tqdm(filtered_sentences):
        to_fill = {}
        for token in sentence:
            position = mapping_dict[token]
            if position in to_fill.keys():
                to_fill[position] += 1
            else:
                to_fill[position] = 1
        data_matrix[idx, list(to_fill.keys())] = list(to_fill.values())
        idx += 1

    with open('filtered_words.pickle', 'wb') as handle:
        pickle.dump(filtered_words, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('data_matrix.pickle', 'wb') as handle:
        pickle.dump(data_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('data_label.pickle', 'wb') as handle:
        pickle.dump(y, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return data_matrix, y


if __name__ == '__main__':
    get_matrices()
