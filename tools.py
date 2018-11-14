import numpy as np
from itertools import product, combinations

import gensim.downloader as api
from gensim.models import Word2Vec, FastText, KeyedVectors
from os.path import isfile
import TopicVAE
import torch


## Topic modeling

def create_TopicVAE_model(filename, TopicVAE_model, args, doc_term_tensor):
    if isfile(filename):
        TopicVAE_model = torch.load(filename)
    else:
        TopicVAE_model = TopicVAE_model
        optimizer = torch.optim.Adam(TopicVAE_model.parameters(), args.learning_rate, betas=(args.momentum, 0.999))
        TopicVAE_model = TopicVAE.train(TopicVAE_model, args, optimizer, doc_term_tensor)
        torch.save(TopicVAE_model, filename)
    return TopicVAE_model


# def topic_coherence_NPMI(beta, M, doc_term_matrix):
#   corpus_word_count = doc_term_matrix.sum()
#   K = beta.shape[1] # beta has dim V x K
#   coherences = np.zeros(K)
#   for t in range(K):
#     index = np.argsort(-beta[:, t])[0:M]
#     cart_prod = product(list(index), list(index))
#     for ind1, ind2 in cart_prod:
#       if ind1 == ind2:
#         pass
#       else:
#         d_ind2 = (doc_term_matrix[:, ind2] > 0).sum()
#         d_ind1 = (doc_term_matrix[:, ind1] > 0).sum()
#         d_ind12 = ((doc_term_matrix[:, ind1] > 0) & (doc_term_matrix[:, ind2] > 0)).sum()
#         p_1 = d_ind1 / corpus_word_count
#         p_2 = d_ind2 / corpus_word_count
#         p_12 = d_ind12 / corpus_word_count
#
#         # coherences[t] += np.log1p(d_ind12) - np.log(d_ind1) #- np.log(d_ind2)
#
#   return coherences

def topic_coherence_NPMI(beta, M, doc_term_matrix):
    corpus_word_count = doc_term_matrix.sum()
    epsilon = 1e-8
    K = beta.shape[1] # beta has dim V x K
    coherences = np.zeros(K)
    for t in range(K):
        index = np.argsort(-beta[:, t])[0:M]
        combos = combinations(list(index), 2)
        for ind1, ind2 in combos:
            if_2 = doc_term_matrix[:, ind2] > 0
            if_1 = doc_term_matrix[:, ind1] > 0
            d_ind2 = if_2.sum()
            d_ind1 = if_1.sum()
            d_ind12 = (if_1 & if_2).sum()
            p_1 = d_ind1 / corpus_word_count
            p_2 = d_ind2 / corpus_word_count
            p_12 = d_ind12 / corpus_word_count
            coherences[t] += (np.log(p_12 + epsilon) - np.log(p_1 + epsilon) - np.log(p_2 + epsilon)) / -np.log(p_12 + epsilon)
    return coherences


associations = {
    'jesus': ['prophet', 'jesus', 'matthew', 'christ', 'worship', 'church'],
    'comp ': ['floppy', 'windows', 'microsoft', 'monitor', 'workstation', 'macintosh',
              'printer', 'programmer', 'colormap', 'scsi', 'jpeg', 'compression'],
    'car  ': ['wheel', 'tire'],
    'polit': ['amendment', 'libert', 'regulation', 'president'],
    'crime': ['violent', 'homicide', 'rape'],
    'midea': ['lebanese', 'israel', 'lebanon', 'palest'],
    'sport': ['coach', 'hitter', 'pitch'],
    'gears': ['helmet', 'bike'],
    'nasa ': ['orbit', 'spacecraft'],
}
def identify_topic_in_line(line):
    topics = []
    for topic, keywords in associations.items():
        for word in keywords:
            if word in line:
                topics.append(topic)
                break
    return topics
def print_top_words(beta, feature_names, n_top_words=10):
    '''
    beta - V X K
    feature_names - V
    '''
    print('---------------Printing the Topics------------------')
    beta = beta.T
    for i in range(len(beta)):
        line = " ".join([feature_names[j]
                            for j in beta[i].argsort()[:-n_top_words - 1:-1]])
        topics = identify_topic_in_line(line)
        print('|'.join(topics))
        print('     {}'.format(line))
    print('---------------End of Topics------------------')

## perplexity

def perplexity(model, test_set):
    '''
    model - trained model
    test_set - tensor
    '''
    doc_lens = test_set.sum(1)
    _, log_liks = model.forward(test_set, compute_loss = True, avg_loss = False)
    return (log_liks / doc_lens).mean().exp().detach().numpy()



## Word vectors

def create_language_model(filename, model = None, doc_term_matrix = None,
                             vocab_list = [], epochs = 10, sentences = []):
    '''

    '''
    if not isfile(filename):
        dict_word_freq = dict(zip(vocab_list, list(doc_term_matrix.sum(0))))
        model.build_vocab_from_freq(word_freq = dict_word_freq)
        # train the model
        model.train(sentences = sentences, epochs = epochs, total_examples = doc_term_matrix.shape[1])
        # save the model
        model.save(filename)
    else:
        model = KeyedVectors.load(filename)

    return model

def create_embedding_matrix(language_model, vocab_list):
    _, dim = language_model.syn1neg.shape
    embedding_matrix = np.random.randn(len(vocab_list), dim)
    iterator = 0
    for word in vocab_list:
        if word in language_model.wv.vocab:
            embedding_matrix[iterator] = language_model.wv.word_vec(word)
        else:
            continue
            # embedding_matrix2[iterator] = pretrained_language_model.wv.most_similar(word)
            # or something like that
        iterator += 1

    return embedding_matrix
