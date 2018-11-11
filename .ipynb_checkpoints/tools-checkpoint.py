import numpy as np
from itertools import product

def topic_coherence(beta, M, doc_term_matrix):
  K = beta.shape[1] # beta has dim V x K
  coherences = np.zeros(K)
  for t in range(K):
    index = np.argsort(-beta[:, t])[0:M]
    cart_prod = product(list(index), list(index))
    for ind1, ind2 in cart_prod:
      if ind1 == ind2:
        pass
      else:
        d_ind1 = (doc_term_matrix[:, ind1] > 0).sum()
        d_ind12 = ((doc_term_matrix[:, ind1] > 0) & (doc_term_matrix[:, ind2] > 0)).sum()
        coherences[t] += np.log1p(d_ind12) - np.log(d_ind1)

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
    print('---------------Printing the Topics------------------')
    for i in range(len(beta)):
        line = " ".join([feature_names[j] 
                            for j in beta[i].argsort()[:-n_top_words - 1:-1]])
        topics = identify_topic_in_line(line)
        print('|'.join(topics))
        print('     {}'.format(line))
    print('---------------End of Topics------------------')
