# math689project

| Embedding Method | Task1: | Task2: |
| ------------- |:-------------:| -----:|
| word2vec SG  |  X | X |
| word2vec CBOW  | X | X |
| GloVe | X  | X |
| SVD | X | X |
| elmo | X | X |
| SD-SG | X | X |
| SD-CBOW | X | X |
| tentatively dynamic word embeddings | X | X |



| Embedding Method | vector dim 1: | vector dim 2: |
| ------------- |:-------------:| -----:|
| word2vec SG  |  X | X |
| word2vec CBOW  | X | X |
| GloVe | X  | X |
| SVD | X | X |
| elmo | X | X |
| SD-SG | X | X |
| SD-CBOW | X | X |
| tentatively dynamic word embeddings | X | X |



| Idea: | Type of Problem | Bayesian |
| ------------- |:-------------:|-------------:|
| Comparison of downstream tasks | Empirical | not |
| vector dimension d | Empirical / New Model for Old Problem | prior on d (dirichlet process) |
| polysemy in vectors | New Model for Old Problem | maybe |
| faction detection | New / Old Model for New Problem | maybe |
| tweak an existing model | semi-new model | change prior |


## Some code for topic modeling:

Main Paper:
"Discovering Discrete Latent Topics with Neural Variational Inference" by Miao et. al.

Other potentially useful stuff:
"Neural Variational Inference for Text Modeling" by Miao et. al. 
https://github.com/ysmiao/nvdm

"Gaussian LDA for Topic Models with Word Embeddings" by Das et. al.
https://github.com/rajarshd/Gaussian_LDA


## Data

20 News Groups
http://qwone.com/~jason/20Newsgroups/







