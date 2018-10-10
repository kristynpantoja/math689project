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


Some code for topic modeling:

"Gaussian LDA for Topic Models with Word Embeddings" by Das et. al.
https://github.com/rajarshd/Gaussian_LDA








