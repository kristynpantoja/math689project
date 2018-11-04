#!/usr/bin/env bash

conda create --name math689env python=3.7

source activate math689env

conda install numpy

conda install -c pytorch pytorch

pip install --upgrade pip

pip install pyro-ppl

conda install nb_conda # for environment picking  

conda install scikit-learn

conda install gensim
