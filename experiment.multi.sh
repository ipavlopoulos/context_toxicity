#!/bin/bash
# 17/4/2020
# The first time you run this script uncomment the three lines below:
# git clone https://github.com/ipavlopoulos/ssig.git
# wget http://nlp.stanford.edu/data/glove.6B.zip && mkdir embeddings && mv glove* embeddings
# python experiments.py --create_random_splits 10 --with_context_data 1

r=5

# train @gn
nohup python experiments.py --with_context_data 0 --model_name "RNN:OOC" --repeat $r > rnn@gn.log
nohup python experiments.py --with_context_data 0 --model_name "BERT:OOC" --repeat $r > bert@gn.log

# train @gc
# w/o using the parent comment
nohup python experiments.py --with_context_data 1 --model_name "RNN:OOC" --repeat $r > rnn@gc.log
nohup python experiments.py --with_context_data 1 --model_name "BERT:OOC" --repeat $r > bert@gc.log
# w/ using the parent comment
nohup python experiments.py --with_context_data 1 --model_name "RNN:INC1" --repeat $r > rnn@gc.c1.log
nohup python experiments.py --with_context_data 1 --model_name "BERT:INC1" --repeat $r > bert@gc.c1.log
nohup python experiments.py --with_context_data 1 --model_name "BERT:INC2" --repeat $r > bert@gc.c2.log
