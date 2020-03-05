#!/bin/bash

# Pilot study with 3 random restarts (3rr)
#nohup python experiments_v2.py --model_name "none" --repeat 10 --data data/cctk_c.csv.zip > results/@test/rnn.log
#nohup python experiments_v2.py --model_name "ch" --repeat 3 --data data/cctk_c.csv.zip > results/@test/rnn-h.log
#nohup python experiments_v2.py --model_name "ca" --repeat 3 --data data/cctk_c.csv.zip > results/@test/rnn-a.log
#nohup python experiments_v2.py --model_name "ci" --repeat 3 --data data/cctk_c.csv.zip > results/@test/rnn-i.log
#nohup python experiments_v2.py --model_name "cc" --repeat 3 --data data/cctk_c.csv.zip > results/@test/rnn-c.log

# X1: Vocabulary augmented with the parent comment (RNN, 10rr)
#nohup python experiments_v2.py --model_name "none" --repeat 10 --data data/cctk_c.csv.zip > results/@test/rr10/rnn.log
#nohup python experiments_v2.py --model_name "ch" --repeat 10 --data data/cctk_c.csv.zip > results/@test/rr10/rnn-h.log
#nohup python experiments_v2.py --model_name "cc" --repeat 10 --data data/cctk_c.csv.zip > results/@test/RNN/rr10_augv/rnn-cc.log

# X2: Vocabulary not augmented at all (RNN, 10rr) - this addition helps rnn-h and hurts rnn (s.s. test may verify this trend, talk with @jigsaw)
#nohup python experiments_v2.py --aug_voc 0 --model_name "none" --repeat 10 --data data/cctk_c.csv.zip > results/@test/rr10-av/rnn.log
#nohup python experiments_v2.py --aug_voc 0 --model_name "ch" --repeat 10 --data data/cctk_c.csv.zip > results/@test/rr10-av/rnn-h.log

# X3: Vocabulary augmented (BERT, 10rr)
nohup python experiments_v2.py --model_name "bert" --repeat 10 --data data/cctk_c.csv.zip > results/@test/BERT/bert.log
nohup python experiments_v2.py --model_name "bert_sep" --repeat 10 --data data/cctk_c.csv.zip > results/@test/BERT/bert-sep.log
#nohup python experiments_v2.py --model_name "bert_cc" --repeat 10 --data data/cctk_c.csv.zip > results/@test/BERT/bert-cc.log