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
#nohup python experiments_v2.py --model_name "bert" --repeat 10 --data data/cctk_c.csv.zip > results/@test/BERT/bert.log
#nohup python experiments_v2.py --model_name "bert_sep" --repeat 10 --data data/cctk_c.csv.zip > results/@test/BERT/bert-sep.log
#nohup python experiments_v2.py --model_name "bert_cc" --repeat 10 --data data/cctk_c.csv.zip > results/@test/BERT/bert-cc.log

# X4: Add the parent label to RnnChl
#nohup python experiments_v2.py --aug_voc 0 --model_name "chl" --repeat 10 --data data/cctk_c.csv.zip > results/@test/RNN/rr10_no_augv/rnn-hl.log

# X5: Add the toxicity score to RnnChl
#nohup python experiments_v2.py --aug_voc 0 --parent_label "RockV6_2:TOXICITY:probability" --model_name "chl" --repeat 10 --data data/parent_scored_cctk_c.csv.gz > results/@test/RNN/rr10_no_augv/rnn-hl.perspective.log
#nohup python experiments_v2.py --aug_voc 1 --model_name "none" --repeat 10 --data data/parent_scored_cctk_c.csv.gz > results/@test/RNN/rr10_no_augv/rnn.baseline.log
#nohup python experiments_v2.py --aug_voc 0 --parent_label "toxicity_parent" --model_name "chl" --repeat 10 --data data/parent_scored_cctk_c.csv.gz > results/@test/RNN/rr10_no_augv/rnn-hl.gold.log

# X6: Use Perspective for parent and target before an MLP
#nohup python experiments_v2.py --model_name "baseline_perspective" --repeat 10 --data data/cctk_c-perspective.csv > results/@test/BASELINES/b1.log

# X7: Use Perspective to add scores for both parent and target
#nohup python experiments_v2.py --model_name "rnn_chl2" --repeat 10 --data data/cctk_c-perspective.csv > results/@test/RNN/rnn-hl2.perspective.log

# X8: Use Perspective to add parent score to BERT-N
#nohup python experiments_v2.py --model_name "bert_hl" --repeat 10 --data data/cctk_c-perspective.csv > results/@test/BERT/bert-hl.perspective.log

# X9: Use Perspective to add parent score to BERT-N
nohup python experiments_v2.py --model_name "bert_hl_gold" --repeat 10 --data data/cctk_c-perspective.csv > results/@test/BERT/bert-hl-GOLD.perspective.log

# X10: Use the stored Perspective score of the PARENT as prediction
#nohup python experiments_v2.py --model_name "baseline_perspective@parent" --repeat 10 --data data/cctk_c-perspective.csv > results/@test/BASELINES/perspective.log

# X11: Use the stored Perspective score of the TARGET as prediction
#nohup python experiments_v2.py --model_name "baseline_perspective@target" --repeat 10 --data data/cctk_c-perspective.csv > results/@test/BASELINES/perspective@target.log