#!/bin/bash

# 3 random restarts
#nohup python experiments_v2.py --model_name "none" --repeat 10 --data data/cctk_c.csv.zip > results/@test/rnn.log
nohup python experiments_v2.py --model_name "ch" --repeat 3 --data data/cctk_c.csv.zip > results/@test/rnn-h.log
nohup python experiments_v2.py --model_name "ca" --repeat 3 --data data/cctk_c.csv.zip > results/@test/rnn-a.log
#nohup python experiments_v2.py --model_name "ci" --repeat 3 --data data/cctk_c.csv.zip > results/@test/rnn-i.log
#nohup python experiments_v2.py --model_name "cc" --repeat 3 --data data/cctk_c.csv.zip > results/@test/rnn-c.log

# 10 random restarts
nohup python experiments_v2.py --model_name "none" --repeat 10 --data data/cctk_c.csv.zip > results/@test/rr10/rnn.log
nohup python experiments_v2.py --model_name "ch" --repeat 10 --data data/cctk_c.csv.zip > results/@test/rr10/rnn-h.log