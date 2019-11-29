#!/bin/bash
nohup python experiments.py --with_context_data False --model_name "RNN:OOC" > RNN.ooc.log
nohup python experiments.py --with_context_data True --model_name "RNN:INC1" > RNN.inc1.log
nohup python experiments.py --with_context_data False --model_name "BERT:OOC" > BERT.ooc.log
nohup python experiments.py --with_context_data True --model_name "BERT:INC1" > BERT.inc1.log
nohup python experiments.py --with_context_data True --model_name "BERT:INC3" > BERT.inc3.log
