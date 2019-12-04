#!/bin/bash
split=0
nohup python experiments.py --with_context_data False --model_name "RNN:OOC" --use_balanced_datasets True --at_split $split > RNN.ooc.log
nohup python experiments.py --with_context_data True --model_name "RNN:INC1" --use_balanced_datasets True --at_split $split > RNN.inc1.log
nohup python experiments.py --with_context_data False --model_name "BERT:OOC" --use_balanced_datasets True --at_split $split > BERT.ooc.log
nohup python experiments.py --with_context_data True --model_name "BERT:INC1" --use_balanced_datasets True --at_split $split > BERT.inc1.log
nohup python experiments.py --with_context_data True --model_name "BERT:INC2" --use_balanced_datasets True --at_split $split > BERT.inc2.log
