#!/bin/bash
repeat=10
split="standard.622" #"standard"/"standard.622"
schema="standard" #"standard/balanced"
bert_weights="bert_pretrained_100K.h5"
#nohup python experiments.py --with_context_data False --schema $schema --split $split --model_name "RNN:OOC" --repeat $repeat > RNN.ooc.log
#nohup python experiments.py --with_context_data True --schema $schema --split $split --model_name "RNN:INC1" --repeat $repeat > RNN.inc1.log
#nohup python experiments.py --bert_weights $bert_weights --with_context_data False --schema $schema --split $split --model_name "BERT:OOC" --repeat $repeat > BERT.ooc.log
#nohup python experiments.py --bert_weights $bert_weights --with_context_data True --schema $schema --split $split --model_name "BERT:INC1" --repeat $repeat > BERT.inc1.log
#nohup python experiments.py --bert_weights $bert_weights --with_context_data True --schema $schema --split $split --model_name "BERT:INC2" --repeat $repeat > BERT.inc2.log

# Repeating the BERT:OOC experiment
#nohup python experiments.py --with_context_data False --schema $schema --split $split --model_name "BERT:OOC" --repeat $repeat > BERT.ooc.log

# Training context-insensitive models on context-aware data
nohup python experiments.py --with_context_data True --schema $schema --split $split --model_name "RNN:OOC" --repeat $repeat > RNN.ooc.log
nohup python experiments.py --with_context_data True --schema $schema --split $split --model_name "BERT:OOC" --repeat $repeat > BERT.ooc.log
