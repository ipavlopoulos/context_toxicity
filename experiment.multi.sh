#!/bin/bash
repeat=10
#nohup python experiments.py --with_context_data False --schema "standard.622" --model_name "RNN:OOC" --repeat $repeat > RNN.ooc.log
#nohup python experiments.py --with_context_data True --schema "standard.622" --model_name "RNN:INC1" --repeat $repeat > RNN.inc1.log
nohup python experiments.py --with_context_data False --schema "standard.622" --model_name "BERT:OOC" --repeat $repeat > BERT.ooc.log
nohup python experiments.py --with_context_data True --schema "standard.622" --model_name "BERT:INC1" --repeat $repeat > BERT.inc1.log
nohup python experiments.py --with_context_data True --schema "standard.622" --model_name "BERT:INC2" --repeat $repeat > BERT.inc2.log
