# context_toxicity
`Toxicity Detection in Context` 
* Concerning comments existing in a thread.
* Context information: 
    * The parent comment.
    * The discussion topic.
    
### Manual additions
* Add a folder `data` wherein the related CSV files will be located.
    * This research studies two schemas; in context annotation (IC) and out of context (OC), so two CSV are used.
* Add a folder `embeddings` when using pre-trained embeddings.

### Building the datasets
Create random splits:
>python experiments.py --create_random_splits 10

Downsample the two categories (one per dataset) to make the datasets equibalanced while equally sized:
>python experiments.py --create_balanced_datasets

Then, create 10 random splits:
>python experiments.py --create_random_splits 10 --use_balanced_datasets True

### Running a classifier

Run a simple bi-LSTM by:
> nohup python experiments.py --with_context_data False --with_context_model "RNN:OOC" --repeat 10 > rnn.ooc.log &

* You can train it also in IC data, by changing the related argument.
    * If you call "RNN:INC1", the same LSTM will be trained, but another LSTM will encode the parent text (IC data required) and concatenate the two encoded texts before the dense layers on the top.
    * If you call "BERT:OOC1" you have a simple BERT.
    * If you call "BERT:OOC2" you concatenate the parent text (IC data required) with a SEPARATED token.
    * If you call "BERT:CA" you extend BERT:OOC1 with the LSTM encoded parent text, similarly to the RNN:INC1.

The names are messy, but these will hopefully change. 