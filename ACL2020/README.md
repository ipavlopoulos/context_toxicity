# Toxicity detection w/ and w/o context
* Concerning comments existing in a thread.
* Context information: 
    * The parent comment.
    * The discussion topic.
* The large dataset (CAT_LARGE) can be found in the `data` folder.
    * `gn.csv` comprises the out of context annotations.
    * `gc.csv` comprises the in-context annotations.
* The small dataset (CAT_SMALL) is also included.
    
### Word embeddings
* You will need to add a folder `embeddings` when using pre-trained embeddings.
    * For example, GloVe embeddings.

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

### The article
* Presented at the 58th Annual Meeting of the Association for Computational Linguistics ([link](https://arxiv.org/abs/2006.00998)).

### How to cite this work:
```
@inproceedings{pavlopoulos-etal-2020-toxicity,
    title = "Toxicity Detection: Does Context Really Matter?",
    author = "Pavlopoulos, John  and
      Sorensen, Jeffrey  and
      Dixon, Lucas  and
      Thain, Nithum  and
      Androutsopoulos, Ion",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.acl-main.396",
    doi = "10.18653/v1/2020.acl-main.396",
    pages = "4296--4305",
}
```