# The Dataset (CCC)

To build the dataset of this work, we used the publicly available Civil Comments (CC) dataset (Borkan et al., 2019). 

CC was originally annotated by ten annotators per post, but the parent post (the previous post in the thread) was not 
shown to the annotators. 
 We call this new dataset Civil Comments in Context (CCC). Each CCC post was rated either as NON-TOXIC, UNSURE, TOXIC, or
VERY TOXIC, as in the original CC dataset.
We unified the latter two labels in both CC and CCC annotations to simplify the problem.  
In only 71 posts (0.07%) an annotator said UNSURE, meaning annotators were confident
in their decisions most of the time. We exclude these 71 posts from our study, as there are too few
to generalize about.

The data are stored as a CSV. The data file contains 8 columns:

* id = the id of the target post on the civil comments platform 
* tox_codes_oc = the toxic codes given by the annotators whao did not have access to the parent post
* text = the target posts
* toxicity_annotator_count = the number of the annotators who annotated this post
* parent = the parent post
* tox_codes_ic = the toxic codes given by the annotators who did have access to the parent post
* tox_codes_parent = the toxic codes (out of context) of the parent post
* workers_ic = the ids of the annotators on the appen platform


## Please cite us:

@inproceedings{xenos-etal-2021-context,
    title = "Context Sensitivity Estimation in Toxicity Detection",
    author = "Xenos, Alexandros  and
      Pavlopoulos, John  and
      Androutsopoulos, Ion",
    booktitle = "Proceedings of the 5th Workshop on Online Abuse and Harms (WOAH 2021)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.woah-1.15",
    doi = "10.18653/v1/2021.woah-1.15",
    pages = "140--145",
    abstract = "User posts whose perceived toxicity depends on the conversational context are rare in current toxicity detection datasets. Hence, toxicity detectors trained on current datasets will also disregard context, making the detection of context-sensitive toxicity a lot harder when it occurs. We constructed and publicly release a dataset of 10k posts with two kinds of toxicity labels per post, obtained from annotators who considered (i) both the current post and the previous one as context, or (ii) only the current post. We introduce a new task, context-sensitivity estimation, which aims to identify posts whose perceived toxicity changes if the context (previous post) is also considered. Using the new dataset, we show that systems can be developed for this task. Such systems could be used to enhance toxicity detection datasets with more context-dependent posts or to suggest when moderators should consider the parent posts, which may not always be necessary and may introduce additional costs.",
} 
