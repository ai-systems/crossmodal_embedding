# Summary

Welcome :blush: !

This is the repo for our EACL2021 paper STAR: Cross-modal STAtement Representation for selecting relevant mathematical premises.

# Dataset

The dataset used in this work can be found in the ```dataset''' folder, containg the randomly retrieved pairs and the similar (BM25 retrieved) pairs.

# Running the code

Run this command to install the requirements:

```
pip install -r requirements.txt
```

Use the following command to run STAR:

```
python -m crossmodal_embedding.flows.train_star_flow --num_negatives=[NUM_NEG] \
                                                    --use_similar or --use_random [Only one option allowed]
                                                    
```

where num_negatives is the number of negative pairs for each positive one.

For example, to run STAR with 1 negative pair for each positive one and using the randomly retrieved pairs, the following command should be run:

```
python -m crossmodal_embedding.flows.train_star_flow --num_negatives=1 --use_random
                                                     
```

# Citation

You can cite our paper using:

```
@inproceedings{ferreira-freitas-2021-star,
    title = "{STAR}: Cross-modal [{STA}]tement [{R}]epresentation for selecting relevant mathematical premises",
    author = "Ferreira, Deborah  and
      Freitas, Andr{\'e}",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.eacl-main.282",
    pages = "3234--3243"
}
```

## Contact us

If you have any question, suggestions or ideas related to this dataset, please do not hesitate to contact me.

deborah[dot]ferreira[at]manchester[dot]ac[dot]uk
