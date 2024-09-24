# Channel Capacity Constrained Estimation

## Intro
1. The Model_box.py contains gcn_box.py, gat_box.py, and gdc_box.py for generating C3E variants with different widths and depths.
2. The Train_Val_Test.py consists of the corresponding training, validation, and testing of C3E variants on various datasets (Cora, Citeseer, Pubmed, AmazonPhoto, AmazonComputers).
3. The Utility.py has all functions for calculating the entropy of node representation, normalized Dirichlet energy, and other statistics (simply save the model and test it with test masks, and use **get_model_rep()** to retrieve the node representation).

## Demo
