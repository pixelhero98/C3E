# Channel Capacity Constrained Estimation

## Intro
1. The Model_box.py contains **gcn_box.py**, and **gdc_box.py** for generating C3E variants with different widths and depths.
2. The **Train_Val_Test.py** consists of the corresponding training, validation, and testing of C3E variants on various datasets (**Cora**, **Citeseer**, **Pubmed**, **AmazonPhoto**, **AmazonComputers**, etc).
3. The **Utility.py** has all functions for calculating the entropy of node representation, normalized Dirichlet energy, and other statistics (simply save the model and test it with test masks, and use **get_model_rep()** to retrieve the node representation). Note that for transparency, you may have to use these functions to check these metrics by running them yourself. 

## Example resultss (**no graph rewiring, round to the continuous values to the nearest for implementation**)

**Cora**,

---

#### Depth: 2
* **Network channel capacity**: 15.9584
* **Lower bound**: 10.0112 in `[7.9040, 17.5644]`
* **Representation compression**: 0.0107
* <details>
    <summary>View Dimensions & Probabilities</summary>

    * **Hidden dimensions**: `[1983.0742, 1123.0810]`
    * **Rounded hidden dims**: `[1983, 1123]`
    * **Dropout probabilities**: `[0.5805, 0.3616]`
    </details>



#### Depth: 3
* **Network channel capacity**: 23.4857
* **Lower bound**: 12.1243 in `[7.9040, 17.5644]`
* **Representation compression**: 0.0183
* <details>
    <summary>View Dimensions & Probabilities</summary>

    * **Hidden dimensions**: `[1983.0862, 1274.8117, 830.9384]`
    * **Rounded hidden dims**: `[1983, 1275, 831]`
    * **Dropout probabilities**: `[0.5805, 0.3913, 0.3946]`
    </details>


#### Depth: 4
* **Network channel capacity**: 30.9106
* **Lower bound**: 13.6767 in `[7.9040, 17.5644]`
* **Representation compression**: 0.0265
* <details>
    <summary>View Dimensions & Probabilities</summary>

    * **Hidden dimensions**: `[1983.0882, 1274.7815, 1038.7230, 701.8598]`
    * **Rounded hidden dims**: `[1983, 1275, 1039, 702]`
    * **Dropout probabilities**: `[0.5805, 0.3913, 0.4490, 0.4032]`
    </details>



#### Depth: 5
* **Network channel capacity**: 38.2757
* **Lower bound**: 14.8987 in `[7.9040, 17.5644]`
* **Representation compression**: 0.0353
* <details>
    <summary>View Dimensions & Probabilities</summary>

    * **Hidden dimensions**: `[1983.0854, 1271.3599, 1053.2914, 911.4606, 619.8425]`
    * **Rounded hidden dims**: `[1983, 1271, 1053, 911, 620]`
    * **Dropout probabilities**: `[0.5805, 0.3907, 0.4531, 0.4639, 0.4048]`
    </details>



#### Depth: 6
* **Network channel capacity**: 45.6218
* **Lower bound**: 15.9161 in `[7.9040, 17.5644]`
* **Representation compression**: 0.0443
* <details>
    <summary>View Dimensions & Probabilities</summary>

    * **Hidden dimensions**: `[1983.0882, 1277.7392, 1067.4793, 941.0846, 843.7655, 558.0908]`
    * **Rounded hidden dims**: `[1983, 1278, 1067, 941, 844, 558]`
    * **Dropout probabilities**: `[0.5805, 0.3918, 0.4552, 0.4685, 0.4727, 0.3981]`
    </details>


#### Depth: 7
* **Network channel capacity**: 52.9313
* **Lower bound**: 16.7777 in `[7.9040, 17.5644]`
* **Representation compression**: 0.0537
* <details>
    <summary>View Dimensions & Probabilities</summary>

    * **Hidden dimensions**: `[1983.0883, 1281.1045, 1072.7756, 949.0070, 865.8996, 790.9780, 511.0134]`
    * **Rounded hidden dims**: `[1983, 1281, 1073, 949, 866, 791, 511]`
    * **Dropout probabilities**: `[0.5805, 0.3925, 0.4557, 0.4694, 0.4771, 0.4774, 0.3925]`
    </details>



**Citeseer**,
---

#### Depth: 2
* **Network channel capacity**: 17.2631
* **Lower bound**: 11.4531 in `[8.1098, 18.0218]`
* **Representation compression**: 0.0044
* <details>
    <summary>View Dimensions & Probabilities</summary>

    * **Hidden dimensions**: `[5270.4205, 2912.4121]`
    * **Rounded hidden dims**: `[5270, 2912]`
    * **Dropout probabilities**: `[0.5873, 0.3559]`
    </details>



#### Depth: 3
* **Network channel capacity**: 25.3745
* **Lower bound**: 13.8906 in `[8.1098, 18.0218]`
* **Representation compression**: 0.0075
* <details>
    <summary>View Dimensions & Probabilities</summary>

    * **Hidden dimensions**: `[5270.4213, 3337.3608, 2178.5658]`
    * **Rounded hidden dims**: `[5270, 3337, 2179]`
    * **Dropout probabilities**: `[0.5873, 0.3877, 0.3950]`
    </details>



#### Depth: 4
* **Network channel capacity**: 33.3886
* **Lower bound**: 15.6878 in `[8.1098, 18.0218]`
* **Representation compression**: 0.0109
* <details>
    <summary>View Dimensions & Probabilities</summary>

    * **Hidden dimensions**: `[5270.4219, 3356.2480, 2732.2018, 1845.2651]`
    * **Rounded hidden dims**: `[5270, 3356, 2732, 1845]`
    * **Dropout probabilities**: `[0.5873, 0.3891, 0.4488, 0.4031]`
    </details>



#### Depth: 5
* **Network channel capacity**: 41.3389
* **Lower bound**: 17.1029 in `[8.1098, 18.0218]`
* **Representation compression**: 0.0145
* <details>
    <summary>View Dimensions & Probabilities</summary>

    * **Hidden dimensions**: `[5270.4284, 3345.8902, 2767.7160, 2394.2945, 1634.6767]`
    * **Rounded hidden dims**: `[5270, 3346, 2768, 2394, 1635]`
    * **Dropout probabilities**: `[0.5873, 0.3883, 0.4527, 0.4638, 0.4057]`
    </details>


#### Depth: 6
* **Network channel capacity**: 49.2641
* **Lower bound**: 18.2784 in `[8.1098, 18.0218]`
* **Representation compression**: 0.0182
* <details>
    <summary>View Dimensions & Probabilities</summary>

    * **Hidden dimensions**: `[5270.4177, 3358.8859, 2798.2368, 2462.5779, 2203.3201, 1476.9303]`
    * **Rounded hidden dims**: `[5270, 3359, 2798, 2463, 2203, 1477]`
    * **Dropout probabilities**: `[0.5873, 0.3892, 0.4545, 0.4681, 0.4722, 0.4013]`
    </details>
## To be continued...(more statistical results of different models and datasets)
