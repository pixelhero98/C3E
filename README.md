# Channel Capacity Constrained Estimation

## Intro
1. The Model_box.py contains **gcn_box.py**, and **gdc_box.py** for generating C3E variants with different widths and depths.
2. The **Train_Val_Test.py** consists of the corresponding training, validation, and testing of C3E variants on various datasets (**Cora**, **Citeseer**, **Pubmed**, **AmazonPhoto**, **AmazonComputers**).
3. The **Utility.py** has all functions for calculating the entropy of node representation, normalized Dirichlet energy, and other statistics (simply save the model and test it with test masks, and use **get_model_rep()** to retrieve the node representation). Note that for transparency, you may have to use these functions to check these metrics by running them yourself. 

## Demo (Cora, **no graph rewiring, round to the continuous values to the nearest for implementation**)
   
Number of Propagation Layers: 2

Weights: [1982.0861337185738, 1146.535210713598]

Current Channel Capacity: 7.942719404150568

Current Constraints: 2.1135356037493027/7.903965634032166 #**Current Constraints** is \phi_0 - ln(n) \in [0, ln(n)/\eta - ln(n)] in the paper

Dropout Probabilities: [0.5803912569433047, 0.36646659486422706]

Representation Compression Ratio: 0.0120239629176994067

=========================================================================================

Number of Propagation Layers: 3

Weights: [1982.0877777435787, 1280.113030330709, 887.9094205864952]

Current Channel Capacity: 10.59783498649899

Current Constraints: 4.23506770362854/7.903965634032166

Dropout Probabilities: [0.5803914589431685, 0.39240779634481593, 0.4095480746570942]

Representation Compression Ratio: 0.019841007191145612

=========================================================================================

Number of Propagation Layers: 4

Weights: [1982.0873813514452, 1290.745740629883, 1057.0456262297066, 799.180929199864]

Current Channel Capacity: 13.236524538501568

Current Constraints: 5.802031508621/7.903965634032166

Dropout Probabilities: [0.5803914102388438, 0.394381776437316, 0.450229795181338, 0.4305406184725745]

Representation Compression Ratio: 0.023988230934324639

=========================================================================================

Number of Propagation Layers: 5

Weights: [1982.0880521776921, 1280.693363134103, 1063.8681454974221, 932.1988367565484, 720.6749153978122]

Current Channel Capacity: 15.808484081705346

Current Constraints: 7.023090329868874/7.903965634032166

Dropout Probabilities: [0.5803914926626206, 0.3925158323888879, 0.45375996389123585, 0.4670178130514959, 0.4360132856235889]

Representation Compression Ratio: 0.026734786044857

## To be continued...(more statistical results of different models and datasets)
