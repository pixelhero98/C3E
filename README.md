# Channel Capacity Constrained Estimation

## Intro
1. The Model_box.py contains **gcn_box.py**, **gat_box.py**, and **gdc_box.py** for generating C3E variants with different widths and depths.
2. The **Train_Val_Test.py** consists of the corresponding training, validation, and testing of C3E variants on various datasets (**Cora**, **Citeseer**, **Pubmed**, **AmazonPhoto**, **AmazonComputers**).
3. The **Utility.py** has all functions for calculating the entropy of node representation, normalized Dirichlet energy, and other statistics (simply save the model and test it with test masks, and use **get_model_rep()** to retrieve the node representation). Note that for transparency, you may have to use these functions to check these metrics by running them yourself. 

## Demo (Cora, **no graph rewiring, round to the continuous values to the nearest for implementation**)
   
Number of Message-passing Layers: 2

Weights: [1982.0861337185738, 1146.535210713598]

Current Model Entropy: 7.942719404150568

Current Constraints/Maximum Graph Entropy: 2.1135356037493027/7.903965634032166 #**Current Constraints** is \phi_0 - log(n) in the paper

Dropout Probabilities: [0.5803912569433047, 0.36646659486422706]

Aspect Ratio: 0.0020239629176994067

=========================================================================================

Number of Message-passing Layers: 3

Weights: [1982.0877777435787, 1280.113030330709, 887.9094205864952]

Current Model Entropy: 10.59783498649899

Current Constraints/Maximum Graph Entropy: 4.23506770362854/7.903965634032166

Dropout Probabilities: [0.5803914589431685, 0.39240779634481593, 0.4095480746570942]

Aspect Ratio: 0.0029841007191145612

=========================================================================================

Number of Message-passing Layers: 4

Weights: [1982.0873813514452, 1290.745740629883, 1057.0456262297066, 799.180929199864]

Current Model Entropy: 13.236524538501568

Current Constraints/Maximum Graph Entropy: 5.802031508621/7.903965634032166

Dropout Probabilities: [0.5803914102388438, 0.394381776437316, 0.450229795181338, 0.4305406184725745]

Aspect Ratio: 0.003988230934324639

=========================================================================================

Number of Message-passing Layers: 5

Weights: [1982.0880521776921, 1280.693363134103, 1063.8681454974221, 932.1988367565484, 720.6749153978122]

Current Model Entropy: 15.808484081705346

Current Constraints/Maximum Graph Entropy: 7.023090329868874/7.903965634032166

Dropout Probabilities: [0.5803914926626206, 0.3925158323888879, 0.45375996389123585, 0.4670178130514959, 0.4360132856235889]

Aspect Ratio: 0.005116734786044857

=========================================================================================

Number of Message-passing Layers: 6

Weights: [1982.0883033979742, 1275.9884968638673, 1064.2028922564896, 944.8233098927088, 852.1087422581357, 671.8715955185958]

Current Model Entropy: 18.357902118015677

Current Constraints/Maximum Graph Entropy: 8.030894635739568/7.903965634032166

Dropout Probabilities: [0.5803915235298072, 0.3916385570657267, 0.45475036665975743, 0.4702891922872704, 0.4742019828953471, 0.4408663149150337]

Aspect Ratio: 0.0063019278683976755

=========================================================================================

Number of Message-passing Layers: 7

Weights: [1982.0880048498516, 1275.085590838045, 1067.21138645557, 949.7888295418971, 869.7780106745388, 799.7441810426232, 642.7262777674285]

Current Model Entropy: 20.897392593656914

Current Constraints/Maximum Graph Entropy: 8.891430069853396/7.903965634032166

Dropout Probabilities: [0.5803914868474953, 0.3914699519012754, 0.4556259931175206, 0.47089178375332896, 0.47801377308628, 0.4790257865455855, 0.4455732689996561]

Aspect Ratio: 0.007519579317410295

=========================================================================================

Number of Message-passing Layers: 8

Weights: [1982.0880753805334, 1276.878421167012, 1072.9855433918476, 956.8134188058397, 879.4342792225802, 821.7733446511927, 766.3435233611641, 628.6143704611412]

Current Model Entropy: 23.438877405893983

Current Constraints/Maximum Graph Entropy: 9.64558787406959/7.903965634032166

Dropout Probabilities: [0.5803914955135308, 0.3918047094131528, 0.4566160252571384, 0.47138334220542033, 0.47893009214760596, 0.48305294022839806, 0.48254856981671534, 0.45063322215316726]

Aspect Ratio: 0.008744197298334477

=========================================================================================


Number of Message-passing Layers: 9

Weights: [1982.0871369720678, 1280.3123569123165, 1080.4798222614477, 965.8650340543048, 890.1435279799067, 834.9904248776021, 791.3834004048285, 746.4560358466064, 625.5178166603082]

Current Model Entropy: 25.989291413282988

Current Constraints/Maximum Graph Entropy: 10.319195818249089/7.903965634032166

Dropout Probabilities: [0.5803913802121763, 0.3924449961791494, 0.45767680518138476, 0.47199524120936887, 0.47960098147623675, 0.4840148346130023, 0.4865937880348016, 0.4853926998166548, 0.45592546499143705]

Aspect Ratio: 0.009958661335561432

## To be continued...(more statistical results of different models and datasets)
