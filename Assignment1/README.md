# Implementing decision tree

###### To run for kaggle1 dataset: python decision_tree-kaggle_1.py 

## Tree building:
First, we have saved the input data into an numpy array and removed the header from it.
Later we have implemented a binary tree named d_tree class which has the following attributes:

### Class d_tree:
* __Threshold height__ (this is a hyperparameter used to limit the tree growth)
* __Isleaf__ (indicator of the leaf)
* __Left,right__ childs of the tree
* __Data__ which has the total data stored in it
* __Height__ (a variable which keeps on the count of height)
* __Pred__(The prediction value of the tree i.e. the mean)
* __Corr__(This matrix stores the correlation value of the all classes w.r.t to the last column(i.e. the output))
* __split_attribute__(The split attribute id)
* __split_index__(the value of the attribute about which the row splitting is done)
* __d__ (the size of the data present)
* __Left_data__ and __right_data__(which store the left and right split matrices)
* __standard_dev__(which is the mean square loss of the last column (note it isn’t standard deviation)

### Splitting criteria:
We have selected the split attribute on the basis of correlation i.e. we have calculated the correlation of each attribute w.r.t the output value and have chosen the attribute with the __highest correlation coefficient__. 
The split value of the attribute is chosen using the square loss or the absolute loss (as per the argument).


### Pruning:
We have done the pruning using one-fold cross validation.\
The best height is selected on the basis of validation accuracy.


## Results:
### For kaggle 1:

#### Best_loss_values:
Mean square train = 3.56(the min value)\
Mean square val = 7.09(the min value)\
When val is min train error is 3.79

Absolute train = 1.0536(the min value)\
Absolute val =  1.774(the min value)\
When val is min train error is 1.184

### For kaggle 2:

#### Best_loss_values:
Mean square train = 0.6611(the min value)\
Mean square val = 0.725(the min value)\
When val is min train error is 0.723


### For Kaggle 1:

Training time -> 0.712 s  (approximately)\
Inference time ->  0.003 s  (approximately)

### For Kaggle 2:

Training time -> 16 s (approximately)\
Inference time ->  0.012 s (approximately)
