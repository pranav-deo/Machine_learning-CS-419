# CS 419 Assignment-1 Report (160070048-170040012)

## Tree building:
First We have saved the input data into an numpy array and removed the header from it.
Later we have implemented a binary tree named d_tree class

Which has the following attributes:

### Class d_tree:
* Threshold height (this is a hyperparameter used to limit the tree growth)
* Isleaf (indicator of the leaf)
* Left,right child’s of the tree
* Data which has the total data stored in it
* Height (a variable which keeps on the count of height)
* Pred(The prediction value of the tree i.e. the mean)
* Corr(This matrix stores the correlation value of the all classes w.r.t to the last column(i.e. the output))
* split_attribute(The split attribute id)
* split_index(the value of the attribute about which the row splitting is done)
* d (the size of the data present)
* Left_data and right_data(which store the left and right split matrices)
* standard_dev(which is the mean square loss of the last column (note it isn’t standard deviation)

### Splitting criteria:
We have selected the split attribute on the basis of correlation i.e. we have calculated the correlation of each attribute w.r.t the output value and have chosen the attribute with the highest correlation coefficient. 
The split value of the attribute is chosen using the square loss or the absolute loss (as per the argument).


### Pruning:
We have done the pruning using one-fold cross validation
The best height is selected on the basis of validation accuracy.


## Results:
### For kaggle 1:

#### Best_loss_values:
Mean square train = 3.56(the min value)
Mean square val = 7.09(the min value)
When val is min train error is 3.79

Absolute train = 1.0536(the min value)
Absolute val =  1.774(the min value)
When val is min train error is 1.184

### For kaggle 2:

#### Best_loss_values:
Mean square train = 0.6611(the min value)
Mean square val = 0.725(the min value)
When val is min train error is 0.723


### For Kaggle 1:

Training time -> 0.712 s  (approximately)
Inference time ->  0.003 s  (approximately)

### For Kaggle 2:

Training time -> 16 s (approximately)
Inference time ->  0.012 s (approximately)
