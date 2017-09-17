## SVHN ++
### Dataset

The SVHN data must be downloaded into the folder ./svhn/mat in your home directory only then will one be able to run the files.

The data can be downloaded from http://ufldl.stanford.edu/housenumbers/ "Format 2: Cropped Digits" is what this project aims to work on

### Results

The SVHN++ is the final implimentation, uses dropout and attains an accuracy of **88%** on the test data.
Accuracy with dropout -  88 %  
Accuracy without dropout - 81 %

### Model

The technique used was, 2 convolutional layers followed by dropout before readout.  
Dropout was performed only on the training data and not on the test data.   
Conventional pooling (i.e. max pooling ) was used although using other forms gives one a better accuracy(L4 pooling for instance )  
( read https://arxiv.org/pdf/1204.3968.pdf )

The convolutional model looks like this, with changes to the number of filters.

15 filters (5,5) -> maxpooling -> 512 filters (7,7) -> maxpooling -> dense -> dropout(on training) -> readout.

## CIFAR-10

### Dataset
The cifar folder and other dependencies are not within this folder, therefore to run it successfully one must download the CIFAR -10 dataset into this folder. 

The data set is available [here] https://www.cs.toronto.edu/~kriz/cifar.html
### Model

Trial 1, uses a  32 (5,5) -> 32 (7,7) ->  64 (7,7). kernel sequence.
