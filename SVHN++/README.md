Accuracy with dropout -  88 %.

Accuracy without dropout - 81 %

The technique used was, 2 convolutional layers followed by dropout before readout. \\
Dropout was performed only on the training data and not on the test data. \\
Conventional pooling (i.e. max pooling ) was used although using other forms gives one a better accuracy(L4 pooling for instance )

( read https://arxiv.org/pdf/1204.3968.pdf )

The convolutional model looks like this, with changes to the number of filters.

15 filters (5,5) -> maxpooling -> 512 filters (7,7) -> maxpooling -> dense -> dropout(on training) -> readout.
