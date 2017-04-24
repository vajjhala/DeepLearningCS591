For the bonus question, I got an accuracy of 88 %. The original was 81 %. 
The technique I used was, 2 convolutional layers followed by dropout before readout.
Dropout was performed only on the training data and not on the test data.
I used the conventional pooling (i.e. max pooling ) although I have read using other forms gives one a better accuracy(L4 pooling for instance )

The convolutional model looks like this, with changes to the number of filters.

15 filters (5,5) -> maxpooling -> 512 filters (7,7) -> maxpooling -> dense -> dropout(on training) -> readout.
