Here again the cifar folder and other dependencies are not within this folder, to run it 
successfully one must download them to this place.

Trial 1, using  32 (5,5) -> 32 (7,7) ->  64 (7,7). kernel sequence.

The accuracy of CIFAR on the pre-trained model SVHM, is yeilding bad results.
My guess is, the initialised weight ( as obtained from teh SVHN model ) are a 
very bad initialisation. The slope of the function at these points is possibly very 
flat and this is descending into a local minima which is far higher than the 
global minima ( Assuming it's a convex function ).
This is my guess. 




