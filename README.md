
This repository contains the code to generate recommendations using a fidelity value for the users and artists. 

For each dataset we have to run a script to make the split by time:
 - generate_fidelity_mtrx_{dataset}_time.py

Then we have to run the training and evaluation with the script:
 - model_predict.py -d '{dimensions}' -f '{folder}'

The parameter {folder} indicates where the matrix with train and test is located, depending on the folder the dataset that is used e.g:

 - python model_predict.py -f pandora -d 30

Parameters: 
 - '-f' folder to use
 - '-d' dimension to use for ALS algorithm
 - '-r' indicates if we want the full train set, trim-hard or trim-soft, values are 50, 0, or none
 - '-p' used to print the 'p-values', values can be True or False

Note: After parameter tunning the best number of dimensions for each dataset is: 
 - pandora -> 200
 - lastfm -> 200
 - nowplaying -> 30
 - 30music -> 50
