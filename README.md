# Structure from linear string inputs: Learning human language derived CFG with LSTMs
## Installation 
We pre-supposed that the person planing to run this project has conda installed

We have included the yaml environment file  PT2.yaml

To create an enviroment for the project run

** conda env create -f PT2.yaml **

then switch to the environment by running 

** source activate PT2 **



** NOTE THAT THIS FILE CONTAINS A CPU BASED PYTOURCH PACKAGE
WE CAN NOT PROVIDE GPU ENVIRONMENT SINCE THE CHOICE OF APPROPRIATE VERSION OF GPU-PT DEPENDS ON YOUR LOCAL MACHNE CUDA  **


** THIS WILL RESULT IN A VERY SLOW (POSSIBLY DAYS) MODEL TRAINING for a^nb^n datasets ! **

## Repository structure
* Big_set_GR-UGR  - A set of grammatical and ungrammatical examples we sample from to construct GRAMMAR2 datasets. 
* Datasets 
   1.  different_size_train_anbn  - all of the datasets generated using a^nb^n grammar
   2.  different_size_train_grammar2 -all of the datasets generated using GRAMMAR2
   
* training_logs -The concolidated logs for all of our experiments

* figures - All figures used in the report. The figures are generated from the Logs from training_logs  

* prediction_results - The folder with the predicion results for pre-trained models

* pretrained_models

   1.  60000_99.8.pt - A converging a^nb^n model, that makes a small number of mistakes on test
   2.  60000_best.pt - A converged a^b^n model, no mistakes on test, perfect generalization on different lengths
   3.  15000-perfect_one_unsure.pt   - A converged GRAMMAR2 model we discuss in our report 
  

Project files
  * TestPYT_final.py - ALL THE TRAINING CODE, can be run from the command line
  * helper_functions.py - A set of helper functions icnluding dataset generators and the figure creation script, does not have a command line interface 
  * load_model.py - All the prediction and result generation code, can be run from the command line
  * PT2.yaml - An environment set-up file 


## Commands to run

### To train a network on a dataset run 


#### On install without gpu
python -W ignore TestPYT_final.py --epochs 15 --d_hidden 20 --d_embed 50 --gpu -1 --train_dataset Datasets/different_size_train_anbn/new_train.tsv --test_dataset Datasets/different_size_train_anbn/new_dev.tsv --save_path anbn-test  | tee log-anbn.txt

python -W ignore TestPYT_final.py --epochs 15 --d_hidden 20 --d_embed 50 --gpu -1 --train_dataset Datasets/different_size_train_grammar2/grammar2_train-15000.tsv --test_dataset Datasets/different_size_train_grammar2/grammar2_dev.tsv --save_path grammar-test  | tee log-grammar.txt

Note that without gpu use, the training might take days, especially on  the bigger version of a^n b^n dataset ! We provide some of the pretrained models in the pretrained folder
#### On install with gpu

python -W ignore TestPYT_final.py --epochs 15 --d_hidden 20 --d_embed 50 --gpu 1 --train_dataset Datasets/different_size_train_anbn/new_train.tsv --test_dataset Datasets/different_size_train_anbn/new_dev.tsv --save_path anbn-test  | tee final-log.txt

###  To predict run:
#### Perfect result with one uncertain example

python -W ignore  load_model.py \
--train_address \
  Datasets/different_size_train_grammar2/grammar2_train-15000.tsv \
--test_address \
  Datasets/different_size_train_grammar2/grammar2_dev.tsv \
--number_of_examples 4000 \
--analysis_file_address predicted_dev.txt \
--model_address \
  pre-trained_models/grammaticals/15000-perfect_one_unsure.pt
  
  
#### Perfect example, advesarial  result:

python -W ignore  load_model.py \
--train_address \
  Datasets/different_size_train_grammar2/grammar2_train-15000.tsv \
--test_address \
  Datasets/different_size_train_grammar2/testing_theory.tsv \
--number_of_examples 7 \
--analysis_file_address predicted_fail.txt \
--model_address \
  pre-trained_models/grammaticals/15000-perfect_one_unsure.pt
  
  
#### Perfect result on the counting dataset, small network of 5-5:  Trained on lengths (0-250, predicted  for lengthes 250-500)
  
Note that anbn models can take a while: up to 20 mins to process due to the long dataset generation phase 
  python -W ignore  load_model.py \
--train_address \
  Datasets/different_size_train_anbn/new_train.tsv \
--test_address \
  Datasets/different_size_train_anbn/new_dev.tsv \
--number_of_examples 8000 \
--analysis_file_address predicted_anbn.txt \
--model_address \
  pre-trained_models/anbn/60000_best.pt
  
#### Work in progress snapshot for a^nb^n dataset, makes a small number of appropriate mistakes  
  python -W ignore  load_model.py \
--train_address \
  Datasets/different_size_train_anbn/new_train-15000.tsv \
--test_address \
  Datasets/different_size_train_anbn/new_dev.tsv \
--number_of_examples 8000 \
--analysis_file_address predicted_anbn_prelim.txt \
--model_address \
  pre-trained_models/anbn/60000_99.8.pt 








