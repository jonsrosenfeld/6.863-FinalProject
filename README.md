# 6.863-FinalProject
## Installation 
This code was written in python 2.7.
To run it define a new enviroment using cona 
and install

pytourch
tourchtext
spacy
sklearn
matplotlib
nltk
## Repository structure
* Big_set_GR-UGR  - A set of grammatical and ungrammatical examples we sample from to construct GRAMMAR2 datasets. 
* Datasets 
** 1
** 2
*
*
*




## Commands to run

### To train a network on a dataset run 


#### On install without gpu
python -W ignore TestPYT_final.py --epochs 15 --d_hidden 20 --d_embed 50 --gpu -1 --train_dataset Datasets/different_size_train_anbn/new_train.tsv --test_dataset Datasets/different_size_train_anbn/new_dev.tsv --save_path anbn-test  | tee final-log.txt

Note that without gpu use, the training might take days, especially on  the bigger version of a^n b^n dataset ! We provide some of the pretrained models
#### On install with gpu

python -W ignore TestPYT_final.py --epochs 15 --d_hidden 20 --d_embed 50 --gpu 1 --train_dataset Datasets/different_size_train_anbn/new_train.tsv --test_dataset Datasets/different_size_train_anbn/new_dev.tsv --save_path anbn-test  | tee final-log.txt


