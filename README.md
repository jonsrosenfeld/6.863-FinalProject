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



## Commands to run

### To train a network on a dataset run 


#### On install without gpu
python -W ignore TestPYT_final.py --epochs 15 --d_hidden 20 --d_embed 50 --gpu -1 --train_dataset Datasets/different_size_train_anbn/new_train.tsv --test_dataset Datasets/different_size_train_anbn/new_dev.tsv --save_path anbn-test  | tee final-log.txt

Note that without gpu use, the training might take hours, especially on a^n b^n dataset 
#### On install with gpu

python -W ignore TestPYT_final.py --epochs 15 --d_hidden 20 --d_embed 50 --gpu 1 --train_dataset Datasets/different_size_train_anbn/new_train.tsv --test_dataset Datasets/different_size_train_anbn/new_dev.tsv --save_path anbn-test  | tee final-log.txt


