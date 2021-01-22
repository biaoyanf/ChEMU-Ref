#!/bin/bash

#python minimize.py
python get_char_vocab.py


#python filter_embeddings.py /home/biaoyan/data/Embedding/glove.840B.300d.txt train.english.jsonlines dev.english.jsonlines
python filter_embeddings.py ./glove.840B.300d.txt train.english.jsonlines dev.english.jsonlines

#python cache_elmo.py train.english.jsonlines dev.english.jsonlines test.english.jsonlines
python cache_elmo.py train.english.jsonlines 
