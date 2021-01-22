# ChEMU-Ref: A Corpus for Modeling Anaphora Resolution in the Chemical Domain

## Introduction

This repository contains code introduced in the following paper:

- ChEMU-Ref: A Corpus for Modeling Anaphora Resolution in the Chemical Domain

- Biaoyan Fang, Christian Druckenbrodt, Saber A. Akhondi, Jiayuan He, Timothy Baldwin and Karin Verspoor

- In EACL 2021

## Dataset 

- ChEMU-Ref Dataset will be available in [ChEMU 2021 Shared task](http://chemu.eng.unimelb.edu.au/).

- For detailed annotation guideline of ChEMU-Ref corpus, please refer to [ChEMU-Ref annotation guideline](https://data.mendeley.com/datasets/r28xxr6p92)


## Getting Started 
- Install python (preference 3) requirement: `pip install -r requirements.txt`
- Download [GloVe](http://nlp.stanford.edu/data/glove.840B.300d.zip) and [ChELMo](https://github.com/zenanz/ChemPatentEmbeddings) embeddings
- run `setup_all.sh`
- To train your own models, modify the related codes in `setup_training.sh` and run it.
- Install brat evalation tool 

## Training Instructions
- Experiment configurations are found in `experiments.conf`
- Choose an experiment that you would like to run, e.g. `best`
- Training: `python train.py <experiment>`
- Results are stored in the `logs` directory.

## Evaluation
- Evaluation: `python evaluate.py <experiment>`
- Evaluation tool provides differnet settings, `exact` and `relax` mention matching. For this paper, we use `exact` mention matching.

## Input Data format
The input format is *.jsonlines* file. Each line of the *.jsonlines* file is a batch of sentences and must in the following format
```
{
"doc_key": "0414", 
"sentences": [["Step", "4", "Synthesis", "of", "Compound", "22", "To", "a", "solution", "of", "Compound", "21", "(", "445", "mg", ",", "0.88", "mmol", ")", "in", "dichloromethane", "(", "2", "mL", ")", "was", "added", "trifluoroacetic", "acid", "(", "2", "mL", ")", ",", "and", "the", "reaction", "mixture", "was", "stirred", "at", "room", "temperature", "for", "3", "hours", "."], ["The", "solvent", "was", "distilled", "off", "under", "reduced", "pressure", ",", "and", "saturated", "sodium", "hydrogen", "carbonate", "aqueous", "solution", "was", "added", "to", "the", "residue", ",", "followed", "by", "extraction", "with", "chloroform", "twice", "."], ["Then", ",", "the", "organic", "layer", "was", "dried", "over", "sodium", "sulfate", "."], ["The", "solvent", "was", "distilled", "off", "under", "reduced", "pressure", "to", "give", "Compound", "22", "(", "269", "mg", ",", "Yield", "100%", ")", "."]], 
"Coreference": [[[[47, 48]], [[20, 24]]], [[[87, 88]], [[73, 73]]], [[[97, 105]], [[4, 5]]]], 
"Transformed": [], 
"Reaction-associated": [[[[35, 37]], [[27, 32]]], [[[35, 37]], [[7, 24]]]], 
"Work-up": [[[[66, 67]], [[35, 37]]], [[[78, 80]], [[57, 62]]], [[[78, 80]], [[66, 67]]], [[[78, 80]], [[73, 73]]], [[[97, 105]], [[84, 85]]], [[[97, 105]], [[78, 80]]]], 
"Contained": []
}
```
For referring realtion, each pair contains two mentions: the first one is the *anaphora* and the second one is the *antecedent*.


