# ChEMU-Ref: A Corpus for Modeling Anaphora Resolution in the Chemical Domain

## Introduction

This repository contains code introduced in the following paper:

- [ChEMU-Ref: A Corpus for Modeling Anaphora Resolution in the Chemical Domain](https://www.aclweb.org/anthology/2021.eacl-main.116/)

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

## Experiment result on the ChEMU-Ref Dataset (Full corpus)

Below displays the primary experiment result on the full ChEMU-Ref dataset based on our proposed model. For further experiment results with providing oracle mentions and ChELMO embedding, please refers to [Full experiment results](https://github.com/biaoyanf/ChEMU-Ref/blob/main/experiment_result/Full%20ChEMU-Ref%20experiment%20result.pdf).

Relation | Method | P<sub>A</sub> | R<sub>A</sub> | F<sub>A</sub> | P<sub>R</sub> | R<sub>R</sub> | F<sub>R</sub>  
| ------------- | ------------- | ------------- | -------------| ------------- | ------------- | ------------- | ------------- | 
| Coref. (Surface) | coreference  | 89.4 | 55.9 | 68.7 | 79.2 | 47.7 | 59.5  
|                    | joint_train | 91.4 | 56.0 | 69.5 | 81.3 | 48.0 | 60.3  
|     Coref. (Atom) | coreference  | 89.4 | 55.9 | 68.7 | 81.3 | 48.3 | 60.6 
|                    | joint_train | 91.4 | 56.0 | 69.5 | 83.9 | 48.8 | 61.7   
|           Bridging | bridging | 89.5 | 83.9 | 86.6 | 81.4 | 72.8 | 76.8 
|                 | joint_train | 91.2 | 84.1 | 87.5 | 83.1 | 74.1 | 78.3  
|                 TR | bridging | 78.6 | 84.7 | 81.5 | 77.4 | 84.7 | 80.8  
|                 | joint_train | 79.7 | 85.9 | 82.7 | 77.6 | 85.9 | 81.5  
|                 RA | bridging | 89.5 | 84.6 | 87.0 | 80.6 | 68.5 | 74.0  
|                 | joint_train | 91.4 | 85.6 | 88.4 | 82.7 | 69.2 | 75.3     
|                 WU | bridging | 91.5 | 84.0 | 87.5 | 81.9 | 74.3 | 77.9  
|                 | joint_train | 93.1 | 83.7 | 88.1 | 83.6 | 76.0 | 79.6   
|                 CT | bridging | 89.8 | 77.5 | 83.1 | 85.1 | 70.0 | 76.8  
|                 | joint_train | 91.3 | 77.0 | 83.3 | 85.9 | 69.4 | 76.4   
|          Overall| joint_train | 91.2 | 74.0 | 81.7 | 82.8 | 68.7 | 75.1  

Anaphora resolution results over the test dataset (%). Models are trained for "coreference", "bridging" or "joint_train" (both tasks jointly). Models were trained over 30,000 epochs, and averaged over 3 runs with different random seeds. F<sub>A</sub> and F<sub>R</sub> denote the F1 score for anaphor and relation prediction, respectively.
