#!/usr/bin/env python

# Convert text and standoff annotations into CoNLL format.

from __future__ import print_function

import os
import re
import sys
import json

from collections import namedtuple
from io import StringIO
from os import path

from sentencesplit import sentencebreaks_to_newlines
options = None

#EMPTY_LINE_RE = re.compile(r'^\s*$')
#CONLL_LINE_RE = re.compile(r'^\S+\t\d+\t\d+.')

TOKENIZATION_REGEX = re.compile(r'([0-9a-zA-Z]+|[^0-9a-zA-Z])')

NEWLINE_TERM_REGEX = re.compile(r'(.*?\n)')

def text_tokenizer(file_name):
    """Convert plain text into CoNLL format."""
    with open(file_name, "r") as fr:
# potrentially change the sentence splitting here
        ls = fr.readlines()
        sentences = []
        for l in ls:
            print("sentences: ", l)
            l = sentencebreaks_to_newlines(l)
            sentences.extend([s for s in NEWLINE_TERM_REGEX.split(l) if s])
    lines = []
    offset = 0
    token_index = 0 
    for s in sentences:
        nonspace_token_seen = False
        
# potrentially change the tokenizer here
        tokens = [t for t in TOKENIZATION_REGEX.split(s) if t]
        for t in tokens:
            if not t.isspace():
                lines.append([offset, offset + len(t), t, token_index])
                token_index += 1
                nonspace_token_seen = True
            offset += len(t)
        # sentences delimited by empty lines
        if nonspace_token_seen:
            lines.append([])
    return lines  #[ [span_stat_index, span_end_index, text, token_index]   ]
# e.g.
#  [[0, 7, 'Example', 0],
#  [8, 10, '50', 1],
#  [10, 11, '.', 2],
#  []]   3 end with [] means sentence end.


def align_span_and_token_from_tokenized_conll(text_path, token_path):
    text = None
    with open(text_path, "r") as fr:
        text = fr.read()
    span_and_token = []
    with open(token_path, "r") as fr:
        tokens = fr.readlines()
        span_start_index = 0
        token_index = 0
        for token in tokens:
            token = token.split("\n")[0]
            if len(token) == 0:
                span_start_index+=1
                span_and_token.append([]) # [] means the end of the sentence
                continue

            find_start = False
            while find_start == False:
                if text[span_start_index: span_start_index+1] == token[0]:
                    find_start = True
                if find_start == False: 
                    span_start_index+=1
                    continue
                find_token = False
                for span_end_index in range(span_start_index+1, len(text)+1):
                    if str(text[span_start_index: span_end_index]) == str(token):
                        span_and_token.append([span_start_index, span_end_index, token, token_index])
                        token_index+=1
                        span_start_index = span_end_index
                        find_token = True
                        break
                if find_token == False:
                    print("error. can not find the tokne.", token, span_start_index)
    
    #     print(span_and_token)
        assert(len(span_and_token) == len(tokens))

    return span_and_token  # [span_index_start, span_index_end, txt, token_id]



def process_ann(file_name):
    """ read the file of ann and matain entity label and relationship """
    mentions = {}
    mentions_span = {}
#     mentions_type = {}
    relations = {}
    relation_type = ["Coreference", "Transformed", "Reaction-associated", "Work-up", "Contained"]
    for r_type in relation_type:
        relations[r_type] = []
        
    with open(file_name, "r") as fr:
        lines = fr.readlines()
        for line in lines:
            ###filter out some unrelated label from NER
            split_line = line.split("\t")
            label = split_line[1].split(" ")[0]

            ## mention
            if "T" in split_line[0]:
                ## text
                mention_text = split_line[2].split("\n")[0]
                mentions[split_line[0]] = mention_text

                ### span
                span = split_line[1].split(";")
#               discontious mention:
                if len(span)> 1: 
                    mentions_span[split_line[0]] = []
                    for sp in span:
                        s = sp.split(" ")
                        mentions_span[split_line[0]].append((int(s[-2]), int(s[-1])))
#                     contious mention:
                else:
                    span = split_line[1].split(" ")
                    mentions_span[split_line[0]] = [(int(span[1]), int(span[2]))]
#           
                ### mention_type 
#                     print(split_line)
#                 ty = split_line[1].split(" ")[0]
#                 mentions_type[ann_file][split_line[0]] = ty

            ## relation    
            if "R" in split_line[0]:
                r_element = split_line[1].split(" ")
                r_type = r_element[0]
                r_anaphor_t = r_element[1].split(":")[1]
                r_antecedent_t = r_element[2].split(":")[1]
                relations[r_type].append(list([r_anaphor_t, r_antecedent_t]))

    return mentions, mentions_span, relations

def transfer_spanID_to_tokenID(lines, mentions_span):
    mentions_token = {}
    for t_key in mentions_span.keys():
        spans = mentions_span[t_key]
        # maybe discontinuous mention 
        token_temp = []
        for s in spans:
            start_t = None
            end_t = None
            
            start_s = s[0]
            end_s = s[1]
            
            find_start_id = False
            find_end_id = False
            
            for line in lines: 
                if line == []:
                    continue
                if line[0] == start_s:
                    find_start_id = True
                    start_t = line[3]
                if line[1] == end_s: 
                    find_end_id = True
                    end_t = line[3]
                
                if find_start_id and find_end_id: 
                    break
            if not find_start_id or not find_end_id:
                print("A: something wrong with the ID matching")
            elif start_t > end_t:
                print("b: something wrong with the ID matching")
              
            token_temp.append([start_t, end_t])   
        mentions_token[t_key]= token_temp

    return mentions_token
        
def constcut_jsonline_format(doc_id, lines, mentions_token, relations):
    # get sentences 
    instance = {}
#     Index(['speakers', 'doc_key', 'sentences', 'constituents', 'clusters', 'ner'], dtype='object')
    instance["doc_key"] = doc_id
    
    sentences = []
    sentence = []
    for line in lines:
        if line == []:
            sentences.append(sentence)
            sentence = []
            continue
        else:
            sentence.append(line[2])
#     relations_types = relations.keys() # coref, TR, RA, WU, CA

    instance["sentences"] = sentences
    
    #useless keys
    # speaker
    instance["speakers"] = []
    for s in sentences:
        instance["speakers"].append(["-"]*len(s))
    instance["constituents"] = []
    instance["ner"] = []
    
    # relation keys
    relations_token = {}
    for r_type in relations:
        relations_token[r_type] = []
        for anphor_antecedent_pair in relations[r_type]:
            relations_token[r_type].append([mentions_token[anphor_antecedent_pair[0]],mentions_token[anphor_antecedent_pair[1]]])
        
        instance[r_type] = relations_token[r_type]

    return instance

def convert_brat_to_training(annotation_path, token_conll_path, convert_path, convert_name):
    annotation_file_name = os.listdir(annotation_path)
    annotation_ann_name = []
    annotation_txt_name = []
    annotation_doc_id = []

    for file in annotation_file_name:
        if ".ann" in file:
            annotation_ann_name.append(file)
            annotation_doc_id.append(file.split(".")[0])        
        elif ".txt" in file:
            annotation_txt_name.append(file)
            
    
    for doc_id in annotation_doc_id:
        temp_jsonlines_path = convert_path_final+doc_id+".jsonlines"
        with open(temp_jsonlines_path, 'w') as fw:
            txt_path = annotation_path+doc_id+".txt"
            token_conll_path_per = token_conll_path+doc_id+".conll"
            lines = align_span_and_token_from_tokenized_conll(txt_path, token_conll_path_per)
            ann_path = annotation_path+doc_id+".ann"
            mentions, mentions_span, relations = process_ann(ann_path)
            mentions_token = transfer_spanID_to_tokenID(lines, mentions_span)
            instance = constcut_jsonline_format(doc_id, lines, mentions_token, relations)

        # acttually, we should dump this jonsonlines 
            json.dump(instance, fw)
            fw.write("\n")
            
    convert_path_final = convert_path + convert_name       
    with open(convert_path_final, "w") as fw:
        for doc_id in annotation_doc_id:
            temp_jsonlines_path = convert_path_final+doc_id+".jsonlines"
            with open(temp_jsonlines_path, "r") as fr:
                fw.write(fr.read())
                
    print("convertion done!!!")
                
if __name__ == "__main__":
    #annotation_path = "./chemical-patents/Round4_/"
    annotation_path = "/home/biaoyan/data/Chemical/chemical-patents/Round4_test/"
    #convert_path = "./chemical-patents/Round4_1/"
    token_conll_path = "/home/biaoyan/data/Chemical/chemical-patents/annotated_40_snippets_only_token/"
    convert_path = "/home/biaoyan/data/Chemical/chemical-patents/Round4_test_jsonlines/"
    convert_name = "convert_brat_training.jsonlines"
    
    convert_brat_to_training(annotation_path, token_conll_path, convert_path, convert_name)
    
    
            