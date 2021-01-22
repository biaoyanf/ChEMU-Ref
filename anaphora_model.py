from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import operator
import random
import math
import json
import threading
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import h5py

import util
import coref_ops
import metrics
import brat

import json
import copy

class AnaphoraModel(object):
  def __init__(self, config):
    self.config = config
    self.context_embeddings = util.EmbeddingDictionary(config["context_embeddings"])
    self.head_embeddings = util.EmbeddingDictionary(config["head_embeddings"], maybe_cache=self.context_embeddings)
    self.char_embedding_size = config["char_embedding_size"]
    self.char_dict = util.load_char_dict(config["char_vocab_path"])   
    self.max_span_width = config["max_span_width"]
    
    # new
    self.bridging_types = config["bridging_types"]
    self.mention_loss_rate = config["mention_loss_rate"]
    self.use_glove_or_w2v = config["use_glove_or_w2v"]
    self.provide_gold_mention_for_relation_prediction = config["provide_gold_mention_for_relation_prediction"]

    self.training_setting = config["training_setting"]

    
    if config["lm_path"]:
      self.lm_file = h5py.File(self.config["lm_path"], "r")
    else:
      self.lm_file = None
    self.lm_layers = self.config["lm_layers"]
    self.lm_size = self.config["lm_size"]
    self.eval_data = None # Load eval data lazily.

    input_props = []
    input_props.append((tf.string, [None, None])) # Tokens.
    input_props.append((tf.float32, [None, None, self.context_embeddings.size])) # Context embeddings.
    input_props.append((tf.float32, [None, None, self.head_embeddings.size])) # Head embeddings.
    input_props.append((tf.float32, [None, None, self.lm_size, self.lm_layers])) # LM embeddings.
    input_props.append((tf.int32, [None, None, None])) # Character indices.
    input_props.append((tf.int32, [None])) # Text lengths.

    input_props.append((tf.bool, [])) # Is training.
    
    input_props.append((tf.int32, [None])) # Gold starts anaphor bridging.
    input_props.append((tf.int32, [None])) # Gold ends anaphor bridging.
    input_props.append((tf.int32, [None])) # Gold starts antecedent bridging.
    input_props.append((tf.int32, [None])) # Gold ends antecedent bridging.
    input_props.append((tf.float32, [None, len(self.bridging_types)+1]))  # Gold labels - bridging relations types +1 - [1,0,0,0,0]
    
    input_props.append((tf.int32, [None])) # Gold starts anaphor coref.
    input_props.append((tf.int32, [None])) # Gold ends anaphor coref.
    input_props.append((tf.int32, [None])) # Gold starts antecedent coref.
    input_props.append((tf.int32, [None])) # Gold ends antecedent coref.
    
    
    input_props.append((tf.int32, [None])) # Gold all span starts bridging
    input_props.append((tf.int32, [None])) # Gold all span ends bridging
    
    input_props.append((tf.int32, [None])) # Gold all span starts coref
    input_props.append((tf.int32, [None])) # Gold all span ends coref
    
    input_props.append((tf.int32, [None])) # Gold all span starts bridging and coref 
    input_props.append((tf.int32, [None])) # Gold all span ends bridging and coref

    self.queue_input_tensors = [tf.placeholder(dtype, shape) for dtype, shape in input_props]
    dtypes, shapes = zip(*input_props)
    queue = tf.PaddingFIFOQueue(capacity=10, dtypes=dtypes, shapes=shapes)
    self.enqueue_op = queue.enqueue(self.queue_input_tensors)
    self.input_tensors = queue.dequeue()

    self.predictions, self.loss = self.get_predictions_and_loss(*self.input_tensors)
    
    self.global_step = tf.Variable(0, name="global_step", trainable=False)
    self.reset_global_step = tf.assign(self.global_step, 0)
    learning_rate = tf.train.exponential_decay(self.config["learning_rate"], self.global_step,
                                               self.config["decay_frequency"], self.config["decay_rate"], staircase=True)
    trainable_params = tf.trainable_variables()
    gradients = tf.gradients(self.loss, trainable_params)
    gradients, _ = tf.clip_by_global_norm(gradients, self.config["max_gradient_norm"])
    optimizers = {
      "adam" : tf.train.AdamOptimizer,
      "sgd" : tf.train.GradientDescentOptimizer
    }
    optimizer = optimizers[self.config["optimizer"]](learning_rate)
    self.train_op = optimizer.apply_gradients(zip(gradients, trainable_params), global_step=self.global_step)
    
    
  # add for dynamic
  def set_provide_gold_mention_for_relation_prediction(self, bool_provided):
    self.provide_gold_mention_for_relation_prediction = bool_provided
    
  # done with adding dynamic   
    
  def start_enqueue_thread(self, session):
    with open(self.config["train_path"]) as f:
      train_examples = [json.loads(jsonline) for jsonline in f.readlines()]
    def _enqueue_loop():
      while True:
        random.shuffle(train_examples)
        for example in train_examples:
          tensorized_example = self.tensorize_example(example, is_training=True)
          
          feed_dict = dict(zip(self.queue_input_tensors, tensorized_example))
                      
          session.run(self.enqueue_op, feed_dict=feed_dict)
    enqueue_thread = threading.Thread(target=_enqueue_loop)
    enqueue_thread.daemon = True
    enqueue_thread.start()

  def restore(self, session):
    # Don't try to restore unused variables from the TF-Hub ELMo module.
    vars_to_restore = [v for v in tf.global_variables() if "module/" not in v.name]
    saver = tf.train.Saver(vars_to_restore)
    checkpoint_path = os.path.join(self.config["log_dir"], "model.max.ckpt")
    print("Restoring from {}".format(checkpoint_path))
    session.run(tf.global_variables_initializer())
    saver.restore(session, checkpoint_path)

  def load_lm_embeddings(self, doc_key):
    if self.lm_file is None:
      return np.zeros([0, 0, self.lm_size, self.lm_layers])
    file_key = doc_key.replace("/", ":")
    group = self.lm_file[file_key]
    num_sentences = len(list(group.keys()))
    sentences = [group[str(i)][...] for i in range(num_sentences)]
    lm_emb = np.zeros([num_sentences, max(s.shape[0] for s in sentences), self.lm_size, self.lm_layers])
    for i, s in enumerate(sentences):
      lm_emb[i, :s.shape[0], :, :] = s
    return lm_emb

  def tensorize_mentions(self, mentions):
    if len(mentions) > 0:
      starts, ends = zip(*mentions)
    else:
      starts, ends = [], []
    return np.array(starts), np.array(ends)

  def tensorize_span_labels(self, tuples, label_dict):
    if len(tuples) > 0:
      starts, ends, labels = zip(*tuples)
    else:
      starts, ends, labels = [], [], []
    return np.array(starts), np.array(ends), np.array([label_dict[c] for c in labels])
  
  def find_middle(self, entity, cluster):
    for i in range(1, len(cluster)-1 ):
      if entity == cluster[i]:
        return True
    return False
    
  def get_coref_clusters(self, coref_links):
#     from:
#  [[[[9, 9]], [[5, 7]]],
#  [[[16, 16]], [[11, 14], [18, 30]]],
#  [[[275, 289]], [[9, 9]]],
#  [[[[1, 1]]], [[275, 289]]], 
#  [[[[2, 2]]], [[275, 289]]],
#  [[[9, 9]], [[3, 3]]]]
#     to:   discontinue mentions are filltered out
#  [[[[[1, 1]]], [[275, 289]], [[9, 9]], [[5, 7]]],
#  [[[[2, 2]]], [[275, 289]], [[9, 9]], [[5, 7]]],
#  [[[[1, 1]]], [[275, 289]], [[9, 9]], [[3, 3]]],
#  [[[[2, 2]]], [[275, 289]], [[9, 9]], [[3, 3]]]]

    clusters = []
    for co_index, coref_pair in enumerate(coref_links):
        find_cluster = False
        clusters_num = len(clusters)
        if len(coref_pair[0]) != 1 or len(coref_pair[1]) != 1:  # filtering out discontinue mentions 
            continue
        for cl_index in range(clusters_num): 
            if coref_pair[0] == clusters[cl_index][-1]:  ## pair [c,d] and cluster [a,b,c]  get [a, b, c, d]
                clusters[cl_index].append(coref_pair[1])
                find_cluster = True
    #             break
            elif coref_pair[1] == clusters[cl_index][0]: ## pair [d,a] and cluster [a,b,c]  get [d, a, b, c]
                temp_cluster = []
                temp_cluster.append(coref_pair[0])
                for ele_index, element in enumerate(clusters[cl_index]):
                    temp_cluster.append(element)
                clusters[cl_index] = temp_cluster
                find_cluster = True
    #             break
            elif self.find_middle(coref_pair[0], clusters[cl_index]): ## pair [b,d] and cluster [a,b,c]   in the middle   get [a,b,c] and [a,b,d]
                temp_cluster = []
                for ele_index, element in enumerate(clusters[cl_index]):
                    if element != coref_pair[0]:
                        temp_cluster.append(element)
                    else:
                        break
                temp_cluster.append(coref_pair[0])
                temp_cluster.append(coref_pair[1])
                clusters.append(temp_cluster)
                find_cluster = True
    #             break
            elif self.find_middle(coref_pair[1], clusters[cl_index]): ## pair [d,b] and cluster [a,b,c]   in the middle   get [a,b,c] and [d,b,c]
                temp_cluster = []
                temp_cluster.append(coref_pair[0])
                temp_cluster.append(coref_pair[1])
                start_insert = False
                for ele_index, element in enumerate(clusters[cl_index]):
                    if start_insert:
                        temp_cluster.append(element)
                    if element == coref_pair[1]:
                        start_insert = True  # from the next element
                clusters.append(temp_cluster)
                find_cluster = True
    #             break
        if not find_cluster:
            clusters.append(coref_pair)
    return clusters

  def tensorize_example(self, example, is_training):

    # get bridging 
    relation_map = {br:i+1 for i, br in enumerate(self.bridging_types)}
    relation_map["No-relation"] = 0
#    instance["Coreference"] =
#     [[[[148, 149]], [[145, 146]]],
    #  [[[160, 169]], [[46, 46]]],
    #  [[[46, 46]], [[7, 44]]]]
    gold_mentions_anaphor_bridging = []
    gold_mentions_antecedent_bridging = []
    gold_labels_bridging = []
    
    gold_mentions_bridging = set()
    for br in self.bridging_types:
        if len(example[br]) == 0:
            continue
        for mention_pair in example[br]:
#             discard discontious mention but keep the same format  0-anaphor  1-antecedent
            if len(mention_pair[0]) == 1 and len(mention_pair[1]) == 1:
#                 gold_mention_pairs.append([mention_pair[0], mention_pair[1]])
                gold_mentions_anaphor_bridging.append(tuple(mention_pair[0][0]))
                gold_mentions_antecedent_bridging.append(tuple(mention_pair[1][0]))
                gold_label_bridging = np.zeros(len(self.bridging_types)+1)
                gold_label_bridging[relation_map[br]] = 1
                gold_labels_bridging.append(gold_label_bridging)
                
                gold_mentions_bridging.add(tuple(mention_pair[0][0]))
                gold_mentions_bridging.add(tuple(mention_pair[1][0]))
                
    # gold_mention_pairs e.g. [[[122, 122]], [[116, 120]]] and their coressponding label e.g.[0,1,0,0,0]
    gold_starts_anaphor_bridging, gold_ends_anaphor_bridging = self.tensorize_mentions(gold_mentions_anaphor_bridging)
    gold_starts_antecedent_bridging, gold_ends_antecedent_bridging = self.tensorize_mentions(gold_mentions_antecedent_bridging)
    
    gold_labels_bridging = np.array(gold_labels_bridging)
    
    
   # get coref: coref is a bit tricky
    coref_clusters = self.get_coref_clusters(example["Coreference"])
    # might have duplicate so need to do something
#     e.g. 
    #  [[[[275, 289]], [[9, 9]], [[5, 7]]],
    #  [[[275, 289]], [[9, 9]], [[3, 3]]]]
#     we get all the pair from anaphor to antecedent
    gold_anaphor_and_antecedent_pair_coref = set()
    for cluster in coref_clusters: 
        for i in range(len(cluster)):
            for j in range(i+1, len(cluster)):
                gold_anaphor_and_antecedent_pair_coref.add((tuple(cluster[i][0]), tuple(cluster[j][0]))) # tuple(anaphora, antecedent)
    

    gold_mentions_anaphor_coref = []
    gold_mentions_antecedent_coref = []    
    
    gold_mentions_coref = set()
    for anaphor_and_antecedent_pair in gold_anaphor_and_antecedent_pair_coref:
        gold_mentions_anaphor_coref.append(anaphor_and_antecedent_pair[0])
        gold_mentions_antecedent_coref.append(anaphor_and_antecedent_pair[1])
        
        gold_mentions_coref.add(anaphor_and_antecedent_pair[0])
        gold_mentions_coref.add(anaphor_and_antecedent_pair[1])
        
    gold_starts_anaphor_coref, gold_ends_anaphor_coref = self.tensorize_mentions(gold_mentions_anaphor_coref)
    gold_starts_antecedent_coref, gold_ends_antecedent_coref = self.tensorize_mentions(gold_mentions_antecedent_coref)
    
    
    gold_mentions = gold_mentions_bridging.union(gold_mentions_coref)
    # sorted might need to check 
    gold_mentions_bridging = sorted(gold_mentions_bridging)
    gold_mentions_coref = sorted(gold_mentions_coref)
    gold_mentions = sorted(gold_mentions)
    
    gold_all_span_starts_bridging, gold_all_span_ends_bridging = self.tensorize_mentions(gold_mentions_bridging)
    gold_all_span_starts_coref, gold_all_span_ends_coref = self.tensorize_mentions(gold_mentions_coref)
    gold_all_span_starts, gold_all_span_ends = self.tensorize_mentions(gold_mentions)
    

    
    sentences = example["sentences"]
    num_words = sum(len(s) for s in sentences)
    
#     assert num_words == len(speakers)
    text_len = np.array([len(s) for s in sentences])

    doc_key = example["doc_key"]
    
    lm_emb = None
    
    if is_training and len(sentences) > self.config["max_training_sentences"]:
        sentences, text_len, gold_starts_anaphor_bridging, gold_ends_anaphor_bridging, gold_starts_antecedent_bridging, gold_ends_antecedent_bridging, gold_labels_bridging, gold_starts_anaphor_coref, gold_ends_anaphor_coref, gold_starts_antecedent_coref, gold_ends_antecedent_coref, gold_all_span_starts_bridging, gold_all_span_ends_bridging, gold_all_span_starts_coref, gold_all_span_ends_coref, gold_all_span_starts, gold_all_span_ends, lm_emb = self.truncate_example(sentences, text_len, gold_starts_anaphor_bridging, gold_ends_anaphor_bridging, gold_starts_antecedent_bridging, gold_ends_antecedent_bridging, gold_labels_bridging, gold_starts_anaphor_coref, gold_ends_anaphor_coref, gold_starts_antecedent_coref, gold_ends_antecedent_coref, gold_all_span_starts_bridging, gold_all_span_ends_bridging, gold_all_span_starts_coref, gold_all_span_ends_coref, gold_all_span_starts, gold_all_span_ends, doc_key)
        

    else:
      lm_emb = self.load_lm_embeddings(doc_key)
    
    max_sentence_length = max(len(s) for s in sentences)
    max_word_length = max(max(max(len(w) for w in s) for s in sentences), max(self.config["filter_widths"]))

    tokens = [[""] * max_sentence_length for _ in sentences]
    context_word_emb = np.zeros([len(sentences), max_sentence_length, self.context_embeddings.size])
    head_word_emb = np.zeros([len(sentences), max_sentence_length, self.head_embeddings.size])
    char_index = np.zeros([len(sentences), max_sentence_length, max_word_length])
    for i, sentence in enumerate(sentences):
      for j, word in enumerate(sentence):
        tokens[i][j] = word
        context_word_emb[i, j] = self.context_embeddings[word]
        head_word_emb[i, j] = self.head_embeddings[word]
        char_index[i, j, :len(word)] = [self.char_dict[c] for c in word]
    tokens = np.array(tokens)


    # hangdle the gold_labels_bridging after truncatation or it doesnt have briging
    if len(gold_labels_bridging) == 0:
      gold_labels_bridging = np.zeros([0, len(self.bridging_types)+1])
    
    
    example_tensors = (tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len, is_training, gold_starts_anaphor_bridging, gold_ends_anaphor_bridging, gold_starts_antecedent_bridging, gold_ends_antecedent_bridging, gold_labels_bridging, gold_starts_anaphor_coref, gold_ends_anaphor_coref, gold_starts_antecedent_coref, gold_ends_antecedent_coref, gold_all_span_starts_bridging, gold_all_span_ends_bridging, gold_all_span_starts_coref, gold_all_span_ends_coref, gold_all_span_starts, gold_all_span_ends)
    
    
    return example_tensors


  def truncate_example(self, sentences, text_len, gold_starts_anaphor_bridging, gold_ends_anaphor_bridging, gold_starts_antecedent_bridging, gold_ends_antecedent_bridging, gold_labels_bridging, gold_starts_anaphor_coref, gold_ends_anaphor_coref, gold_starts_antecedent_coref, gold_ends_antecedent_coref, gold_all_span_starts_bridging, gold_all_span_ends_bridging, gold_all_span_starts_coref, gold_all_span_ends_coref, gold_all_span_starts, gold_all_span_ends, doc_key):                                                
                                                
    max_training_sentences = self.config["max_training_sentences"]
    num_sentences = len(sentences)
    assert num_sentences > max_training_sentences

    sentence_offset = random.randint(0, num_sentences - max_training_sentences)
    sentences = sentences[sentence_offset:sentence_offset + max_training_sentences]
    
    max_sentence_length = max(len(s) for s in sentences)
    
    word_offset = text_len[:sentence_offset].sum()
    num_words = text_len[sentence_offset:sentence_offset + max_training_sentences].sum()
    
    text_len = text_len[sentence_offset:sentence_offset + max_training_sentences]


    # bridgiing 
    gold_spans_anaphor_bridging = np.logical_and(gold_ends_anaphor_bridging >= word_offset, gold_starts_anaphor_bridging < word_offset + num_words)
    gold_spans_antecedent_bridging = np.logical_and(gold_ends_antecedent_bridging >= word_offset, gold_starts_antecedent_bridging < word_offset + num_words)

    gold_spans_bridging = np.logical_and(gold_spans_anaphor_bridging, gold_spans_antecedent_bridging)
                                                                                              
    gold_starts_anaphor_bridging = gold_starts_anaphor_bridging[gold_spans_bridging] - word_offset
    gold_ends_anaphor_bridging = gold_ends_anaphor_bridging[gold_spans_bridging] - word_offset
                                                
    gold_starts_antecedent_bridging = gold_starts_antecedent_bridging[gold_spans_bridging] - word_offset
    gold_ends_antecedent_bridging = gold_ends_antecedent_bridging[gold_spans_bridging] - word_offset
                                             
    gold_labels_bridging = gold_labels_bridging[gold_spans_bridging]
    
    # coref   same process as bridging
    gold_spans_anaphor_coref = np.logical_and(gold_ends_anaphor_coref >= word_offset, gold_starts_anaphor_coref < word_offset + num_words)
    gold_spans_antecedent_coref = np.logical_and(gold_ends_antecedent_coref >= word_offset, gold_starts_antecedent_coref < word_offset + num_words)

    gold_spans_coref = np.logical_and(gold_spans_anaphor_coref, gold_spans_antecedent_coref)
                                                                                                
    gold_starts_anaphor_coref = gold_starts_anaphor_coref[gold_spans_coref] - word_offset
    gold_ends_anaphor_coref = gold_ends_anaphor_coref[gold_spans_coref] - word_offset
                                                
    gold_starts_antecedent_coref = gold_starts_antecedent_coref[gold_spans_coref] - word_offset
    gold_ends_antecedent_coref = gold_ends_antecedent_coref[gold_spans_coref] - word_offset
    
    
    # all mention    same as coref and bridging
    
    gold_all_span_bridging = np.logical_and(gold_all_span_ends_bridging >= word_offset, gold_all_span_starts_bridging < word_offset + num_words)
    gold_all_span_starts_bridging = gold_all_span_starts_bridging[gold_all_span_bridging] - word_offset
    gold_all_span_ends_bridging = gold_all_span_ends_bridging[gold_all_span_bridging] - word_offset
    
    gold_all_span_coref = np.logical_and(gold_all_span_ends_coref >= word_offset, gold_all_span_starts_coref < word_offset + num_words)
    gold_all_span_starts_coref = gold_all_span_starts_coref[gold_all_span_coref] - word_offset
    gold_all_span_ends_coref = gold_all_span_ends_coref[gold_all_span_coref] - word_offset
    
    gold_all_span = np.logical_and(gold_all_span_ends >= word_offset, gold_all_span_starts < word_offset + num_words)
    gold_all_span_starts = gold_all_span_starts[gold_all_span] - word_offset
    gold_all_span_ends = gold_all_span_ends[gold_all_span] - word_offset
    
    
    ### get truncated lm_embedding 
    lm_emb = None
    if self.lm_file is None:
      lm_emb =  np.zeros([0, 0, self.lm_size, self.lm_layers])
    else: 
      file_key = doc_key.replace("/", ":")
      lm_group = self.lm_file[file_key]    
      lm_num_sentences = len(list(lm_group.keys()))
      lm_emb = np.zeros([max_training_sentences, max_sentence_length, self.lm_size, self.lm_layers])
      count = 0
      for i in range(lm_num_sentences):
        if i >= sentence_offset and i < sentence_offset + max_training_sentences:
          lm_sentence = lm_group[str(i)][...]
          lm_emb[count, :lm_sentence.shape[0], :, :] = lm_sentence
          count+=1
        
    return sentences, text_len, gold_starts_anaphor_bridging, gold_ends_anaphor_bridging, gold_starts_antecedent_bridging, gold_ends_antecedent_bridging, gold_labels_bridging, gold_starts_anaphor_coref, gold_ends_anaphor_coref, gold_starts_antecedent_coref, gold_ends_antecedent_coref, gold_all_span_starts_bridging, gold_all_span_ends_bridging, gold_all_span_starts_coref, gold_all_span_ends_coref, gold_all_span_starts, gold_all_span_ends, lm_emb


  def get_dropout(self, dropout_rate, is_training):
    return 1 - (tf.to_float(is_training) * dropout_rate)

  def coarse_to_fine_pruning(self, top_span_emb, top_span_mention_scores, c):
    k = util.shape(top_span_emb, 0)
    top_span_range = tf.range(k) # [k]
    antecedent_offsets = tf.expand_dims(top_span_range, 1) - tf.expand_dims(top_span_range, 0) # [k, k]
    antecedents_mask = antecedent_offsets >= 1 # [k, k]
    fast_antecedent_scores = tf.expand_dims(top_span_mention_scores, 1) + tf.expand_dims(top_span_mention_scores, 0) # [k, k]
    fast_antecedent_scores += tf.log(tf.to_float(antecedents_mask)) # [k, k]
    fast_antecedent_scores += self.get_fast_antecedent_scores(top_span_emb) # [k, k]
    
    _, top_antecedents = tf.nn.top_k(fast_antecedent_scores, c, sorted=False) # [k, c]
    top_antecedents_mask = util.batch_gather(antecedents_mask, top_antecedents) # [k, c]
    top_fast_antecedent_scores = util.batch_gather(fast_antecedent_scores, top_antecedents) # [k, c]
    top_antecedent_offsets = util.batch_gather(antecedent_offsets, top_antecedents) # [k, c]
    return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets

  def distance_pruning(self, top_span_emb, top_span_mention_scores, c):
    k = util.shape(top_span_emb, 0)
    top_antecedent_offsets = tf.tile(tf.expand_dims(tf.range(c) + 1, 0), [k, 1]) # [k, c]
    raw_top_antecedents = tf.expand_dims(tf.range(k), 1) - top_antecedent_offsets # [k, c]
    top_antecedents_mask = raw_top_antecedents >= 0 # [k, c]
    top_antecedents = tf.maximum(raw_top_antecedents, 0) # [k, c]

    top_fast_antecedent_scores = tf.expand_dims(top_span_mention_scores, 1) + tf.gather(top_span_mention_scores, top_antecedents) # [k, c]
    top_fast_antecedent_scores += tf.log(tf.to_float(top_antecedents_mask)) # [k, c]
    return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets

  def distance_pruning_for_gold_mention(self, top_span_emb, c):
    k = util.shape(top_span_emb, 0)
    top_antecedent_offsets = tf.tile(tf.expand_dims(tf.range(c) + 1, 0), [k, 1]) # [k, c]
    raw_top_antecedents = tf.expand_dims(tf.range(k), 1) - top_antecedent_offsets # [k, c]
    top_antecedents_mask = raw_top_antecedents >= 0 # [k, c]
    top_antecedents = tf.maximum(raw_top_antecedents, 0) # [k, c]

    return top_antecedents, top_antecedents_mask, top_antecedent_offsets

#     top_antecedents_mask = 
# [[False False False False False]
#  [ True False False False False]
#  [ True  True False False False]
#  [ True  True  True False False]
#  [ True  True  True  True False]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]]
# top_antecedents =                                             
# [[0 0 0 0 0]
#  [0 0 0 0 0]
#  [1 0 0 0 0]
#  [2 1 0 0 0]
#  [3 2 1 0 0]
#  [4 3 2 1 0]
#  [5 4 3 2 1]
#  [6 5 4 3 2]
#  [7 6 5 4 3]
#  [8 7 6 5 4]]                                                
                                           
  def get_top_labels_bridging(self, top_span_starts, top_span_ends, top_antecedents, top_antecedents_mask, gold_starts_anaphor, gold_ends_anaphor, gold_starts_antecedent, gold_ends_antecedent, gold_labels, top_antecedent_labels_prediction):  
  
    num_of_labels = util.shape(gold_starts_anaphor, 0)

    k = util.shape(top_antecedents, 0)
    c = util.shape(top_antecedents, 1)
    
    # predicted anaphor
    constructed_top_span_starts_anaphor = tf.tile(tf.expand_dims(top_span_starts, 1), [1, c*num_of_labels]) # [k, c*num_labeled]   
    constructed_top_span_ends_anaphor = tf.tile(tf.expand_dims(top_span_ends, 1), [1, c*num_of_labels]) # [k, c*num_labeled]
#    e.g. 
#    [[0 0 0 0 0]
# [1 1 1 1 1]
# [2 2 2 2 2]
# [3 3 3 3 3]
# [4 4 4 4 4]
# [5 5 5 5 5]]
    # flat it? 
    constructed_top_span_starts_anaphor = tf.reshape(constructed_top_span_starts_anaphor, [k*c*num_of_labels])
    constructed_top_span_ends_anaphor = tf.reshape(constructed_top_span_ends_anaphor, [k*c*num_of_labels])
    
    # predicted antecedent
    top_antecedents_starts = tf.gather(top_span_starts, top_antecedents)
    top_antecedents_ends = tf.gather(top_span_ends, top_antecedents) 
    constructed_top_span_starts_antecedent = tf.tile(tf.expand_dims(tf.reshape(top_antecedents_starts, [k*c]), 1), [1, num_of_labels])  # [k*c, num_labeled]
    constructed_top_span_ends_antecedent = tf.tile(tf.expand_dims(tf.reshape(top_antecedents_ends, [k*c]), 1), [1, num_of_labels])  # [k*c, num_labeled]
    # flat it?
    constructed_top_span_starts_antecedent = tf.reshape(constructed_top_span_starts_antecedent, [k*c*num_of_labels])
    constructed_top_span_ends_antecedent = tf.reshape(constructed_top_span_ends_antecedent, [k*c*num_of_labels])
    
    # there is the mask for antecedent!
    constructed_top_antecedents_mask = tf.tile(tf.expand_dims(tf.reshape(top_antecedents_mask, [k*c]), 1), [1, num_of_labels])  # [k*c, num_labeled]
    # flat it?
    constructed_top_antecedents_mask = tf.reshape(constructed_top_antecedents_mask, [k*c*num_of_labels])
    
    # gold anaphor 
    constructed_gold_span_starts_anaphor = tf.tile(gold_starts_anaphor, [k*c])
    constructed_gold_span_ends_anaphor = tf.tile(gold_ends_anaphor, [k*c])
    # flat it?  
    constructed_gold_span_starts_anaphor = tf.reshape(constructed_gold_span_starts_anaphor, [k*c*num_of_labels])
    constructed_gold_span_ends_anaphor = tf.reshape(constructed_gold_span_ends_anaphor, [k*c*num_of_labels])
    
    # gold antecedent 
    constructed_gold_span_starts_antecedent = tf.tile(gold_starts_antecedent, [k*c])
    constructed_gold_span_ends_antecedent = tf.tile(gold_ends_antecedent, [k*c])
    # flat it?  
    constructed_gold_span_starts_antecedent = tf.reshape(constructed_gold_span_starts_antecedent, [k*c*num_of_labels])
    constructed_gold_span_ends_antecedent = tf.reshape(constructed_gold_span_ends_antecedent, [k*c*num_of_labels])
    
    # broadcast the label as well? 
    constructed_gold_label = tf.tile(gold_labels, [k*c,1])
    #and flat it?
    constructed_gold_label = tf.reshape(constructed_gold_label, [k*c*num_of_labels, len(self.bridging_types)+1])

    #now we can do comparasion! 
    same_starts_anaphor = tf.equal(constructed_top_span_starts_anaphor, constructed_gold_span_starts_anaphor)
    same_ends_anaphor = tf.equal(constructed_top_span_ends_anaphor, constructed_gold_span_ends_anaphor)
    same_anaphor = tf.logical_and(same_starts_anaphor, same_ends_anaphor)
    
    same_starts_antecedent = tf.equal(constructed_top_span_starts_antecedent, constructed_gold_span_starts_antecedent)
    same_ends_antecedent = tf.equal(constructed_top_span_ends_antecedent, constructed_gold_span_ends_antecedent)
    same_antecedent = tf.logical_and(same_starts_antecedent, same_ends_antecedent)
    
    same_pair = tf.logical_and(same_anaphor, same_antecedent)  # [k*c*num_labeled]
    
    
    ## same_pair and antecedent mask:
    same_pair_with_mask = tf.logical_and(same_pair, constructed_top_antecedents_mask)
    filtered_gold_labels = tf.boolean_mask(constructed_gold_label, same_pair_with_mask) 
    
    # predicted labels
    constructed_top_antecedent_labels_prediction = tf.tile(tf.expand_dims(tf.reshape(top_antecedent_labels_prediction, [k*c, len(self.bridging_types)+1]), 1), [1, num_of_labels, 1])
    #and flat it ?
    constructed_top_antecedent_labels_prediction = tf.reshape(constructed_top_antecedent_labels_prediction, [k*c*num_of_labels, len(self.bridging_types)+1])   
    filtered_top_antecedent_labels_prediction = tf.boolean_mask(constructed_top_antecedent_labels_prediction, same_pair_with_mask) 
        

    # get negative sample 
    same_pair_no_redundancy = tf.reduce_max(tf.to_int32(tf.reshape(same_pair, [k*c, num_of_labels])), 1)   #  [k*c]
    # flat it again -  already is. tbh
    same_pair_no_redundancy = tf.reshape(same_pair_no_redundancy, [k*c]) #  [k*c]
    
    not_same_pair_no_redundancy = tf.logical_not(tf.cast(same_pair_no_redundancy, tf.bool)) #  [k*c]
    
    # need to mask it as well
    not_same_pair_no_redundancy_with_mask = tf.logical_and(not_same_pair_no_redundancy, tf.reshape(top_antecedents_mask, [k*c]))
    
    
  
    # need to control the "no gold" case  not_same_pair_no_redundancy_with_mask can be []
    not_same_pair_no_redundancy_with_mask = tf.where(tf.broadcast_to(tf.cast(num_of_labels, tf.bool), [k*c]), not_same_pair_no_redundancy_with_mask, tf.cast(tf.ones([k*c]), tf.bool))
   
    
    negative_top_antecedent_labels_prediction = tf.boolean_mask(tf.reshape(top_antecedent_labels_prediction, [k*c, len(self.bridging_types)+1]), not_same_pair_no_redundancy_with_mask)
    
    no_relation = [0 for i in range(len(self.bridging_types)+1)]
    no_relation[0] = 1
    
    negative_gold_labels = tf.boolean_mask(tf.broadcast_to(tf.constant(no_relation, tf.float32), [k*c, len(self.bridging_types)+1]), not_same_pair_no_redundancy_with_mask)
    
    # concat the fileter and negative 
    final_top_antecedent_labels_prediction = tf.concat([filtered_top_antecedent_labels_prediction, negative_top_antecedent_labels_prediction], 0)   #[k*c]
    
    final_gold_labels = tf.concat([filtered_gold_labels, negative_gold_labels], 0) #[k*c]
  
    return final_top_antecedent_labels_prediction, final_gold_labels
    

  def get_top_scores_coref(self, top_span_starts, top_span_ends, top_antecedents, top_antecedents_mask, gold_starts_anaphor, gold_ends_anaphor, gold_starts_antecedent, gold_ends_antecedent, top_antecedent_scores_prediction):     

    num_of_labels = util.shape(gold_starts_anaphor, 0)

    k = util.shape(top_antecedents, 0)
    c = util.shape(top_antecedents, 1)

    # predicted anaphor
    constructed_top_span_starts_anaphor = tf.tile(tf.expand_dims(top_span_starts, 1), [1, c*num_of_labels]) # [k, c*num_labeled]   
    constructed_top_span_ends_anaphor = tf.tile(tf.expand_dims(top_span_ends, 1), [1, c*num_of_labels]) # [k, c*num_labeled]
#    e.g. 
#    [[0 0 0 0 0]
# [1 1 1 1 1]
# [2 2 2 2 2]
# [3 3 3 3 3]
# [4 4 4 4 4]
# [5 5 5 5 5]]
    # flat it? 
    constructed_top_span_starts_anaphor = tf.reshape(constructed_top_span_starts_anaphor, [k*c*num_of_labels])
    constructed_top_span_ends_anaphor = tf.reshape(constructed_top_span_ends_anaphor, [k*c*num_of_labels])
    
    # predicted antecedent
    top_antecedents_starts = tf.gather(top_span_starts, top_antecedents)
    top_antecedents_ends = tf.gather(top_span_ends, top_antecedents) 
    constructed_top_span_starts_antecedent = tf.tile(tf.expand_dims(tf.reshape(top_antecedents_starts, [k*c]), 1), [1, num_of_labels])  # [k*c, num_labeled]
    constructed_top_span_ends_antecedent = tf.tile(tf.expand_dims(tf.reshape(top_antecedents_ends, [k*c]), 1), [1, num_of_labels])  # [k*c, num_labeled]
    # flat it?
    constructed_top_span_starts_antecedent = tf.reshape(constructed_top_span_starts_antecedent, [k*c*num_of_labels])
    constructed_top_span_ends_antecedent = tf.reshape(constructed_top_span_ends_antecedent, [k*c*num_of_labels])
    
    # gold anaphor 
    constructed_gold_span_starts_anaphor = tf.tile(gold_starts_anaphor, [k*c])
    constructed_gold_span_ends_anaphor = tf.tile(gold_ends_anaphor, [k*c])
    # flat it?  
    constructed_gold_span_starts_anaphor = tf.reshape(constructed_gold_span_starts_anaphor, [k*c*num_of_labels])
    constructed_gold_span_ends_anaphor = tf.reshape(constructed_gold_span_ends_anaphor, [k*c*num_of_labels])
    
    # gold antecedent 
    constructed_gold_span_starts_antecedent = tf.tile(gold_starts_antecedent, [k*c])
    constructed_gold_span_ends_antecedent = tf.tile(gold_ends_antecedent, [k*c])
    
    # flat it?  
    constructed_gold_span_starts_antecedent = tf.reshape(constructed_gold_span_starts_antecedent, [k*c*num_of_labels])
    constructed_gold_span_ends_antecedent = tf.reshape(constructed_gold_span_ends_antecedent, [k*c*num_of_labels])
    
    #now we can do comparasion! 
    same_starts_anaphor = tf.equal(constructed_top_span_starts_anaphor, constructed_gold_span_starts_anaphor)
    same_ends_anaphor = tf.equal(constructed_top_span_ends_anaphor, constructed_gold_span_ends_anaphor)
    same_anaphor = tf.logical_and(same_starts_anaphor, same_ends_anaphor)
    
    same_starts_antecedent = tf.equal(constructed_top_span_starts_antecedent, constructed_gold_span_starts_antecedent)
    same_ends_antecedent = tf.equal(constructed_top_span_ends_antecedent, constructed_gold_span_ends_antecedent)
    same_antecedent = tf.logical_and(same_starts_antecedent, same_ends_antecedent)
    
    same_pair = tf.logical_and(same_anaphor, same_antecedent)
    
    # get reduced sample 
    same_pair_no_redundancy = tf.reduce_max(tf.to_int32(tf.reshape(same_pair, [k*c, num_of_labels])), 1)   #  [k*c]
    # flat it again -  already is. tbh
    same_pair_no_redundancy = tf.reshape(same_pair_no_redundancy, [k*c]) #  [k*c] [int/float]
    same_pair_no_redundancy = tf.cast(same_pair_no_redundancy, tf.bool) # [k*c] [bool]
    
    # need to mask it 
    same_pair_no_redundancy_with_mask = tf.logical_and(same_pair_no_redundancy, tf.reshape(top_antecedents_mask, [k*c]))
    # shape it back
    same_pair_no_redundancy_with_mask = tf.reshape(same_pair_no_redundancy_with_mask, [k,c])
    
    
    # need to control the "no gold" case  same_pair_no_redundancy_with_mask can be []
    same_pair_no_redundancy_with_mask = tf.where(tf.broadcast_to(tf.cast(num_of_labels, tf.bool), [k,c]), same_pair_no_redundancy_with_mask, tf.cast(tf.zeros([k,c]), tf.bool))
    
    
    
    gold_scores = tf.where(same_pair_no_redundancy_with_mask, tf.cast(tf.ones([k,c]), tf.bool), tf.cast(tf.zeros([k,c]), tf.bool))  # [k, c] need bool
    
#     need to get a dummy for both
#     for the gold 
    gold_score_is_dummy = tf.reshape(tf.logical_not(tf.cast(tf.reduce_max(tf.to_int32(same_pair_no_redundancy_with_mask), 1), tf.bool)), [k,1])  #[k, 1]

    final_gold_scores = tf.concat([gold_score_is_dummy, gold_scores], 1) # [k, c + 1]  bool!

#     for the prediction  --- easy
    dummy_scores = tf.zeros([k, 1]) # [k, 1]   follow lee's setting - put dummy as 0
    final_top_antecedent_labels_prediction = tf.concat([dummy_scores, top_antecedent_scores_prediction], 1) # [k, c + 1]
    
    return final_top_antecedent_labels_prediction, final_gold_scores
    

  
  def get_top_span_mention_labels(self, top_span_starts, top_span_ends, gold_starts_anaphor, gold_ends_anaphor, gold_starts_antecedent, gold_ends_antecedent):
    
    same_start_in_anaphor = tf.equal(tf.expand_dims(top_span_starts, 1), tf.expand_dims(gold_starts_anaphor, 0)) # [k, num_labeled]
    same_end_in_anaphor = tf.equal(tf.expand_dims(top_span_ends, 1), tf.expand_dims(gold_ends_anaphor, 0)) # [k, num_labeled]
    same_span_in_anaphor = tf.logical_and(same_start_in_anaphor, same_end_in_anaphor) # [k, num_labeled]   
            
    same_start_in_antecedent = tf.equal(tf.expand_dims(top_span_starts, 1), tf.expand_dims(gold_starts_antecedent, 0)) # [k, num_labeled]
    same_end_in_antecedent= tf.equal(tf.expand_dims(top_span_ends, 1), tf.expand_dims(gold_ends_antecedent, 0)) # [k, num_labeled]
    same_span_in_antecedent= tf.logical_and(same_start_in_antecedent, same_end_in_antecedent) # [k, num_labeled]  
    
    same_span_intersection = tf.logical_or(same_span_in_anaphor, same_span_in_antecedent) # [k, num_labeled]
    same_span = tf.cast(tf.reduce_max(tf.to_int32(same_span_intersection), 1), tf.float32) # [k]
    
    # need to make sure it is the label cuz sometime gold is empty and then get a negative label which is bug 
    same_span = tf.math.maximum(same_span, tf.zeros([util.shape(same_span, 0)]))
    
    return same_span


  def get_predictions_and_loss(self, tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len, is_training, gold_starts_anaphor_bridging, gold_ends_anaphor_bridging, gold_starts_antecedent_bridging, gold_ends_antecedent_bridging, gold_labels_bridging, gold_starts_anaphor_coref, gold_ends_anaphor_coref, gold_starts_antecedent_coref, gold_ends_antecedent_coref, gold_all_span_starts_bridging, gold_all_span_ends_bridging, gold_all_span_starts_coref, gold_all_span_ends_coref, gold_all_span_starts, gold_all_span_ends):

    self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)
    self.lexical_dropout = self.get_dropout(self.config["lexical_dropout_rate"], is_training)
    self.lstm_dropout = self.get_dropout(self.config["lstm_dropout_rate"], is_training)

    num_sentences = tf.shape(context_word_emb)[0]
    max_sentence_length = tf.shape(context_word_emb)[1]
    
    context_emb_list = []
    head_emb_list = []
    
    if self.use_glove_or_w2v:
        print("-----------------using glove or w2v -----------------------")
        context_emb_list.append(context_word_emb)
        head_emb_list.append(head_word_emb)
    else:
        print("-----------------not using glove or w2v -----------------------")
    
    
    if self.config["char_embedding_size"] > 0:     
      char_emb = tf.gather(tf.get_variable("char_embeddings", [len(self.char_dict), self.config["char_embedding_size"]]), char_index) # [num_sentences, max_sentence_length, max_word_length, emb]
      flattened_char_emb = tf.reshape(char_emb, [num_sentences * max_sentence_length, util.shape(char_emb, 2), util.shape(char_emb, 3)]) # [num_sentences * max_sentence_length, max_word_length, emb]
      flattened_aggregated_char_emb = util.cnn(flattened_char_emb, self.config["filter_widths"], self.config["filter_size"]) # [num_sentences * max_sentence_length, emb]
      aggregated_char_emb = tf.reshape(flattened_aggregated_char_emb, [num_sentences, max_sentence_length, util.shape(flattened_aggregated_char_emb, 1)]) # [num_sentences, max_sentence_length, emb]
      context_emb_list.append(aggregated_char_emb)
      head_emb_list.append(aggregated_char_emb)

    if not self.lm_file:
      elmo_module = hub.Module("https://tfhub.dev/google/elmo/2")
      lm_embeddings = elmo_module(
          inputs={"tokens": tokens, "sequence_len": text_len},
          signature="tokens", as_dict=True)
      word_emb = lm_embeddings["word_emb"]  # [num_sentences, max_sentence_length, 512]
      lm_emb = tf.stack([tf.concat([word_emb, word_emb], -1),
                         lm_embeddings["lstm_outputs1"],
                         lm_embeddings["lstm_outputs2"]], -1)  # [num_sentences, max_sentence_length, 1024, 3]
    lm_emb_size = util.shape(lm_emb, 2)
    lm_num_layers = util.shape(lm_emb, 3)
    with tf.variable_scope("lm_aggregation"):
      self.lm_weights = tf.nn.softmax(tf.get_variable("lm_scores", [lm_num_layers], initializer=tf.constant_initializer(0.0)))
      self.lm_scaling = tf.get_variable("lm_scaling", [], initializer=tf.constant_initializer(1.0))

    flattened_lm_emb = tf.reshape(lm_emb, [num_sentences * max_sentence_length * lm_emb_size, lm_num_layers])
    flattened_aggregated_lm_emb = tf.matmul(flattened_lm_emb, tf.expand_dims(self.lm_weights, 1)) # [num_sentences * max_sentence_length * emb, 1]
    aggregated_lm_emb = tf.reshape(flattened_aggregated_lm_emb, [num_sentences, max_sentence_length, lm_emb_size])
                            
    aggregated_lm_emb *= self.lm_scaling
    context_emb_list.append(aggregated_lm_emb)

    context_emb = tf.concat(context_emb_list, 2) # [num_sentences, max_sentence_length, emb]
    head_emb = tf.concat(head_emb_list, 2) # [num_sentences, max_sentence_length, emb]
    
    context_emb = tf.nn.dropout(context_emb, self.lexical_dropout) # [num_sentences, max_sentence_length, emb]
    head_emb = tf.nn.dropout(head_emb, self.lexical_dropout) # [num_sentences, max_sentence_length, emb]

    text_len_mask = tf.sequence_mask(text_len, maxlen=max_sentence_length) # [num_sentence, max_sentence_length]
    
    sentence_indices = tf.tile(tf.expand_dims(tf.range(num_sentences), 1), [1, max_sentence_length]) # [num_sentences, max_sentence_length]

    flattened_sentence_indices = self.flatten_emb_by_sentence(sentence_indices, text_len_mask) # [num_words]
    flattened_head_emb = self.flatten_emb_by_sentence(head_emb, text_len_mask) # [num_words]
    
    context_outputs = self.lstm_contextualize(context_emb, text_len, text_len_mask) # [num_words, emb]

    num_words = util.shape(context_outputs, 0)

    if not self.provide_gold_mention_for_relation_prediction:  # need to predict mention first
        
        candidate_starts = tf.tile(tf.expand_dims(tf.range(num_words), 1), [1, self.max_span_width]) # [num_words, max_span_width]
        candidate_ends = candidate_starts + tf.expand_dims(tf.range(self.max_span_width), 0) # [num_words, max_span_width]
        candidate_start_sentence_indices = tf.gather(flattened_sentence_indices, candidate_starts) # [num_words, max_span_width]
        candidate_end_sentence_indices = tf.gather(flattened_sentence_indices, tf.minimum(candidate_ends, num_words - 1)) # [num_words, max_span_width]
        candidate_mask = tf.logical_and(candidate_ends < num_words, tf.equal(candidate_start_sentence_indices, candidate_end_sentence_indices)) # [num_words, max_span_width]
        flattened_candidate_mask = tf.reshape(candidate_mask, [-1]) # [num_words * max_span_width]

        candidate_starts = tf.boolean_mask(tf.reshape(candidate_starts, [-1]), flattened_candidate_mask) # [num_candidates]
        candidate_ends = tf.boolean_mask(tf.reshape(candidate_ends, [-1]), flattened_candidate_mask) # [num_candidates]
        candidate_sentence_indices = tf.boolean_mask(tf.reshape(candidate_start_sentence_indices, [-1]), flattened_candidate_mask) # [num_candidates]

        candidate_span_emb = self.get_span_emb(flattened_head_emb, context_outputs, candidate_starts, candidate_ends) # [num_candidates, emb]  num_candidates == k?

        candidate_mention_scores = self.get_mention_scores(candidate_span_emb) # [k, 1]

        candidate_mention_scores = tf.squeeze(candidate_mention_scores, 1) # [k]

        k = tf.to_int32(tf.floor(tf.to_float(tf.shape(context_outputs)[0]) * self.config["top_span_ratio"]))
    #     somehow the coref_ops.extract_spans keep the top span over the document, which only remain number of "tf.to_float(tf.shape(context_outputs)[0]) * self.config["top_span_ratio"]" span
        top_span_indices = coref_ops.extract_spans(tf.expand_dims(candidate_mention_scores, 0),
                                                   tf.expand_dims(candidate_starts, 0),
                                                   tf.expand_dims(candidate_ends, 0),
                                                   tf.expand_dims(k, 0),
                                                   util.shape(context_outputs, 0),
                                                   True) # [1, k]
        top_span_indices.set_shape([1, None])
        top_span_indices = tf.squeeze(top_span_indices, 0) # [k]

        top_span_starts = tf.gather(candidate_starts, top_span_indices) # [k]
        top_span_ends = tf.gather(candidate_ends, top_span_indices) # [k]
        top_span_emb = tf.gather(candidate_span_emb, top_span_indices) # [k, emb]
        top_span_mention_scores = tf.gather(candidate_mention_scores, top_span_indices) # [k]
        top_span_sentence_indices = tf.gather(candidate_sentence_indices, top_span_indices) # [k]

        c = tf.minimum(self.config["max_top_antecedents"], k)

        # for each span, consider its corresponding top_antecedents, coarse_to_fine use NN? and distance_pruning just previous c antecedents based on the distance                                                
        if self.config["coarse_to_fine"]:
          top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = self.coarse_to_fine_pruning(top_span_emb, top_span_mention_scores, c)
        else:
          top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = self.distance_pruning(top_span_emb, top_span_mention_scores, c)

        top_antecedent_emb = tf.gather(top_span_emb, top_antecedents) # [k, c, emb]               

  ## bridging_prediction
        top_antecedent_labels_prediction = self.get_slow_antecedent_predictions_bridging(top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets) # [k, c, bridging_label_size + 1]

        final_top_antecedent_labels_prediction, final_gold_labels = self.get_top_labels_bridging(top_span_starts, top_span_ends, top_antecedents, top_antecedents_mask, gold_starts_anaphor_bridging, gold_ends_anaphor_bridging, gold_starts_antecedent_bridging, gold_ends_antecedent_bridging, gold_labels_bridging, top_antecedent_labels_prediction)  # [filtered_gold_labels]
        
        
        loss_for_bridging = tf.keras.losses.categorical_crossentropy(final_gold_labels, final_top_antecedent_labels_prediction) 
        loss_for_bridging = tf.reduce_sum(loss_for_bridging) # []

  ## coref prediction
        top_antecedent_scores_prediction = self.get_slow_antecedent_predictions_coref(top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets) # [k, c]
        final_top_antecedent_scores_prediction, final_gold_scores = self.get_top_scores_coref(top_span_starts, top_span_ends, top_antecedents, top_antecedents_mask, gold_starts_anaphor_coref, gold_ends_anaphor_coref, gold_starts_antecedent_coref, gold_ends_antecedent_coref, top_antecedent_scores_prediction)  # need to get [k, c+1]
        
        loss_for_coref = self.softmax_loss(final_top_antecedent_scores_prediction, final_gold_scores) # [k]
        loss_for_coref = tf.reduce_sum(loss_for_coref) # []
        
  ## mention prediction/detection

        # detect whether the top_span is the span in relation ot not. 
        top_span_mention_label_in_bridging = self.get_top_span_mention_labels(top_span_starts, top_span_ends, gold_starts_anaphor_bridging, gold_ends_anaphor_bridging, gold_starts_antecedent_bridging, gold_ends_antecedent_bridging)
        
#         same_span = tf.cast(tf.reduce_max(tf.to_int32(same_span_intersection), 1), tf.float32) # [k]
        top_span_mention_label_in_coref = self.get_top_span_mention_labels(top_span_starts, top_span_ends, gold_starts_anaphor_coref, gold_ends_anaphor_coref, gold_starts_antecedent_coref, gold_ends_antecedent_coref)
    
        # merge these two: 
        top_span_mention_label = tf.reduce_max(tf.concat([tf.reshape(top_span_mention_label_in_bridging, [k,1]), tf.reshape(top_span_mention_label_in_coref, [k,1])], 1), 1)
               
    
        if self.training_setting == "joint_training":  # joint train coref and briding 
            
#      mention loss need mention from coref and bridging 
            # compare it with top_span_mention_scores        label is top_span_mention_label
            loss_for_mention = tf.nn.sigmoid_cross_entropy_with_logits(labels = top_span_mention_label, logits = top_span_mention_scores)
            loss_for_mention_reduce = tf.reduce_sum(loss_for_mention) # []
            # scale the weight of mention loss
            loss_for_mention_scale = tf.math.multiply(loss_for_mention_reduce, self.mention_loss_rate)
           
            loss_for_relation = tf.math.add(loss_for_bridging, loss_for_coref)
            loss = tf.math.add(loss_for_mention_scale, loss_for_relation)
            return [candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedents_mask, top_antecedent_labels_prediction, final_top_antecedent_scores_prediction],[loss_for_mention_scale, loss_for_relation, loss] 
            
        
        elif self.training_setting == "bridging":   # only has bridging
            # compare it with top_span_mention_scores        label is top_span_mention_label_in_bridging                                         
            loss_for_mention = tf.nn.sigmoid_cross_entropy_with_logits(labels = top_span_mention_label_in_bridging, logits = top_span_mention_scores)
            loss_for_mention_reduce = tf.reduce_sum(loss_for_mention) # []
            # scale the weight of mention loss
            loss_for_mention_scale = tf.math.multiply(loss_for_mention_reduce, self.mention_loss_rate)
            
            loss = tf.math.add(loss_for_bridging, loss_for_mention_scale)
            return [candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedents_mask, top_antecedent_labels_prediction], [loss_for_mention_scale, loss_for_bridging, loss]  
              
        elif self.training_setting == "coreference":  # only has coref
            # compare it with top_span_mention_scores        label is top_span_mention_label_in_bridging
            loss_for_mention = tf.nn.sigmoid_cross_entropy_with_logits(labels = top_span_mention_label_in_coref, logits = top_span_mention_scores)
            loss_for_mention_reduce = tf.reduce_sum(loss_for_mention) # []
            # scale the weight of mention loss
            loss_for_mention_scale = tf.math.multiply(loss_for_mention_reduce, self.mention_loss_rate)
            
            loss = tf.math.add(loss_for_coref, loss_for_mention_scale)
            return [candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedents_mask, final_top_antecedent_scores_prediction], [loss_for_mention_scale, loss_for_coref, loss] 
            
            
    
    else:  #self.provide_gold_mention_for_relation_prediction == true
        print("-------- just use gold_mention for relationship prediction ---------------")
        # need to change, separate the provide of coref and bridging gold mention
        if self.training_setting == "joint_training":
            top_span_emb = self.get_span_emb(flattened_head_emb, context_outputs, gold_all_span_starts, gold_all_span_ends)
            c = util.shape(gold_all_span_starts, 0)
            top_antecedents, top_antecedents_mask, top_antecedent_offsets = self.distance_pruning_for_gold_mention(top_span_emb, c)
            top_antecedent_emb = tf.gather(top_span_emb, top_antecedents) # [k, c, emb] 
            
            # mention 
            top_span_mention_label = self.get_top_span_mention_labels(gold_all_span_starts, gold_all_span_ends, gold_all_span_starts, gold_all_span_ends, gold_all_span_starts, gold_all_span_ends)
            top_span_mention_scores = self.get_mention_scores(top_span_emb) # [k, 1]
            top_span_mention_scores = tf.squeeze(top_span_mention_scores, 1) # [k]
            loss_for_mention = tf.nn.sigmoid_cross_entropy_with_logits(labels = top_span_mention_label, logits = top_span_mention_scores)
            loss_for_mention_reduce = tf.reduce_sum(loss_for_mention) # []
            # scale the weight of mention loss
            loss_for_mention_scale = tf.math.multiply(loss_for_mention_reduce, self.mention_loss_rate)
            
            # bridging
            top_antecedent_labels_prediction = self.get_slow_antecedent_predictions_bridging(top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets) # [k, c, bridging_label_size + 1]      
            final_top_antecedent_labels_prediction, final_gold_labels = self.get_top_labels_bridging(gold_all_span_starts, gold_all_span_ends, top_antecedents, top_antecedents_mask, gold_starts_anaphor_bridging, gold_ends_anaphor_bridging, gold_starts_antecedent_bridging, gold_ends_antecedent_bridging, gold_labels_bridging, top_antecedent_labels_prediction)  # [filtered_gold_labels]
            loss_for_bridging = tf.keras.losses.categorical_crossentropy(final_gold_labels, final_top_antecedent_labels_prediction) 
            loss_for_bridging = tf.reduce_sum(loss_for_bridging) # []
            # coref
            top_antecedent_scores_prediction = self.get_slow_antecedent_predictions_coref(top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets) # [k, c]
            final_top_antecedent_scores_prediction, final_gold_scores = self.get_top_scores_coref(gold_all_span_starts, gold_all_span_ends, top_antecedents, top_antecedents_mask, gold_starts_anaphor_coref, gold_ends_anaphor_coref, gold_starts_antecedent_coref, gold_ends_antecedent_coref, top_antecedent_scores_prediction)  # need to get [k, c+1]
            loss_for_coref = self.softmax_loss(final_top_antecedent_scores_prediction, final_gold_scores) # [k]
            loss_for_coref = tf.reduce_sum(loss_for_coref) # []
            
            loss_for_relation = tf.math.add(loss_for_bridging, loss_for_coref)
            loss = tf.math.add(loss_for_mention_scale, loss_for_relation)
            
            return [[], [], [], gold_all_span_starts, gold_all_span_ends, top_antecedents, top_antecedents_mask, top_antecedent_labels_prediction, final_top_antecedent_scores_prediction], [loss_for_mention_scale, loss_for_relation, loss] 
        
        elif self.training_setting == "bridging":
            top_span_emb = self.get_span_emb(flattened_head_emb, context_outputs, gold_all_span_starts_bridging, gold_all_span_ends_bridging)
            c = util.shape(gold_all_span_starts_bridging, 0)
            top_antecedents, top_antecedents_mask, top_antecedent_offsets = self.distance_pruning_for_gold_mention(top_span_emb, c)
            top_antecedent_emb = tf.gather(top_span_emb, top_antecedents) # [k, c, emb] 
            
            # mention 
            top_span_mention_label = self.get_top_span_mention_labels(gold_all_span_starts_bridging, gold_all_span_ends_bridging, gold_all_span_starts_bridging, gold_all_span_ends_bridging, gold_all_span_starts_bridging, gold_all_span_ends_bridging)
            top_span_mention_scores = self.get_mention_scores(top_span_emb) # [k, 1]
            top_span_mention_scores = tf.squeeze(top_span_mention_scores, 1) # [k]
            loss_for_mention = tf.nn.sigmoid_cross_entropy_with_logits(labels = top_span_mention_label, logits = top_span_mention_scores)
            loss_for_mention_reduce = tf.reduce_sum(loss_for_mention) # []
            # scale the weight of mention loss
            loss_for_mention_scale = tf.math.multiply(loss_for_mention_reduce, self.mention_loss_rate)
            
            
            # bridging
            top_antecedent_labels_prediction = self.get_slow_antecedent_predictions_bridging(top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets) # [k, c, bridging_label_size + 1]      
            final_top_antecedent_labels_prediction, final_gold_labels = self.get_top_labels_bridging(gold_all_span_starts_bridging, gold_all_span_ends_bridging, top_antecedents, top_antecedents_mask, gold_starts_anaphor_bridging, gold_ends_anaphor_bridging, gold_starts_antecedent_bridging, gold_ends_antecedent_bridging, gold_labels_bridging, top_antecedent_labels_prediction)  # [filtered_gold_labels]
            loss_for_bridging = tf.keras.losses.categorical_crossentropy(final_gold_labels, final_top_antecedent_labels_prediction) 
            loss_for_bridging = tf.reduce_sum(loss_for_bridging) # []
            
            loss = tf.math.add(loss_for_bridging, loss_for_mention_scale)
            
            return [[], [], [], gold_all_span_starts_bridging, gold_all_span_ends_bridging, top_antecedents, top_antecedents_mask, top_antecedent_labels_prediction], [loss_for_mention_scale, loss_for_bridging, loss]
            
        elif self.training_setting == "coreference":
            top_span_emb = self.get_span_emb(flattened_head_emb, context_outputs, gold_all_span_starts_coref, gold_all_span_ends_coref)
            c = util.shape(gold_all_span_starts_coref, 0)
            top_antecedents, top_antecedents_mask, top_antecedent_offsets = self.distance_pruning_for_gold_mention(top_span_emb, c)
            top_antecedent_emb = tf.gather(top_span_emb, top_antecedents) # [k, c, emb] 
            
            # mention 
            top_span_mention_label = self.get_top_span_mention_labels(gold_all_span_starts_coref, gold_all_span_ends_coref, gold_all_span_starts_coref, gold_all_span_ends_coref, gold_all_span_starts_coref, gold_all_span_ends_coref)
            top_span_mention_scores = self.get_mention_scores(top_span_emb) # [k, 1]
            top_span_mention_scores = tf.squeeze(top_span_mention_scores, 1) # [k]
            loss_for_mention = tf.nn.sigmoid_cross_entropy_with_logits(labels = top_span_mention_label, logits = top_span_mention_scores)
            loss_for_mention_reduce = tf.reduce_sum(loss_for_mention) # []
            # scale the weight of mention loss
            loss_for_mention_scale = tf.math.multiply(loss_for_mention_reduce, self.mention_loss_rate)
            
            # coref
            top_antecedent_scores_prediction = self.get_slow_antecedent_predictions_coref(top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets) # [k, c]
            final_top_antecedent_scores_prediction, final_gold_scores = self.get_top_scores_coref(gold_all_span_starts_coref, gold_all_span_ends_coref, top_antecedents, top_antecedents_mask, gold_starts_anaphor_coref, gold_ends_anaphor_coref, gold_starts_antecedent_coref, gold_ends_antecedent_coref, top_antecedent_scores_prediction)  # need to get [k, c+1]
            loss_for_coref = self.softmax_loss(final_top_antecedent_scores_prediction, final_gold_scores) # [k]
            loss_for_coref = tf.reduce_sum(loss_for_coref) # []
            
            loss = tf.math.add(loss_for_coref, loss_for_mention_scale)
            return [[], [], [], gold_all_span_starts_coref, gold_all_span_ends_coref, top_antecedents, top_antecedents_mask, final_top_antecedent_scores_prediction], [loss_for_mention_scale, loss_for_coref, loss]
            
            
    
  def get_span_emb(self, head_emb, context_outputs, span_starts, span_ends):
    span_emb_list = []
                            
    span_start_emb = tf.gather(context_outputs, span_starts) # [k, emb]
    span_emb_list.append(span_start_emb)
    
    span_end_emb = tf.gather(context_outputs, span_ends) # [k, emb]

    span_emb_list.append(span_end_emb)
    
    span_width = 1 + span_ends - span_starts # [k]

    if self.config["use_features"]:
      span_width_index = span_width - 1 # [k]
      span_width_emb = tf.gather(tf.get_variable("span_width_embeddings", [self.config["max_span_width"], self.config["feature_size"]]), span_width_index) # [k, emb]
      span_width_emb = tf.nn.dropout(span_width_emb, self.dropout)

      span_emb_list.append(span_width_emb)

    if self.config["model_heads"]:
      span_indices = tf.expand_dims(tf.range(self.config["max_span_width"]), 0) + tf.expand_dims(span_starts, 1) # [k, max_span_width]
      
      span_indices = tf.minimum(util.shape(context_outputs, 0) - 1, span_indices) # [k, max_span_width]
      span_text_emb = tf.gather(head_emb, span_indices) # [k, max_span_width, emb]
      
      with tf.variable_scope("head_scores"):
        self.head_scores = util.projection(context_outputs, 1) # [num_words, 1]
                            
      span_head_scores = tf.gather(self.head_scores, span_indices) # [k, max_span_width, 1]
                        
      span_mask = tf.expand_dims(tf.sequence_mask(span_width, self.config["max_span_width"], dtype=tf.float32), 2) # [k, max_span_width, 1]
      
      span_head_scores += tf.log(span_mask) # [k, max_span_width, 1]
      
      span_attention = tf.nn.softmax(span_head_scores, 1) # [k, max_span_width, 1]

      span_head_emb = tf.reduce_sum(span_attention * span_text_emb, 1) # [k, emb]
      span_emb_list.append(span_head_emb)    
    
    span_emb = tf.concat(span_emb_list, 1) # [k, emb]
                            
    return span_emb # [k, emb]

  def get_mention_scores(self, span_emb):
    with tf.variable_scope("mention_scores"):
      return util.ffnn(span_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # [k, 1]

  def softmax_loss(self, antecedent_scores, antecedent_labels):
    gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels)) # [k, max_ant + 1]
    marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1]) # [k]
    log_norm = tf.reduce_logsumexp(antecedent_scores, [1]) # [k]
    return log_norm - marginalized_gold_scores # [k]

  def bucket_distance(self, distances):
    """
    Places the given values (designed for distances) into 10 semi-logscale buckets:
    [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
    """
    logspace_idx = tf.to_int32(tf.floor(tf.log(tf.to_float(distances))/math.log(2))) + 3
    use_identity = tf.to_int32(distances <= 4)
    combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
    return tf.clip_by_value(combined_idx, 0, 9)

  def get_slow_antecedent_predictions_bridging(self, top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets):
    k = util.shape(top_span_emb, 0)
    c = util.shape(top_antecedents, 1)

    feature_emb_list = []

    if self.config["use_features"]:
      antecedent_distance_buckets = self.bucket_distance(top_antecedent_offsets) # [k, c]
      antecedent_distance_emb = tf.gather(tf.get_variable("antecedent_distance_emb_for_bridging", [10, self.config["feature_size"]]), antecedent_distance_buckets) # [k, c]
      feature_emb_list.append(antecedent_distance_emb)

    feature_emb = tf.concat(feature_emb_list, 2) # [k, c, emb]
    feature_emb = tf.nn.dropout(feature_emb, self.dropout) # [k, c, emb]

    target_emb = tf.expand_dims(top_span_emb, 1) # [k, 1, emb]
    similarity_emb = top_antecedent_emb * target_emb # [k, c, emb]
    target_emb = tf.tile(target_emb, [1, c, 1]) # [k, c, emb]

    pair_emb = tf.concat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2) # [k, c, emb]
    
#     ffnn come with softmax
    with tf.variable_scope("slow_antecedent_labels"):
      slow_antecedent_scores = util.ffnn_softmax_output(pair_emb, self.config["ffnn_depth"], self.config["ffnn_size"], len(self.bridging_types)+1, self.dropout) # [k, c, bridging_label_size+1]

    return slow_antecedent_scores # [k, c, bridging_label_size+1] or [k, c]

  def get_slow_antecedent_predictions_coref(self, top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets):
    k = util.shape(top_span_emb, 0)
    c = util.shape(top_antecedents, 1)

    feature_emb_list = []


    if self.config["use_features"]:
      antecedent_distance_buckets = self.bucket_distance(top_antecedent_offsets) # [k, c]
      antecedent_distance_emb = tf.gather(tf.get_variable("antecedent_distance_emb_for_coref", [10, self.config["feature_size"]]), antecedent_distance_buckets) # [k, c]
      feature_emb_list.append(antecedent_distance_emb)

    feature_emb = tf.concat(feature_emb_list, 2) # [k, c, emb]
    feature_emb = tf.nn.dropout(feature_emb, self.dropout) # [k, c, emb]

    target_emb = tf.expand_dims(top_span_emb, 1) # [k, 1, emb]
    similarity_emb = top_antecedent_emb * target_emb # [k, c, emb]
    target_emb = tf.tile(target_emb, [1, c, 1]) # [k, c, emb]

    pair_emb = tf.concat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2) # [k, c, emb]

#     ffnn without softmax
    with tf.variable_scope("slow_antecedent_scores"):
      slow_antecedent_scores = util.ffnn(pair_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # [k, c, 1]
    slow_antecedent_scores = tf.squeeze(slow_antecedent_scores, 2) # [k, c]

    return slow_antecedent_scores # [k, c, bridging_label_size+1] or [k, c]

  def get_fast_antecedent_scores(self, top_span_emb):
    with tf.variable_scope("src_projection"):
      source_top_span_emb = tf.nn.dropout(util.projection(top_span_emb, util.shape(top_span_emb, -1)), self.dropout) # [k, emb]
    target_top_span_emb = tf.nn.dropout(top_span_emb, self.dropout) # [k, emb]
    return tf.matmul(source_top_span_emb, target_top_span_emb, transpose_b=True) # [k, k]

  def flatten_emb_by_sentence(self, emb, text_len_mask):
    num_sentences = tf.shape(emb)[0]
    max_sentence_length = tf.shape(emb)[1]

    emb_rank = len(emb.get_shape())
    if emb_rank  == 2:
      flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length])
    elif emb_rank == 3:
      flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length, util.shape(emb, 2)])
    else:
      raise ValueError("Unsupported rank: {}".format(emb_rank))
    return tf.boolean_mask(flattened_emb, tf.reshape(text_len_mask, [num_sentences * max_sentence_length]))

  def lstm_contextualize(self, text_emb, text_len, text_len_mask):
    num_sentences = tf.shape(text_emb)[0]

    current_inputs = text_emb # [num_sentences, max_sentence_length, emb]

    for layer in range(self.config["contextualization_layers"]):
      with tf.variable_scope("layer_{}".format(layer)):
        with tf.variable_scope("fw_cell"):
          cell_fw = util.CustomLSTMCell(self.config["contextualization_size"], num_sentences, self.lstm_dropout)
        with tf.variable_scope("bw_cell"):
          cell_bw = util.CustomLSTMCell(self.config["contextualization_size"], num_sentences, self.lstm_dropout)
        state_fw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_fw.initial_state.c, [num_sentences, 1]), tf.tile(cell_fw.initial_state.h, [num_sentences, 1]))
        state_bw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_bw.initial_state.c, [num_sentences, 1]), tf.tile(cell_bw.initial_state.h, [num_sentences, 1]))

        (fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
          cell_fw=cell_fw,
          cell_bw=cell_bw,
          inputs=current_inputs,
          sequence_length=text_len,
          initial_state_fw=state_fw,
          initial_state_bw=state_bw)

        text_outputs = tf.concat([fw_outputs, bw_outputs], 2) # [num_sentences, max_sentence_length, emb]
        text_outputs = tf.nn.dropout(text_outputs, self.lstm_dropout)
        if layer > 0:
          highway_gates = tf.sigmoid(util.projection(text_outputs, util.shape(text_outputs, 2))) # [num_sentences, max_sentence_length, emb]
          text_outputs = highway_gates * text_outputs + (1 - highway_gates) * current_inputs
        current_inputs = text_outputs

    return self.flatten_emb_by_sentence(text_outputs, text_len_mask)

  def get_predicted_pairs_and_labels(self, top_span_starts, top_span_ends, top_antecedents, top_antecedents_mask, top_antecedent_labels_prediction):
    
    predicted_pairs_starts_anaphor = []
    predicted_pairs_ends_anaphor = []
    
    predicted_pairs_starts_antededent = []
    predicted_pairs_ends_antededent = []
    
    predicted_labels = []
    
    bridging_mapping = {i+1:br for i, br in enumerate(self.bridging_types)}
    
    no_relationship_count = 0
    has_relationship_count = 0
    skip_count = 0
    for row_index in range(len(top_antecedent_labels_prediction)):
      for col_index in range(len(top_antecedent_labels_prediction[row_index])):
        if top_antecedents_mask[row_index][col_index] == False: # if it is false, we need to skip it
            skip_count+=1
            continue
            
        #top_antecedent_labels_prediction[row_index][col_index] = e.g. [1,0,0,0,0]
        label_index = np.argmax(top_antecedent_labels_prediction[row_index][col_index])
        if label_index != 0:   # means has bridging relationship and 
          has_relationship_count+=1
          predicted_pairs_starts_anaphor.append(top_span_starts[row_index])
          predicted_pairs_ends_anaphor.append(top_span_ends[row_index])
          
          predicted_pairs_starts_antededent.append(top_span_starts[top_antecedents[row_index][col_index]])
          predicted_pairs_ends_antededent.append(top_span_ends[top_antecedents[row_index][col_index]])
    
          predicted_labels.append(bridging_mapping[label_index])
        elif label_index == 0:
            no_relationship_count+=1

    
    return [predicted_pairs_starts_anaphor, predicted_pairs_ends_anaphor, predicted_pairs_starts_antededent, predicted_pairs_ends_antededent, predicted_labels]

  def get_predicted_coref(self, top_span_starts, top_span_ends, top_antecedents, top_antecedents_mask, final_top_antecedent_scores_prediction):

    
    predicted_pairs_starts_anaphor = []
    predicted_pairs_ends_anaphor = []
    
    predicted_pairs_starts_antededent = []
    predicted_pairs_ends_antededent = []
    
    for row_index, max_index in enumerate(np.argmax(final_top_antecedent_scores_prediction, axis=1) - 1):
      if max_index < 0: # is dummy!
        continue
      else:
        predicted_pairs_starts_anaphor.append(top_span_starts[row_index])
        predicted_pairs_ends_anaphor.append(top_span_ends[row_index])
        predicted_pairs_starts_antededent.append(top_span_starts[top_antecedents[row_index][max_index]])
        predicted_pairs_ends_antededent.append(top_span_ends[top_antecedents[row_index][max_index]])
           
    predicted_labels = ["Coreference"]*len(predicted_pairs_starts_anaphor)
    
    return [predicted_pairs_starts_anaphor, predicted_pairs_ends_anaphor, predicted_pairs_starts_antededent, predicted_pairs_ends_antededent, predicted_labels]

  def load_eval_data(self):
    if self.eval_data is None:
      def load_line(line):
        example = json.loads(line)
        return self.tensorize_example(example, is_training=False), example
      with open(self.config["eval_path"]) as f:
        self.eval_data = [load_line(l) for l in f.readlines()]
      num_words = sum(tensorized_example[2].sum() for tensorized_example, _ in self.eval_data)
      print("Loaded {} eval examples.".format(len(self.eval_data)))
  
  def evaluate(self, session, official_stdout=False):
    self.load_eval_data()

    predicted_pairs_and_labels = {}
    predicted_coref = {}
    
    
    examples = []
    
    mention_loss = []
    relation_loss = []
    total_loss = []
    
    for example_num, (tensorized_example, example) in enumerate(self.eval_data):
   
      _, _, _, _, _, _, _, gold_starts_anaphor_bridging, gold_ends_anaphor_bridging, gold_starts_antecedent_bridging, gold_ends_antecedent_bridging, _, _, _, _, _, _, _, _, _, _, _ = tensorized_example
      feed_dict = {i:t for i,t in zip(self.input_tensors, tensorized_example)}


      [predictions, loss] = session.run([self.predictions, self.loss], feed_dict=feed_dict)
      

      mention_loss.append(loss[0])
      relation_loss.append(loss[1])
      total_loss.append(loss[2])
      
      top_span_starts = predictions[3]
      top_span_ends = predictions[4]
      top_antecedents = predictions[5]
      top_antecedents_mask = predictions[6]
      
      
      if self.training_setting == "joint_training":
          # bridging  
          top_antecedent_labels_prediction = predictions[7]
   
          predicted_pairs_and_labels[example["doc_key"]]  = self.get_predicted_pairs_and_labels(top_span_starts, top_span_ends, top_antecedents, top_antecedents_mask, top_antecedent_labels_prediction)  #  only mataining the pair that has briging relationship
    
          # coref      
          final_top_antecedent_scores_prediction = predictions[8]
          predicted_coref[example["doc_key"]]  = self.get_predicted_coref(top_span_starts, top_span_ends, top_antecedents, top_antecedents_mask, final_top_antecedent_scores_prediction)  #  for coref
      
      elif self.training_setting == "bridging":
          # bridging  
          top_antecedent_labels_prediction = predictions[7]
   
          predicted_pairs_and_labels[example["doc_key"]]  = self.get_predicted_pairs_and_labels(top_span_starts, top_span_ends, top_antecedents, top_antecedents_mask, top_antecedent_labels_prediction)  #  only mataining the pair that has briging relationship
      
      elif self.training_setting == "coreference":
          # coref      
          final_top_antecedent_scores_prediction = predictions[7]
          predicted_coref[example["doc_key"]]  = self.get_predicted_coref(top_span_starts, top_span_ends, top_antecedents, top_antecedents_mask, final_top_antecedent_scores_prediction)  #  for coref
          

      examples.append(example)
 

      if example_num % 10 == 0:
        print("Predicted {}/{} examples.".format(example_num + 1, len(self.eval_data)))
   
    print("\nfinal sum of mention, relation, total loss")
    print("{:.2f}, {:.2f}, {:.2f}".format(sum(mention_loss), sum(relation_loss), sum(total_loss)))
    print("final ave of mention and total loss")
    print("{:.2f}, {:.2f}, {:.2f} \n".format(sum(mention_loss)/len(mention_loss), sum(relation_loss)/len(relation_loss), sum(total_loss)/len(total_loss)))
    
    mention_results, relation_results = brat.evaluate_brat(self.config["brat_eval_tool_path"], self.config["brat_eval_path"], self.config["brat_token_path"], predicted_pairs_and_labels, predicted_coref, self.training_setting, examples, official_stdout)
    print("mention_results (Precision, Recall and F1):")
    print(mention_results)
    print("relation_results (Precision, Recall and F1):")
    print(relation_results)
 
    summary_dict = {}
    
    summary_dict["Average Mention Precision"] = mention_results["all"][0]
    summary_dict["Average Mention Recall"] = mention_results["all"][1]
    summary_dict["Average Mention F1"] = mention_results["all"][2]
    summary_dict["Average Relation Precision"] = relation_results["All"][0]
    summary_dict["Average Relation Recall"] = relation_results["All"][1]
    summary_dict["Average Relation F1"] = relation_results["All"][2]
    
    return util.make_summary(summary_dict), relation_results["All"][2]

