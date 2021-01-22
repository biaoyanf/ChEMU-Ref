#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf
import anaphora_model as am
import util

import subprocess

if __name__ == "__main__":
  config = util.initialize_from_env()
  

  print('os.environ["CUDA_VISIBLE_DEVICES"]: ',os.environ["CUDA_VISIBLE_DEVICES"])
  
  report_frequency = config["report_frequency"]
  eval_frequency = config["eval_frequency"]
  
  
  run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

  
  model = am.AnaphoraModel(config)
  saver = tf.train.Saver() 

  log_dir = config["log_dir"]
  writer = tf.summary.FileWriter(log_dir, flush_secs=20)

  max_f1 = 0
  
  # store experiment in the correspinding logs/ location
  subprocess.run("cp experiments.conf %s/experiments.conf"%(log_dir),  shell=True, check=True)
    
  with tf.Session() as session:
  
    session.run(tf.global_variables_initializer(),options = run_opts)
    model.start_enqueue_thread(session)
    accumulated_loss = 0.0
    accumulated_relation_loss = 0.0
    accumulated_mention_loss = 0.0
    
    ckpt = tf.train.get_checkpoint_state(log_dir)
    if ckpt and ckpt.model_checkpoint_path:
      print("Restoring from: {}".format(ckpt.model_checkpoint_path))
      saver.restore(session, ckpt.model_checkpoint_path)

    initial_time = time.time()
    while True:
      tf_loss, tf_global_step, _ = session.run([model.loss, model.global_step, model.train_op])
      accumulated_loss += tf_loss[2]
      accumulated_relation_loss += tf_loss[1]
      accumulated_mention_loss += tf_loss[0]

      if tf_global_step % report_frequency == 0:
        total_time = time.time() - initial_time
        steps_per_second = tf_global_step / total_time

        average_loss = accumulated_loss / report_frequency
        average_relation_loss = accumulated_relation_loss / report_frequency
        average_mention_loss = accumulated_mention_loss / report_frequency
        print("[{}] mention loss = {:.2f}, relation loss = {:.2f}, loss={:.2f}, steps/s={:.2f}".format(tf_global_step, average_mention_loss, average_relation_loss, average_loss, steps_per_second))
        writer.add_summary(util.make_summary({"loss": average_loss}), tf_global_step)
        accumulated_loss = 0.0
        accumulated_relation_loss = 0.0
        accumulated_mention_loss = 0.0
        

      if tf_global_step % eval_frequency == 0:
        saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
        eval_summary, eval_f1 = model.evaluate(session)
        
        if eval_f1 > max_f1:
          max_f1 = eval_f1
          util.copy_checkpoint(os.path.join(log_dir, "model-{}".format(tf_global_step)), os.path.join(log_dir, "model.max.ckpt"))

        writer.add_summary(eval_summary, tf_global_step)
        writer.add_summary(util.make_summary({"max_eval_f1": max_f1}), tf_global_step)

        print("[{}] evaL_f1={:.2f}, max_f1={:.2f}".format(tf_global_step, eval_f1, max_f1))
        
#       control training step
      if tf_global_step > 10000:
        break