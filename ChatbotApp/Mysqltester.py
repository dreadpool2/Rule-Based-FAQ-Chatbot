import numpy as np 
import pandas as pd
import MySQLdb as mysq    
import tensorflow_hub as hub
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from absl import logging

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import sentencepiece as spm
import matplotlib.pyplot as plt
import numpy as np
import os
import re

#read data from DataStash.xlsx
dataset = pd.read_excel('DataStash.xlsx', index_col = 0)
print(dataset.head())


#init cursor which points to db
connec = mysq.connect(host = "localhost", user = "sanyog", password = "f20160635.stowe.pyc!")
cursor = connec.cursor()


#initialization of database and table
sql = 'CREATE DATABASE IF NOT EXISTS `Mammoth`'
cursor.execute(sql)
connec.commit()

sql = 'USE `Mammoth`'
cursor.execute(sql)
connec.commit()

sql = 'CREATE TABLE IF NOT EXISTS `Dataset` (`No.` TEXT, `Question` TEXT, `Answer` TEXT, `Question_Vector` BLOB)'
cursor.execute(sql)
connec.commit()

module = hub.Module("/var/www/html/ChatbotApp/universal-sentence-encoder-lite_1")
        
with tf.Session() as sess:
    spm_path = sess.run(module(signature="spm_path"))

sp = spm.SentencePieceProcessor()
sp.Load(spm_path)

#convert to vectorsx
def process_to_IDs_in_sparse_format(sentences):
          # An utility method that processes sentences with the sentence piece processor
          # 'sp' and returns the results in tf.SparseTensor-similar format:
          # (values, indices, dense_shape)
        ids = [sp.EncodeAsIds(x) for x in sentences]
        max_len = max(len(x) for x in ids)
        dense_shape=(len(ids), max_len)
        values=[item for sublist in ids for item in sublist]
        indices=[[row,col] for row in range(len(ids)) for col in range(len(ids[row]))]
        return (values, indices, dense_shape)
        

        ##load the model which converts user's query to vectors
        

input_placeholder = tf.sparse_placeholder(tf.int64, shape=[None, None])
embeddings = module(
    inputs=dict(
        values=input_placeholder.values,
        indices=input_placeholder.indices,
        dense_shape=input_placeholder.dense_shape))
values, indices, dense_shape = process_to_IDs_in_sparse_format(dataset['Question'])
with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    message_embeddings = session.run(
            embeddings,
            feed_dict={input_placeholder.values: values,
                            input_placeholder.indices: indices,
                            input_placeholder.dense_shape: dense_shape})

dataset['Question_Vector'] = dataset['Question_Vector'].astype(object)

for i in range(0, len(dataset['Question'])):
    dataset['Question_Vector'][i] = message_embeddings[i]
    
print(dataset['Question_Vector'].head())

#insert row by row into the database
for i, row in dataset.iterrows():
    sql = "INSERT INTO `Dataset` (`No.`, `Question`, `Answer`, `Question_Vector`) VALUES (%s, %s, %s, %s)"
    
    pdArr = row['Question_Vector'].dumps()
      
    cursor.execute(sql, (str(i), row[0], row[1], pdArr))
    
    connec.commit()
connec.close()



