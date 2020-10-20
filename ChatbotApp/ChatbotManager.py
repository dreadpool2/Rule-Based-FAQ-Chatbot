import numpy as np 
import pandas as pd
import MySQLdb as mysq    
import tensorflow_hub as hub
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from absl import logging

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import logging

logging.warning('This will get logged to a file')
import sentencepiece as spm
import numpy as np


class ChatbotManager():
    
        
    def __init__(self):
 
            #init cursor which points to db
        connec = mysq.connect(host = "localhost", user = "sanyog", password = "f20160635.stowe.pyc!")
   

        cursor = connec.cursor()
        sql = 'USE `Mammoth`'
        cursor.execute(sql)
        connec.commit()
        
        df = pd.read_sql('SELECT * FROM Dataset', con=connec)
        
        print(df.head())
        
        connec.close()
        
        
        #self.model = tf.saved_model.load("../data/tmp/mobilenet/1/")
        self.dataset = df
        self.questions = self.dataset.Question
        
        #convert pickle string to np array
        undo_pick_str = [np.loads(row) for row in self.dataset.Question_Vector]
        self.QUESTION_VECTORS = undo_pick_str
    
        self.COSINE_THRESHOLD = 0.3
                
        ##Create a Chatbot which answers separate personal unrelated questions. Here we have models related to politics, sports etc. 
        self.chitchat_bot = ChatBot(
            'Chatterbot',
            storage_adapter='chatterbot.storage.SQLStorageAdapter', 
            database_uri='sqlite:////var/www/html/ChatbotApp/db.sqlite3'
        ) 
        
    def embed(self,input):
        ##load the model which converts user's query to vectors
        module = hub.Module("/var/www/html/ChatbotApp/universal-sentence-encoder-lite_1")
        
        
        with tf.Session() as sess:
            spm_path = sess.run(module(signature="spm_path"))
        sp = spm.SentencePieceProcessor()
        sp.Load(spm_path)
        
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
    

        input_placeholder = tf.sparse_placeholder(tf.int64, shape=[None, None])
        embeddings = module(
            inputs=dict(
                values=input_placeholder.values,
                indices=input_placeholder.indices,
                dense_shape=input_placeholder.dense_shape))

        values, indices, dense_shape = process_to_IDs_in_sparse_format(input)
        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            message_embeddings = session.run(
                  embeddings,
                  feed_dict={input_placeholder.values: values,
                            input_placeholder.indices: indices,
                            input_placeholder.dense_shape: dense_shape})
        
        return message_embeddings
        
    def cosine_similarity(self,v1, v2):
      
        return np.inner(v1, v2) 
        
        
    def semantic_search(self, query, data, vectors):    
        ##convert the query to vectors and perform cosine similarity between all rows vectors and query vectors.
        
        query_vec = self.embed(query)
        res = []
        for i, d in enumerate(data):
            qvec = vectors[i].ravel()
            sim = self.cosine_similarity(query_vec[0], qvec)
            res.append((sim, d[:100], i))

        
        return sorted(res, key=lambda x : x[0], reverse=True)    

    
    def generate_faq_answer(self, question):
        a = list()
        a.append(question)
        
        div = "|"
        
        
        most_relevant_row = self.semantic_search(a, self.questions, self.QUESTION_VECTORS)
    
        if most_relevant_row[0][0]>=self.COSINE_THRESHOLD:
            answer =  "0" + div + self.dataset.Answer[most_relevant_row[0][2]] + div + ""
        else:
            '''
            q_vec = self.embed(a)
            cAns = self.chitchat_bot.get_response(question)
            d = list()
            d.append(str(cAns))
            
            vex = self.embed(d)
            sim = self.cosine_similarity(q_vec[0][:5], vex[0][:5])
            
            
            stre = "Hello"+str(sim)+"Helo"
            answer =  stre+div+stre+div+stre
            '''
            answer = "-1" + div + "Did you mean to ask <span style='color:white'><br><u>"+most_relevant_row[0][1]+"</u></span> (Yes/No)" + div + self.dataset.Answer[most_relevant_row[0][2]]
            ##answer = self.chitchat_bot.get_response(question)
            
            
        return answer
    
    
    def get_agent_help(self, tStmp, question):
        tSt = tStmp
        user = "sanyog"
        questioned = question
        
        connec = mysq.connect(host = "localhost", user = "sanyog", password = "f20160635.stowe.pyc!")
   

        cursor = connec.cursor()
        sql = 'USE `Mammoth`'
        cursor.execute(sql)
        connec.commit()
                
        sql = "INSERT INTO `AgentHelp` (`Timestamp`, `User`, `Question`, `AgentAnswer`, `Status`) VALUES (%s, %s, %s, %s, %s)"
        cursor.execute(sql, (tSt, user, questioned, "", "open"));
        connec.commit()
        
        ##The query has been pushed to the database. The agent will now select the query.
        
        connec.close()

    
    def check_replies(self, timeStamp):
        connec = mysq.connect(host = "localhost", user = "sanyog", password = "f20160635.stowe.pyc!")
   

        cursor = connec.cursor()
        sql = 'USE `Mammoth`'
        cursor.execute(sql)
        connec.commit()
                
        
        ##The query has been pushed to the database. The agent will now select the query.
        
        
        
        df2 = pd.read_sql("SELECT `AgentAnswer` AS  `AgentAnswer` FROM AgentHelp WHERE `Timestamp` = '"+str(timeStamp)+"'", con=connec)
        connec.close()


        return df2.iloc[0][0];
        
    ##BELOW FUNCTIONS RELATED TO AGENTS
    def get_all_data(self):
        
        connec = mysq.connect(host = "localhost", user = "sanyog", password = "f20160635.stowe.pyc!")
   

        cursor = connec.cursor()
        sql = 'USE `Mammoth`'
        cursor.execute(sql)
        connec.commit()
        
        df2 = pd.read_sql('SELECT * FROM AgentHelp', con=connec)
  
 
        connec.close()

        return df2;    

    
    def update_data_agent(self, timeStamp, answer, status):
        connec = mysq.connect(host = "localhost", user = "sanyog", password = "f20160635.stowe.pyc!")
   

        cursor = connec.cursor()
        sql = 'USE `Mammoth`'
        cursor.execute(sql)
        connec.commit()
                
        
        sql = "UPDATE AgentHelp SET `AgentAnswer`='"+answer+"' WHERE `Timestamp` = '"+timeStamp+"'"
        cursor.execute(sql);
        connec.commit()
        
        connec.close()
        
        
    def drop_all_queries(self):
        connec = mysq.connect(host = "localhost", user = "sanyog", password = "f20160635.stowe.pyc!")
   

        cursor = connec.cursor()
        sql = 'USE `Mammoth`'
        cursor.execute(sql)
        connec.commit()
                
        
        sql = "TRUNCATE AgentHelp"
        cursor.execute(sql);
        connec.commit()
        
        connec.close()
    
    