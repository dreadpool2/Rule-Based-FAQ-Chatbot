import numpy as np 
import pandas as pd
import MySQLdb as mysq    
import tensorflow_hub as hub
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from absl import logging
import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import logging
import json
logging.warning('This will get logged to a file')
import sentencepiece as spm
import numpy as np

##load the model which converts user's query to vectors
##module = hub.Module("/var/www/html/ChatbotApp/universal-sentence-encoder-lite_1")    
  
graph = tf.Graph()
with tf.Session(graph = graph) as session:
    module = hub.Module("/var/www/html/ChatbotApp/universal-sentence-encoder-lite_1")       
    input_placeholder = tf.sparse_placeholder(tf.int64, shape=[None, None])
    embeddings = module(
        inputs=dict(
            values=input_placeholder.values,
            indices=input_placeholder.indices,
            dense_shape=input_placeholder.dense_shape))

    
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
        
        #convert pickle string to np array
        undo_pick_str = [np.loads(row) for row in self.dataset.Question_Vector]
        
        self.dataset.Question_Vector = undo_pick_str
        
        self.COSINE_THRESHOLD = 0.5
                
        ##Create a Chatbot which answers separate personal unrelated questions. Here we have models related to politics, sports etc. 
        self.chitchat_bot = ChatBot(
            'Chatterbot',
            storage_adapter='chatterbot.storage.SQLStorageAdapter', 
            database_uri='sqlite:////var/www/html/ChatbotApp/db.sqlite3'
        ) 
        
        
    def embed(self,input):
       
        with tf.Session(graph = graph) as sess:
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
    

        values, indices, dense_shape = process_to_IDs_in_sparse_format(input)
        with tf.Session(graph = graph) as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            message_embeddings = session.run(
                  embeddings,
                  feed_dict={input_placeholder.values: values,
                            input_placeholder.indices: indices,
                            input_placeholder.dense_shape: dense_shape})
        
        return message_embeddings
        
    def cosine_similarity(self,v1, v2):
      
        return np.inner(v1, v2) 
        
    def semantic_search(self, query):    
        ##convert the query to vectors and perform cosine similarity for all rows vectors and query vectors.
        '''
        query_vec = self.embed(query)
        res = []
        for i in range(0, self.dataset.size):
            qvec = self.dataset.Question_Vector[i].ravel()
            sim = self.cosine_similarity(query_vec[0], qvec)
            res.append((sim, i))
        
        '''
        df2 = self.dataset
        
        def func(query_vec, que_vec):
            qvec = que_vec.ravel()
            sim = self.cosine_similarity(query_vec[0], qvec)
            return sim
            
        #start = time.process_time()
        query_vec = self.embed(query)
        #return (time.process_time() - start)

        #df2['Similarity'] = func(query_vec, df2['Question_Vector'].values)
        
        # your code here    
        df2['Similarity'] = df2.apply(lambda row : func(query_vec, row['Question_Vector']), axis = 1)
        
        df2.sort_values(by=['Similarity'], ascending = False, inplace  = True)
        
        return df2
        
    
    def generate_faq_answer(self, question):
        a = list()
        a.append(question)
        
        div = "|"
        
        
        most_relevant_row = self.semantic_search(a)
    
        
        if most_relevant_row.iloc[0][4]>=self.COSINE_THRESHOLD:
            answer =  "0" + div + most_relevant_row.iloc[0][2] + div + ""
        else:
            answer = "-1" + div + "Did you mean to ask <span style='color:white'><br><u>"+most_relevant_row.iloc[0][1]+"</u></span><br><br> Type <span style='font-weight:1000 !important'>(Yes&#124;No&#124;AgentHelp)</span>" + div + most_relevant_row.iloc[0][2] + div + question
            ##answer = self.chitchat_bot.get_response(question)
        
            
        return answer
    
    
    def get_agent_help(self, user, question):
        divider = "|"
        connec = mysq.connect(host = "localhost", user = "sanyog", password = "f20160635.stowe.pyc!")
   

        cursor = connec.cursor()
        sql = 'USE `Mammoth`'
        cursor.execute(sql)
        connec.commit()
                
        sql = "INSERT INTO `AgentHelp` (`Divider`, `User`, `Question`, `AgentAnswer`, `Status`) VALUES (%s, %s, %s, %s, %s)"
        cursor.execute(sql, (divider, user, question, "", "open"));
        connec.commit()
        
        ##The query has been pushed to the database. The agent will now select the query.
        
        connec.close()

    
    def check_replies(self, user):
        connec = mysq.connect(host = "localhost", user = "sanyog", password = "f20160635.stowe.pyc!")
   

        cursor = connec.cursor()
        sql = 'USE `Mammoth`'
        cursor.execute(sql)
        connec.commit()
                
        
        ##The query has been pushed to the database. The agent will now select the query.
        
        
        
        df2 = pd.read_sql("SELECT `AgentAnswer` AS `Ag`,`Status` AS `St` FROM AgentHelp WHERE `User` = '"+str(user)+"'", con=connec)
        
        
        
        if(df2.empty == False and df2.iloc[0][1] == 'closed'):
            sql = "DELETE FROM AgentHelp WHERE `User` = '"+str(user)+"'"
            cursor.execute(sql)
            connec.commit()
        
        
        connec.close()


        return df2.iloc[0][0]
        
    ##BELOW FUNCTIONS RELATED TO AGENTS
    def get_all_data(self):
        
        connec = mysq.connect(host = "localhost", user = "sanyog", password = "f20160635.stowe.pyc!")
   

        cursor = connec.cursor()
        sql = 'USE `Mammoth`'
        cursor.execute(sql)
        connec.commit()
        
        df2 = pd.read_sql('SELECT * FROM AgentHelp', con=connec)
  
 
        connec.close()
        
        
        #dictionary  = {'ids': df2['User'], 'questions' : df2['Question'], 'statuses' : df2['Status']};
        
        return str(df2.to_json());    

    
    def update_data_agent(self, user, answer, status):
        connec = mysq.connect(host = "localhost", user = "sanyog", password = "f20160635.stowe.pyc!")
   

        cursor = connec.cursor()
        sql = 'USE `Mammoth`'
        cursor.execute(sql)
        connec.commit()
                
        sql = "UPDATE AgentHelp SET `AgentAnswer`=%s, `Status`=%s WHERE `User`=%s"        
        cursor.execute(sql, (str(answer), str(status), str(user)));
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
     
    def set_ongoing(self, id):
        connec = mysq.connect(host = "localhost", user = "sanyog", password = "f20160635.stowe.pyc!")
   

        cursor = connec.cursor()
        sql = 'USE `Mammoth`'
        cursor.execute(sql)
        connec.commit()
                
        df = pd.read_sql("SELECT `Status` FROM AgentHelp WHERE `User`='"+str(id)+"'", con=connec)
        
        if(df.empty):
            connec.close()    
            return "Query already solved!"  
        elif(df.iloc[0][0] == "Ongoing"):
            connec.close()    
            return "Failure"
        else:
            #Set status ongoing
            cursor.execute("UPDATE AgentHelp SET `Status`='Ongoing'  WHERE `User`='"+str(id)+"'");
            connec.commit()
            connec.close()    
            return "Success"
        
        return "Success"
            
        
        
    ## Below state admin portal functions

    
    def add_data(self, ques, ans):
        a = list()
        a.append(ques)
        
        connec = mysq.connect(host = "localhost", user = "sanyog", password = "f20160635.stowe.pyc!")
   

        cursor = connec.cursor()
        sql = 'USE `Mammoth`'
        cursor.execute(sql)
        connec.commit()
                
        df = pd.read_sql("SELECT COUNT(*) FROM Dataset", con=connec)
        
        count = df.iloc[0][0]
        
        
        
        sql = "INSERT INTO `Dataset` (`No.`, `Question`, `Answer`, `Question_Vector`) VALUES (%s, %s, %s, %s)"   
        
        query_vec = self.embed(a)
        pdArr = query_vec[0].dumps()
        
        cursor.execute(sql, (count, ques, ans, pdArr))
        connec.commit()
        connec.close()