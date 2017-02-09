# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 14:11:03 2017

@author: Kartikeya
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 13:04:50 2017

@author: Kartikeya
"""
import web
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim import corpora,models, similarities
from nltk.corpus import stopwords
import glob
import os
from os.path import basename
from azure.storage.blob import BlockBlobService
import urllib2  # the lib that handles the url stuff
import codecs
import json
import inflect


names = []
block_blob_service = BlockBlobService(account_name='saoilandgas', account_key='+YlGC9UHQwGuzpUFHkr83oSsHtu9KGFLRuYybmgIXMHZfHwhF3L8d8hgXSAlcYVqBqVMH2/RDqCbltCnmi6Vew==')
    
generator = block_blob_service.list_blobs('oilandgascorpus')
for blob in generator:
    names.append(blob.name)

    
url = 'https://saoilandgas.blob.core.windows.net/oilandgascorpus/'
    
names = [ x for x in names if ".jpeg" not in x ]
names = [ x for x in names if ".jpg" not in x ]
names = [ x for x in names if ".png" not in x ]
names = [ x for x in names if ".dict" not in x ]
names = [ x for x in names if ".mm" not in x ]
names = [ x for x in names if ".lsi" not in x ]

dictionary = corpora.Dictionary.load('deerwester.dict')
corpus = corpora.MmCorpus('deerwester.mm')

# same for tfidf, lda, ...
lsi = models.LsiModel.load('model.lsi')



from flask import Flask, request
app = Flask(__name__)


@app.route('/todo/api/v1.0/tasks/<string:task_id>', methods=['GET'])
def rank_docs(task_id):
#    doc = request.args.get('doc')
    doc = task_id
    vec_bow = dictionary.doc2bow(doc.lower().split())
    vec_lsi = lsi[vec_bow] # convert the query to LSI space
    # print(vec_lsi)
    index = similarities.MatrixSimilarity(lsi[corpus])

    sims = index[vec_lsi]
    # perform a similarity query against the corpus
    #print(list(enumerate(sims)))

    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    doc_ids = [t[0] for t in sims]
    print (doc_ids)
    doc_count = len(doc_ids)
    doc_names = [url + names[doc_ids[i]] for i in range(0,doc_count)]
    print (doc_names)
    sims_json = [{'scores':str(i[1])} for i in sims]    
    [sims_json[j].update({'rank': j+1,'url': doc_names[j]}) for j in doc_ids] 
    data=json.dumps(sims_json,sort_keys=True,indent=4)
    print(data)
    return data
    
if __name__ == "__main__":
    app.run()