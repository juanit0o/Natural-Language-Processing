import xml.etree.ElementTree as ET
import matplotlib as plt
import pandas as pd
import numpy as np
import trec
import pprint as pp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import pickle
import os

def loadAux(): 
    Queries = "topics-2014_2015-summary.topics"
    Qrels = "qrels-clinical_trials.txt"

    with open(Queries, 'r') as queries_reader:
        txt = queries_reader.read()

    root = ET.fromstring(txt)

    dictionaryGendersQueries = dict()
    dictionaryAgesQueries = dict()

    cases = {}
    for query in root.iter('TOP'):
        q_num = query.find('NUM').text
        q_title = query.find('TITLE').text
        q_gender = query.find('GENDER').text
        q_age = query.find('AGE').text
        cases[q_num] = q_title
        dictionaryGendersQueries[q_num] = q_gender
        dictionaryAgesQueries[q_num] = q_age

        
        
    eval = trec.TrecEvaluation(cases, Qrels)
    #print(cases.values())
    vectorizerQueries = CountVectorizer(ngram_range=(1,1), analyzer = "word", stop_words = None)
    YQueries = vectorizerQueries.fit_transform(cases.values())

    print("Nr de pacientes (queries): " + str(len(cases)))
    return dictionaryGendersQueries, dictionaryAgesQueries, vectorizerQueries, YQueries, eval
    


#Fazer load dos ficheiros de ensaios clinicos e ids
def loadFiles():
    docsGender = pickle.load( open( "./documentsGender.bin", "rb" ) )
    ids = pickle.load( open( "./doc_ids.bin", "rb" ))
    docMaxAge = pickle.load( open( "./docMaxAge.bin", "rb" ) )
    docMinAge = pickle.load( open( "./docMinAge.bin", "rb" ) )
    
    docsGender.append("Female")
    dictionaryGendersDocs = dict(zip(ids, docsGender))

    ################### AGE ##################
    docMaxAge.append("N/A")
    docMinAge.append("N/A")
    #dicionario com minAge e maxAge para cada pessoa com docID
    dictionaryAgesDocs = dict(zip(ids,list(zip(docMinAge, docMaxAge))))
    return docsGender, ids, docMaxAge, docMinAge, dictionaryGendersDocs, dictionaryAgesDocs


#docsGender, ids, docMaxAge, docMinAge  = loadFiles()



#nr ids = 3626, tem o id NCT02634190
#nr genders = 3625
#print(len(docsGender)) #tem 1 a menos do que o id
#print(len(ids))
#docsGender nao tem o genero correspondente para a ultima query
############### ADICIONADO FEMALE PORQUE VISTO A MAO NOS DOCS

