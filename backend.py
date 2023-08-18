
import pandas as pd
import string
import re

import spacy
from spacy.lang.en import English

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

import csv
import langid
import shutil
import os
import concurrent.futures
from tqdm import tqdm
import math
from concurrent.futures import ThreadPoolExecutor


# In[29]:


#text preprocess, input: string, output string
def preprocess_text(text,stop_words):
    if pd.isna(text)==True:
        return ''
    
    # Tokenize sentence
    text = re.sub(r'[-_/]',' ',text)

    tokens = word_tokenize(text)

    # Remove stopwords
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    
    # Lowercasing
    filtered_tokens = [token.lower() for token in filtered_tokens]
    
    # Stemming tokens
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    
    '''
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    '''

    # Removing punctuation and special characters
    tokens=[re.sub(r'[^a-zA-Z0-9]','',token) for token in stemmed_tokens]
    
    # Remove empty tokens
    tokens = [token for token in tokens if token]
    
    # Join tokens back into sentence
    preprocessed_text = " ".join(tokens)
    return preprocessed_text


# In[30]:


def read_inverted_index(path):
    inverted_index={}
    with open(path, 'r') as f:
        # read the file line by line
        for line in f:
            # strip leading and trailing whitespace from the line
            line = line.strip()

            # if the line ends with a column, it is a token
            if line.endswith(':'):
                # extract the token and create an empty dictionary for it
                token = line[:-1]
                inverted_index[token] = {}

            # otherwise, it is a document index and its positions
            else:
                # split the line into the document index and its positions
                index, positions = line.split(': ')

                # convert the index to an integer
                index = int(index)

                # split the positions into a list of integers
                positions = [int(pos) for pos in positions.split(',')]

                # add the document index and its positions to the inverted index for the current token
                inverted_index[token][index] = positions
    return inverted_index


# In[31]:


# document length is defined as the square root of the sum of the squares of the term frequencies of all terms in the document.
def generate_document_lengths(inverted_index):
    document_lengths = {}
    for term in inverted_index:
        for doc_id in inverted_index[term]:
            if doc_id not in document_lengths:
                document_lengths[doc_id] = 0
            tf = len(inverted_index[term][doc_id])
            document_lengths[doc_id] += tf**2
    for doc_id in document_lengths:
        document_lengths[doc_id] = math.sqrt(document_lengths[doc_id])
    return document_lengths

def tfidf(query, inverted_index, document_lengths, num_documents):
    # split query into terms
    query_terms = query.split()
    
    # compute document frequencies for query terms
    query_doc_freqs = {}
    for term in query_terms:
        if term in inverted_index:
            query_doc_freqs[term] = len(inverted_index[term])
        else:
            query_doc_freqs[term] = 0
    
    # compute query weights using tf-idf scheme
    query_weights = {}
    for term, freq in query_doc_freqs.items():
        if freq == 0:
            continue
        idf = math.log(num_documents/freq)
        query_weights[term] = (1 + math.log(freq)) * idf
    
    # compute document scores using tf-idf scheme
    doc_scores = {}
    with ThreadPoolExecutor() as executor:
        futures = []
        for term, weight in query_weights.items():
            if term in inverted_index:
                postings = inverted_index[term]
                for doc_id, positions in postings.items():
                    future = executor.submit(compute_score, doc_id, positions, weight, num_documents, len(postings), document_lengths)
                    futures.append(future)
        for future in futures:
            doc_id, score = future.result()
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0
            doc_scores[doc_id] += score
    
    # sort documents by descending score
    ranked_documents = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    
    return ranked_documents

def compute_score(doc_id, positions, weight, num_documents, num_postings, document_lengths):
    tf = len(positions)
    idf = math.log(num_documents/num_postings)
    tf_idf = (1 + math.log(tf)) * idf
    score = tf_idf * weight / document_lengths[doc_id]
    return (doc_id, score)


# In[33]:


#TFIDF score normalization
def normalize_scores(ranked_documents):
    if not ranked_documents:
        return []
    # find maximum score among all documents
    max_score = max([score for doc_id, score in ranked_documents])
    
    # divide all scores by maximum score
    normalized_documents = [(doc_id, score/max_score) for doc_id, score in ranked_documents]
    
    return normalized_documents

#Weigh the score of info and review to 7:3
def weighted_average(ranked_documents1, ranked_documents2):
    # normalize scores of both result lists
    normalized_documents1 = normalize_scores(ranked_documents1)
    normalized_documents2 = normalize_scores(ranked_documents2)
    
    # create dictionary of document scores for both lists
    scores_dict = {}
    for doc_id, score in normalized_documents1:
        scores_dict[doc_id] = 0.7 * score
    for doc_id, score in normalized_documents2:
        if doc_id in scores_dict:
            scores_dict[doc_id] += 0.3 * score
        else:
            scores_dict[doc_id] = 0.3 * score
    
    # sort documents by descending score
    ranked_documents = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    
    return ranked_documents


# In[34]:


def rank_retrieve(preprocessed_query, info_inverted_index,info_length,review_inverted_index,review_length):
    info_ranked_documents = tfidf(preprocessed_query,info_inverted_index,info_length, len(info_length))
    review_ranked_documents = tfidf(preprocessed_query,review_inverted_index,review_length, len(review_length))
    result = [doc_id for doc_id, score in weighted_average(info_ranked_documents,review_ranked_documents)]
    return result


# In[35]:


'''return dict '''
def single_word_search(term, inverted_index):
    term = term[0]
    result = {}
    if term in inverted_index:
        documents = inverted_index[term]
        for doc_id, positions in documents.items():
            result[doc_id] = {'freq': len(positions), 'pos': positions}
        return result
    else:
        return None


# In[36]:


'''return dict '''

def phrase_search(terms, inverted_index):
    if all(term in inverted_index for term in terms):
        # Find all documents that contain all the terms
        doc_sets = [set(inverted_index[term].keys()) for term in terms]
        docs = set.intersection(*doc_sets)
        # For each document, check if the phrase appears in the correct order
        results = {}
        for doc in docs:
            positions = [inverted_index[term][doc] for term in terms]
            for i in positions[0]:
                if all(positions[j].count(i+j) > 0 for j in range(1, len(terms))):
                    result_pos = [[i+j for j in range(len(terms))] for i in positions[0] if all(positions[j].count(i+j) > 0 for j in range(1, len(terms)))]
                    freq = len(result_pos)
                    results[doc] = {'freq': freq, 'pos': result_pos}
        return results
    else:
        return None


# In[37]:


def phrase_tfidf(phrase,info_inverted_index,review_inverted_index,info_length,review_length,stop_words):
    # Regular expression pattern to match a phrase search query
    pattern = r'"[^"]+"'
    if bool(re.match(pattern,phrase))==True:
        # Find all matches of the pattern in the query string
        matches = re.findall(pattern, phrase)
        result_collection = []
        # Print all matches
        for match in matches:
            text = preprocess_text(match,stop_words)
            text = text.split()
            if len(text)==0:
                result_collection.append([])
            elif len(text)==1:
                result = single_word_search(text,info_inverted_index)
                result = list(result.keys())
                result_collection.append(result)
            else:
                result = phrase_search(text,info_inverted_index)
                result = list(result.keys())
                result_collection.append(result)
        #tfidf
        rank_result = rank_retrieve(
            preprocess_text(phrase,stop_words),info_inverted_index,info_length,review_inverted_index,review_length)
        if len(result_collection)!=0:
            common_elements = list(set(result_collection[0]).intersection(*result_collection[1:]))
            # create a dictionary with the rank of each element in common_elements
            common_elements.sort(key=lambda x: rank_result.index(x))
            final_result = common_elements
        else:
            final_result = []
    else:
        #tfidf
        rank_result = rank_retrieve(
            preprocess_text(phrase,stop_words),info_inverted_index,info_length,review_inverted_index,review_length)    
        final_result=rank_result
    return final_result


# In[38]:


'''term can be ['machin'](single word) or ['machin', 'learn'](phrase)'''
def proximity_search(term_0, term_1, distance, inverted_index):
    if not term_0 or not term_1:
        return []
    else:
        if len(term_0) == 1:
            result_0 = single_word_search(term_0,inverted_index)
        else:
            result_0 = phrase_search(term_0,inverted_index)
            result_0 = {doc_id: {'freq': doc_info['freq'], 'pos': [pos[-1] for pos in doc_info['pos']]}
                        for doc_id, doc_info in result_0.items()}
        if len(term_1) == 1:
            result_1 = single_word_search(term_1,inverted_index)
        else:
            result_1 = phrase_search(term_1,inverted_index) 
            result_1 = {doc_id: {'freq': doc_info['freq'], 'pos': [pos[0] for pos in doc_info['pos']]}
                        for doc_id, doc_info in result_1.items()}
         # Find common documents between the two search terms
        docs = set(result_0.keys()).intersection(set(result_1.keys()))
        results = []
        for doc_id in docs:
            pos_0 = result_0[doc_id]['pos']
            pos_1 = result_1[doc_id]['pos']
            
            for p0 in pos_0:
                for p1 in pos_1:
                    if p1-p0 <= distance:
                        results.append(doc_id)
                        break
                else:
                    continue
                break
    return results


# In[39]:


def evaluate_expression(expression,universal_set):
    a = str()
    try:
        if type(expression[0])==set:
            a = a+str(expression[0])
        for i in range(len(expression)):
            if expression[i] == 'AND':
                a = a+'&'+str(expression[i+1])
            if expression[i] == 'OR':
                a = a+'|'+str(expression[i+1])
            if expression[i] == 'AND NOT':
                a = a+'&'+'('+str(universal_set)+'-'+str(expression[i+1])+')'
            if expression[i] == 'OR NOT':
                a = a+'|'+'('+str(universal_set)+'-'+str(expression[i+1])+')'
            if expression[i] == 'NOT':
                a = a+'('+str(universal_set)+'-'+str(expression[i+1])+')'
        result = eval(a)
    except:
        result = {}
    return result


# In[40]:



def allsearch(query,universal_set,info_inverted_index,review_inverted_index,info_length,review_length,stop_words):
    boolean_word = ['AND NOT', 'OR NOT','AND','OR','NOT']
    query = query.replace('NOT NOT','')
    query = query.replace('(','')
    query = query.replace(')','')
    split_query = re.split(r'\b(AND NOT|OR NOT|AND|OR|NOT)\b', query)
    split_query = [elem for elem in split_query if elem.strip()]
    if any(word in boolean_word for word in split_query): 
        query_phrase=dict()
        for i in range(len(split_query)):
            phrase = split_query[i].strip()
            if phrase not in boolean_word:
                if bool(re.match(r'#(\d+)\{(.*?),(.*?)\}',phrase))==True:
                    prox = re.findall(r'#(\d+)\{(.*?),(.*?)\}',phrase)
                    prox_result_collection = []
                    for m in prox:
                        distance = int(m[0])
                        term_0 = preprocess_text(m[1],stop_words).split()
                        term_1 = preprocess_text(m[2],stop_words).split()
                        prox_result = proximity_search(term_0, term_1, distance, info_inverted_index)
                        prox_result_collection.append(prox_result)
                    query_phrase[i] = set.intersection(*map(set,prox_result_collection))
                else:
                    query_phrase[i] = set(phrase_tfidf(phrase,info_inverted_index,review_inverted_index,info_length,review_length,stop_words))
        for key in query_phrase.keys():
            split_query[key]=query_phrase[key]
        final_result = list(evaluate_expression(split_query,universal_set))
        if len(final_result)==0:
            final_result= phrase_tfidf(query,info_inverted_index,review_inverted_index,info_length,review_length,stop_words)
    else:
        if bool(re.match(r'#(\d+)\{(.*?),(.*?)\}',query))==True:
            prox = re.findall(r'#(\d+)\{(.*?),(.*?)\}',query)
            prox_result_collection = []
            for m in prox:
                distance = int(m[0])
                term_0 = preprocess_text(m[1],stop_words).split()
                term_1 = preprocess_text(m[2],stop_words).split()
                prox_result = proximity_search(term_0, term_1, distance, info_inverted_index)
                prox_result_collection.append(prox_result)
            final_result = list(set.intersection(*map(set,prox_result_collection)))
            if len(final_result)==0:
                final_result= phrase_tfidf(query,info_inverted_index,review_inverted_index,info_length,review_length,stop_words)
        else:
            final_result= phrase_tfidf(query,info_inverted_index,review_inverted_index,info_length,review_length,stop_words)
    return final_result


# In[41]:




# In[ ]:




