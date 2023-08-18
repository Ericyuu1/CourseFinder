from collections import defaultdict
from advancedfilter import build_query
import os
import ujson as json
import pymongo
from suggestion import checkspell
from nltk.corpus import stopwords
from flask import Flask, render_template, request
from backend import allsearch,read_inverted_index,generate_document_lengths
import time
import requests
import nltk
from nltk.corpus import stopwords
from temp import temp

app=Flask(__name__)

course_list=[]
for file in os.listdir('./reviews/'):
    course_list.append(file)
    
#get the index mapping
index_map = {}
for idx, val in enumerate(course_list):
    index_map[val[:-4]] = idx
    
info_inverted_index = read_inverted_index('info_inverted_index.txt')
review_inverted_index = read_inverted_index('review_inverted_index.txt')

info_length = generate_document_lengths(info_inverted_index)
review_length = generate_document_lengths(review_inverted_index)
universal_set = set(index_map.values())

stop_words = set(stopwords.words('english'))



@app.route('/',methods=['GET'])
def index():
    return render_template('search_page.html')


@app.route("/search", methods=["POST"])
def search():
    query = request.form["s_input"]
    start=time.time()
    realspell=checkspell(query)
    results = allsearch(query,universal_set,info_inverted_index,review_inverted_index,info_length,review_length,stop_words)
    end=time.time()
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db= client["test"]
    collection= db["test"]
    # Fetch data from the database using the indices in the results list
    query_filter = {"index": {"$in": results}}
    cursor = collection.find(query_filter)

    # Create a dictionary to store courses grouped by their names
    courses_dict = defaultdict(list)

    for doc in cursor:
        course_name = doc['course_name_x']
        course_link = doc['course_link_x']
        instructor_name = doc['instructor_name']
        instructor_external_link = doc['instructor_external_link_text']
        course_intro = doc['course_intro']

        # If a course with the same name already exists, append the instructor name
        if courses_dict[course_name]:
            courses_dict[course_name][2].append(instructor_name)
        else:
            courses_dict[course_name] = [course_link, course_intro, [instructor_name], instructor_external_link]

    # Extract the information for each course 
    course_names = []
    course_links = []
    course_intros = []
    instructor_names = []
    instructor_external_links = []

    for course_name, course_details in courses_dict.items():
        course_names.append(course_name)
        course_links.append(course_details[0])
        course_intros.append(course_details[1])
        instructor_names.append(", ".join(course_details[2]))
        instructor_external_links.append(course_details[3])

    amount=len(course_names)
    runtime='%.3f'%(end-start)
    return render_template("search_result_page.html", results=course_names,results2=course_links, results3=instructor_names, results4=instructor_external_links, results5=course_intros,real_spell=realspell,time=runtime,amount=amount)

@app.route('/aboutpage')
def information():
    return render_template('about_page.html')

@app.route('/helppage')
def helppage():
    return render_template('Help_page.html')


def server(environ, start_response):
    return app(environ,start_response)

@app.route("/search2", methods=["POST"])
def search2():
    boolean1 = request.form["boolean_1"]
    boolean2 = request.form["boolean_2"]
    boolean3 = request.form["boolean_3"]
    field1=request.form["field_0"]
    field2=request.form["field_1"]
    field3=request.form["field_2"]
    field4=request.form["field_3"]
    input1 = request.form["s_input0"]
    input2 = request.form["s_input1"]
    input3 = request.form["s_input2"]
    input4 = request.form["s_input3"]
    start=time.time()
    user_input = [
        (str(input1), str(field1), str(boolean1)),
        (str(input2), str(field2), str(boolean2)),
        (str(input3), str(field3), str(boolean3)),
        (str(input4), str(field4)),
    ]
    input_values = [x[0] for x in user_input]
    fields = [x[1] for x in user_input]
    booleans = [x[2] if len(x) == 3 else None for x in user_input]
    query = build_query(input_values, fields, booleans)
    cluster= pymongo.MongoClient("mongodb://localhost:27017/")
    db = cluster["test"]
    collection = db["test"]
    results = list(collection.find(query, {"_id": 0, "course_name_x": 1, "course_link_x": 1, "instructor_name": 1, "course_intro": 1, "instructor_external_link_text": 1}))

    # Create a dictionary to store courses grouped by their names
    courses_dict = defaultdict(list)

    for result in results:
        course_name = result['course_name_x']
        course_link = result['course_link_x']
        instructor_name = result['instructor_name']
        instructor_external_link = result['instructor_external_link_text']
        course_intro = result['course_intro']

    # If a course with the same name already exists, append the instructor name
        if courses_dict[course_name]:
            courses_dict[course_name][2].append(instructor_name)
        else:
            courses_dict[course_name] = [course_link, course_intro, [instructor_name], instructor_external_link]

# Extract the information for each course
    course_names = []
    course_links = []
    course_intros = []
    instructor_names = []
    instructor_external_links = []

    for course_name, course_details in courses_dict.items():
        course_names.append(course_name)
        course_links.append(course_details[0])
        course_intros.append(course_details[1])
        instructor_names.append(", ".join(course_details[2]))
        instructor_external_links.append(course_details[3])

    end=time.time()
    amount=len(course_names)
    runtime='%.3f'%(end-start)
    return render_template("search_result_page.html",results=course_names, results2=course_links, results3=instructor_names, results4=instructor_external_links, results5=course_intros,time=runtime,real_spell="no match found",amount=amount)

@app.route('/advancedsearch')
def advancedsearch():
    return render_template('advanced_search.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0',port='5000')
