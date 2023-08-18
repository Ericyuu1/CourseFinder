from pymongo import MongoClient
import pandas as pd

cluster= MongoClient("mongodb://localhost:27017/")
db = cluster["test"]
collection = db["test"]


# Read CSV file into a pandas dataframe
df = pd.read_csv('merged_file.csv')

# Convert dataframe to a list of dictionaries
data = df.to_dict('records')

# Insert data into MongoDB
collection.insert_many(data)
