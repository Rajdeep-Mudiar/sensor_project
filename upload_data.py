from pymongo.mongo_client import MongoClient
import pandas as pd
import json

url="mongodb+srv://raj2031g:12345@cluster0.e01gyci.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client=MongoClient(url)

# Create a database name and collection name
DATABASE_NAME="sensor_detection"
COLLECTION_NAME="wafer_fault"

df=pd.read_csv(r"D:\Coding\PW_Skills\Data_Science\Projects\sensor_detection\v2_complete_project_structure\notebooks\wafer_23012020_041211.csv")

df=df.drop("Unnamed: 0",axis=1)

# Convert to json so that we can pass to mongodb
json_record=list(json.loads(df.T.to_json()).values())

# Push all the data to mongodb
client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)