import csv
import pymongo

# Function to read the CSV file containing track information and MFCC features
def read_csv(csv_file):
    data = []
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data

# Function to insert data into MongoDB
def insert_into_mongodb(data, db_uri, db_name, collection_name):
    client = pymongo.MongoClient(db_uri)
    db = client[db_name]
    collection = db[collection_name]
    collection.insert_many(data)
    client.close()
    print("Data inserted into MongoDB.")

# Main function
if __name__ == "__main__":
    # Specify the CSV file containing track information and MFCC features
    csv_file = "mfcc_features_with_info.csv"
   
    # MongoDB configuration
    db_uri = "mongodb://localhost:27017/"
    db_name = "audio_db"
    collection_name = "audio_features"
   
    # Read the CSV file
    data = read_csv(csv_file)
   
    # Insert data into MongoDB
    insert_into_mongodb(data, db_uri, db_name, collection_name)