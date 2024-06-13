# Project Big Data
 Music Reccomendation system

Extractfeatures.py;
Python script designed to process audio files, extracting MFCC (Mel-Frequency Cepstral Coefficients) features and saving them along with track information into a CSV file. The script requires several Python libraries to be installed: librosa for audio and music analysis, numpy for array operations, and tqdm for displaying progress bars. The script comprises various functions, including one to extract MFCC features from an audio file, another to read necessary columns from a 'tracks.csv' file, and a third to process audio files in a directory and save the extracted features along with track information into a CSV file. The main function checks if the script is being run directly and specifies input and output file paths, reads the 'tracks.csv' file to obtain track information, and then processes audio files using the aforementioned function. To use the script, users must place their audio files in a directory specified by fma_data_directory and ensure the presence of the 'tracks.csv' file containing necessary track information. Execution of the script is straightforward using the command python script_name.py, which will process audio files in the designated directory, extract MFCC features, and save them along with track information to 'mfcc_features_with_info.csv'.

Mongo.py:
This Python script reads data from a CSV file containing track information and MFCC features and inserts it into a MongoDB database. It utilizes the csv module to read the CSV file and the pymongo library to interact with MongoDB.

The script consists of two main functions:

read_csv(csv_file): This function reads the CSV file specified by csv_file and returns the data as a list of dictionaries.

insert_into_mongodb(data, db_uri, db_name, collection_name): This function inserts the data into MongoDB. It connects to the MongoDB server using the URI specified by db_uri, selects the database specified by db_name, accesses the collection specified by collection_name, inserts the data using insert_many(), and then closes the connection to the MongoDB server.

In the main section (if __name__ == "__main__":), the script specifies the CSV file containing the data, sets up MongoDB configuration parameters (db_uri, db_name, collection_name), reads the data from the CSV file using read_csv(), and inserts the data into MongoDB using insert_into_mongodb().

To use the script, ensure that MongoDB is running on your local machine, and replace the csv_file variable with the path to your CSV file. Adjust MongoDB configuration parameters (db_uri, db_name, collection_name) as needed. Then execute the script, and the data will be inserted into MongoDB.

Retrieving_Sampling.py:
This script establishes a connection to a MongoDB database, retrieves data from a specified collection, and converts it into a Spark DataFrame. It defines a schema for the DataFrame, splits the data into training and testing sets using an 80-20 ratio, and displays the first few rows of each set. Finally, it terminates the SparkSession. This integration enables seamless data processing and analysis by leveraging the strengths of MongoDB for flexible data storage and retrieval, and Apache Spark for distributed data processing.

Training.py:
This script connects to a MongoDB database to retrieve audio features data, converts it into a Spark DataFrame, and splits it into training and testing sets. Using the ALS (Alternating Least Squares) algorithm from Spark's machine learning library, it trains a recommendation model to find similar songs based on their audio features. After evaluating the model's performance with root mean squared error (RMSE), the script conducts hyperparameter tuning through cross-validation to optimize the model's parameters. The final tuned model is then evaluated again, and the RMSE after tuning is printed. Finally, the SparkSession is stopped, concluding the script's execution. This workflow demonstrates the integration of MongoDB with Spark for building and optimizing a music recommendation system.
