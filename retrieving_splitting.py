from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType
import pymongo

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["audio_db"]
collection = db["audio_features"]

# Query data from MongoDB
cursor = collection.find()
data = list(cursor)

# Convert data to Spark DataFrame
spark = SparkSession.builder \
    .appName("MongoDB Spark Example") \
    .getOrCreate()

# Define schema excluding the _id field
schema = StructType([
    StructField("track_id", StringType(), True),
    StructField("title", StringType(), True),
    StructField("genres_all", StringType(), True),
    StructField("mfcc_features", StringType(), True)
])

# Create Spark DataFrame
spark_df = spark.createDataFrame(data, schema=schema)

# Split the dataset into training and testing sets
train_df, test_df = spark_df.randomSplit([0.8, 0.2], seed=42)

# Show first few rows of training and testing sets
print("Training set:")
train_df.show(truncate=False)

print("Testing set:")
test_df.show(truncate=False)

# Stop SparkSession
spark.stop()