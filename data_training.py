from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType, ArrayType, IntegerType, FloatType
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, udf
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import pymongo

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["audio_db"]
collection = db["audio_features"]

# Query data from MongoDB
cursor = collection.find()
data = list(cursor)

# Convert data types
for record in data:
    record["track_id"] = int(record["track_id"])
    genres_str = record["genres_all"][1:-1].split(",")
    record["genres_all"] = [int(genre.strip()) if genre.strip() else 0 for genre in genres_str]

# Convert data to Spark DataFrame
spark = SparkSession.builder \
    .appName("Music Recommendation System") \
    .getOrCreate()

# Define schema excluding the _id field
schema = StructType([
    StructField("track_id", IntegerType(), True),
    StructField("title", StringType(), True),
    StructField("genres_all", ArrayType(IntegerType()), True),
    StructField("mfcc_features", StringType(), True)
])

# Create Spark DataFrame
spark_df = spark.createDataFrame(data, schema=schema)

# Define a user-defined function (UDF) to parse the mfcc_features column
parse_mfcc_features_udf = udf(lambda x: [float(i) for i in x[1:-1].split(",")], ArrayType(FloatType()))

# Apply the UDF to convert mfcc_features from string to array of floats
spark_df = spark_df.withColumn("mfcc_features", parse_mfcc_features_udf(spark_df["mfcc_features"]))

# Split the dataset into training and testing sets
train_df, test_df = spark_df.randomSplit([0.8, 0.2], seed=42)

# Train ALS model
als = ALS(maxIter=5, regParam=0.01, userCol="track_id", itemCol="genres_all", ratingCol="mfcc_features")
model = als.fit(train_df)

# Find similar songs using ALS model
similar_songs = model.recommendForAllItems(10)

# Evaluate the model
evaluator = RegressionEvaluator(metricName="rmse", labelCol="mfcc_features", predictionCol="prediction")
rmse = evaluator.evaluate(model.transform(test_df))
print("Root Mean Squared Error (RMSE):", rmse)

# Hyperparameter tuning
param_grid = ParamGridBuilder() \
    .addGrid(als.rank, [5, 10, 15]) \
    .addGrid(als.regParam, [0.01, 0.1, 1.0]) \
    .build()

evaluator = RegressionEvaluator(metricName="rmse", labelCol="mfcc_features", predictionCol="prediction")
cv = CrossValidator(estimator=als,
                    estimatorParamMaps=param_grid,
                    evaluator=evaluator,
                    numFolds=3)

cv_model = cv.fit(train_df)
best_model = cv_model.bestModel
rmse_best = evaluator.evaluate(best_model.transform(test_df))
print("Root Mean Squared Error (RMSE) after hyperparameter tuning:", rmse_best)

# Stop SparkSession
spark.stop()
