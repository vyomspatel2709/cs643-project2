import sys
import os
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 predict.py <testFilePath> [modelPath]")
        sys.exit(1)

    test_file_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("MODEL_PATH", "energy_model")

    spark = SparkSession.builder \
        .appName("CS643_Project2_Prediction") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    test_df = spark.read.csv(test_file_path, header=True, inferSchema=True)
    model = PipelineModel.load(model_path)

    predictions = model.transform(test_df)

    evaluator = RegressionEvaluator(
        labelCol="Energy Consumption",
        predictionCol="prediction",
        metricName="rmse"
    )
    rmse = evaluator.evaluate(predictions)

    print("=" * 50)
    print(f"Test Dataset: {test_file_path}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print("=" * 50)

    spark.stop()

if __name__ == "__main__":
    main()
