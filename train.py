import sys
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

def main():
    train_path = sys.argv[1] if len(sys.argv) > 1 else "TrainingDataset.csv"
    val_path = sys.argv[2] if len(sys.argv) > 2 else "ValidationDataset.csv"
    model_output_path = sys.argv[3] if len(sys.argv) > 3 else "energy_model"

    spark = SparkSession.builder \
        .appName("CS643_Project2_Parallel_Training") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    train_df = spark.read.csv(train_path, header=True, inferSchema=True).repartition(4).cache()
    val_df = spark.read.csv(val_path, header=True, inferSchema=True).repartition(4).cache()

    bldg_indexer = StringIndexer(inputCol="Building Type", outputCol="bldg_idx", handleInvalid="keep")
    day_indexer = StringIndexer(inputCol="Day of Week", outputCol="day_idx", handleInvalid="keep")

    encoder = OneHotEncoder(
        inputCols=["bldg_idx", "day_idx"],
        outputCols=["bldg_vec", "day_vec"],
        dropLast=False
    )

    assembler = VectorAssembler(
        inputCols=[
            "bldg_vec", "day_vec",
            "Square Footage", "Number of Occupants",
            "Appliances Used", "Average Temperature"
        ],
        outputCol="raw_features"
    )

    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)

    evaluator = RegressionEvaluator(
        labelCol="Energy Consumption",
        predictionCol="prediction",
        metricName="rmse"
    )

    candidates = [
        ("Linear Regression (OLS)", LinearRegression(featuresCol="features", labelCol="Energy Consumption", regParam=0.0)),
        ("Linear Regression (Ridge reg=0.01)", LinearRegression(featuresCol="features", labelCol="Energy Consumption", regParam=0.01)),
        ("Random Forest Regressor", RandomForestRegressor(featuresCol="features", labelCol="Energy Consumption", seed=42)),
        ("Gradient-Boosted Trees", GBTRegressor(featuresCol="features", labelCol="Energy Consumption", seed=42))
    ]

    best_rmse = float("inf")
    best_model = None
    best_name = ""

    print("=" * 60)
    print("Evaluating MLlib Models on Validation Dataset:")
    print("=" * 60)

    for name, reg in candidates:
        pipeline = Pipeline(stages=[bldg_indexer, day_indexer, encoder, assembler, scaler, reg])
        model = pipeline.fit(train_df)
        preds = model.transform(val_df)
        rmse = evaluator.evaluate(preds)
        print(f"Model: {name:35s} | Validation RMSE: {rmse:.6f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_name = name

    print("=" * 60)
    print(f"Winner: {best_name} with Validation RMSE = {best_rmse:.6f}")
    print(f"Saving best trained PipelineModel to: {model_output_path}")
    print("=" * 60)

    best_model.write().overwrite().save(model_output_path)
    spark.stop()

if __name__ == "__main__":
    main()
