# CS643 Project 2 – Building Energy Consumption Prediction with Spark MLlib

This project trains a machine learning model with **Apache Spark MLlib (PySpark)** to predict a building's **energy consumption**. It also packages the trained model in a **Docker** container so predictions can run anywhere.

## Dataset

| File | Rows | Purpose |
|------|------|---------|
| `TrainingDataset.csv` | 800 | Model training |
| `ValidationDataset.csv` | 200 | Model selection / evaluation |

**Features**
- `Building Type` (categorical: Residential, Commercial, Industrial)
- `Square Footage`
- `Number of Occupants`
- `Appliances Used`
- `Average Temperature`
- `Day of Week` (categorical: Weekday, Weekend)

**Label:** `Energy Consumption`

## How It Works

### Training (`train.py`)
1. Loads the training and validation CSVs into Spark DataFrames, repartitioned so the work runs in parallel.
2. Builds an ML pipeline:
   - `StringIndexer` + `OneHotEncoder` for the categorical columns
   - `VectorAssembler` to combine all features
   - `StandardScaler` to normalize the features
3. Trains and compares four regressors:
   - Linear Regression (OLS)
   - Linear Regression (Ridge, `regParam=0.01`)
   - Random Forest Regressor
   - Gradient-Boosted Trees Regressor
4. Picks the model with the lowest **validation RMSE** and saves the full `PipelineModel` to `energy_model/`.

### Prediction (`predict.py`)
Loads the saved pipeline model, runs it on a test CSV, and prints the **RMSE**.

## Results

On a 4-node AWS EC2 Spark cluster (1 master + 3 workers, `t2.medium`, Ubuntu 24.04), **Linear Regression** gave the best result, with a validation **RMSE of 0.013751**.

The prediction container is published on Docker Hub: [`vyompatel2709/cs643-energy-predictor`](https://hub.docker.com/r/vyompatel2709/cs643-energy-predictor).

For the full write-up, including cluster setup, Docker deployment, and the AI usage disclosure, see the [Final Report (PDF)](CS643_Project2_Final_Report.pdf).

## Project Structure

```
.
├── train.py               # Parallel model training & selection
├── predict.py             # Load model and evaluate on a test file
├── Dockerfile             # Container image for prediction
├── TrainingDataset.csv
├── ValidationDataset.csv
└── CS643_Project2_Final_Report.pdf
```

## Requirements

- Python 3
- Java 17 (needed by Spark)
- PySpark 3.5.1 and NumPy

```bash
pip install pyspark==3.5.1 numpy
```

## Usage

### 1. Train the model

Locally:
```bash
python3 train.py TrainingDataset.csv ValidationDataset.csv energy_model
```

On a Spark cluster (for example, several EC2 instances):
```bash
spark-submit --master spark://<master-ip>:7077 train.py \
  TrainingDataset.csv ValidationDataset.csv energy_model
```

All three arguments are optional. Their defaults are the file names shown above.

### 2. Run prediction (without Docker)

```bash
python3 predict.py ValidationDataset.csv energy_model
```

### 3. Run prediction with Docker

#### Option A: Pull the pre-built image from Docker Hub

The published image already contains the trained model, so you don't need to train first.

```bash
docker pull vyompatel2709/cs643-energy-predictor:latest
docker run --rm -v "$(pwd)":/data vyompatel2709/cs643-energy-predictor:latest /data/ValidationDataset.csv
```

On Apple Silicon (arm64) Macs, add `--platform linux/amd64`:
```bash
docker run --platform linux/amd64 --rm -v "$(pwd)":/data \
  vyompatel2709/cs643-energy-predictor:latest /data/ValidationDataset.csv
```

To evaluate a different file, such as `TestDataset.csv`, put it in the current directory and pass `/data/TestDataset.csv`.

#### Option B: Build the image yourself

You must train the model first, because the image copies the `energy_model/` folder into itself. That folder isn't stored in this repo.

```bash
docker build -t cs643-project2 .
docker run --rm -v "$(pwd)":/data cs643-project2 /data/ValidationDataset.csv
```

Example output:
```
==================================================
Test Dataset: /data/ValidationDataset.csv
Root Mean Squared Error (RMSE): 0.013751
==================================================
```

## Author

Vyom Patel – CS643 Cloud Computing
