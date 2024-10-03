# Adult Income Prediction Project

This project aims to predict whether an individual's income exceeds $50,000 per year based on census data. It consists of two main parts:
1. Exploratory Data Analysis (EDA) and Model Training
2. Microservice for Model Inference

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Exploratory Data Analysis and Model Training](#exploratory-data-analysis-and-model-training)
3. [Income Prediction Microservice](#income-prediction-microservice)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Running the Microservice](#running-the-microservice)
   - [Locally](#locally)
   - [On a Virtual Machine](#on-a-virtual-machine)
7. [API Endpoints](#api-endpoints)
8. [Using the API](#using-the-api)
9. [Input Format](#input-format)
10. [Output Format](#output-format)
11. [Handling Missing Values](#handling-missing-values)
12. [Error Handling](#error-handling)
13. [API Documentation](#api-documentation)
14. [Example Usage](#example-usage)

## Repository Structure

```
/
├── EDA_and_Model_Training.ipynb
├── inference-service/
│   ├── model/
│   │   └── model.pkl
│   ├── model_docs/
│   │   ├── label_mappings.json
│   │   ├── mode_values.json
│   │   └── standart_scaler.json
│   ├── app.py
│   ├── dockerfile
│   ├── preprocess.py
│   └── requirements.txt
└── README.md
```

## Exploratory Data Analysis and Model Training

The file `EDA_and_Model_Training.ipynb` contains the following:

- Data loading and initial exploration
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA) with visualizations
- Feature engineering and selection
- Model selection, training, and evaluation
- Model serialization (saving the trained model)

To view the EDA and model training process:
1. Open the `EDA_and_Model_Training.ipynb` file in Jupyter Notebook or JupyterLab.
2. Run the cells sequentially to see the analysis, visualizations, and model training process.

## Income Prediction Microservice

The microservice provides income prediction based on various personal and professional features. It uses the machine learning model trained in the EDA process to predict whether an individual's income is above or below $50,000 USD per year.

## Prerequisites

- Python 3.7+
- Docker
- cURL (for testing the API)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/krestds/Adult_Income_Inference.git
   cd Adult_Income_Inference/inference-service
   ```

2. Build the Docker image:
   ```
   docker build -t income-prediction-service .
   ```

## Running the Microservice

### Locally

Run the Docker container:

```
docker run -p 8000:8000 income-prediction-service
```

The service will be available at `http://localhost:8000`.

### On a Virtual Machine

To run the service on a VM:

1. Install Docker on the VM.
2. Copy the project files to the VM:
   ```
   scp -r /path/to/Adult_Income_Inference user@vm_ip:/path/on/vm/
   ```
3. SSH into the VM:
   ```
   ssh user@vm_ip
   ```
4. Navigate to the project directory:
   ```
   cd /path/on/vm/Adult_Income_Inference/inference-service
   ```
5. Build the Docker image:
   ```
   docker build -t income-prediction-service .
   ```
6. Run the Docker container:
   ```
   docker run -p 8000:8000 income-prediction-service
   ```

The service will be available at `http://<vm_ip>:8000`.

## API Endpoints

The API has the following endpoints:

1. GET `/`: Returns a welcome message and basic API information.
2. POST `/predict`: Predicts income based on input features.

### Root Endpoint

When you access `http://localhost:8000` in a web browser or via a GET request, you'll see a JSON response like this:

```json
{
  "message": "Welcome to the Income Prediction API",
  "documentation": "/docs",
  "predict_endpoint": "/predict"
}
```

This endpoint provides a quick way to verify that the API is running and gives basic information about the available endpoints.

### Prediction Endpoint

The `/predict` endpoint is designed to handle POST requests only. It expects a JSON payload with the input features for the prediction.

## Using the API

To use the `/predict` endpoint:
- Use a tool cURL (Postman, or a programming language's HTTP client) to send a POST request.
- Set the `Content-Type` header to `application/json`.
- Include the required input features in the request body as JSON.

## Input Format

Send a POST request to `/predict` with a JSON body containing the following fields:

```json
{
  "age": int,
  "hours_per_week": int,
  "workclass": string,
  "marital_status": string,
  "occupation": string,
  "relationship": string,
  "race": string,
  "sex": string,
  "native_country": string,
  "education_num": int,
  "capital_gain": float,
  "capital_loss": float,
  "fnlwgt": float
}
```

## Output Format

The API returns a JSON response with the following structure:

```json
{
  "prediction": int,
  "probability": float,
  "imputed_features": [string]
}
```

- `prediction`: 0 (income <= $50K) or 1 (income > $50K)
- `probability`: The probability of the prediction
- `imputed_features`: List of features that were imputed due to missing values

## Handling Missing Values

- The API can handle missing values in the input data.
- Missing values should be represented as `null` in the JSON input.
- The API will impute missing values using the mode (most frequent value) for each feature.
- If more than 20% of the features are missing, the API will return an error.

## Error Handling

The API may return the following errors:

- 400 Bad Request: If the input contains unknown categories or too many missing values.
- 405 Method Not Allowed: If a GET request is sent to the `/predict` endpoint.
- 422 Unprocessable Entity: If the input data is invalid or incomplete.
- 500 Internal Server Error: For unexpected errors during prediction.

## Common Errors

1. "Method Not Allowed" error:
   - Cause: Sending a GET request to the `/predict` endpoint instead of a POST request.
   - Solution: Ensure you're sending a POST request with the appropriate JSON payload.

2. "Unprocessable Entity" error:
   - Cause: Sending invalid or incomplete data in the request payload.
   - Solution: Check that all required fields are included and have the correct data types.

## API Documentation

FastAPI automatically generates interactive API documentation. To view and interact with the API documentation:

1. Ensure the microservice is running.
2. Open a web browser and go to `http://localhost:8000/docs`.
3. You'll see the Swagger UI with details about all endpoints.
4. You can test the API directly from this interface:
   - Click on the `/predict` endpoint in the Swagger UI.
   - Click "Try it out".
   - Enter your input data in the JSON format provided.
   - Click "Execute" to send the request and see the response.

Alternatively, you can access the ReDoc version of the documentation at `http://localhost:8000/redoc`.

## Example Usage

Here's an example of how to use the API with cURL:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 44,
       "hours_per_week": 40,
       "workclass": "Private",
       "marital_status": "Married-civ-spouse",
       "occupation": "Machine-op-inspct",
       "relationship": "Husband",
       "race": "White",
       "sex": "Male",
       "native_country": "United-States",
       "education_num": 14,
       "capital_gain": 4880,
       "capital_loss": 0,
       "fnlwgt": 6591
     }'
```

Example response:

```json
{
  "prediction": 1,
  "probability": 0.73,
  "imputed_features": []
}
```

This response indicates that the model predicts an income above $50,000 USD with a probability of 73%, and no features were imputed.
