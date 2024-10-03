# Income Prediction Microservice

This microservice provides income prediction based on various personal and professional features. It uses a machine learning model to predict whether an individual's income is above or below $50,000 USD per year.

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Running the Microservice](#running-the-microservice)
5. [API Endpoints](#api-endpoints)
6. [Using the API](#using-the-api)
7. [Input Format](#input-format)
8. [Output Format](#output-format)
9. [Handling Missing Values](#handling-missing-values)
10. [Error Handling](#error-handling)
11. [Common Errors](#common-errors)
12. [API Documentation](#api-documentation)
13. [Example Usage](#example-usage)

## Repository Structure

```
/inference-service
├── model/
│   └── model.pkl
├── model_docs/
│   ├── label_mappings.json
│   ├── mode_values.json
│   └── standart_scaler.json
├── app.py
├── dockerfile
├── preprocess.py
└── requirements.txt
```

## Prerequisites

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

Run the Docker container:

```
docker run -p 8000:8000 income-prediction-service
```

The service will be available at `http://localhost:8000`.

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
