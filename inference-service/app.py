from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from typing import Dict, List, Union
import pickle
import logging
from preprocess import preprocess_input


app = FastAPI(
    title="Custom Inference API",
    description="This API provides income prediction based on various personal and professional features. \
                **Important Note on Missing Values:** \
                - The API can handle missing values in the input data. \
                - Missing values can be represented 'null' in JSON. \
                - The API will impute missing values using mode (most frequent value) for each feature. \
                - However, if more than 20 percent of the features are missing, the API will return an error. \
                - The response will include a list of imputed features.",
    version="1.0.0",
)


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)


class PredictionInput(BaseModel):

    age: int = Field(None, description="Age of the individual")
    hours_per_week: int = Field(None, description="Number of work hours per week")
    workclass: str = Field(None, description="Type of employer (e.g., Private, Self-emp-not-inc, Federal-gov)")
    marital_status: str = Field(None, description="Marital status (e.g., Never-married, Married-civ-spouse)")
    occupation: str = Field(None, description="Occupation category")
    relationship: str = Field(None, description="Family role (e.g., Not-in-family, Husband)")
    race: str = Field(None, description="Race of the individual")
    sex: str = Field(None, description="Gender of the individual")
    native_country: str = Field(None, description="Country of origin")
    education_num: int = Field(None, description="Education level (in years)")
    capital_gain: float = Field(None, description="Capital gains")
    capital_loss: float = Field(None, description="Capital losses")
    fnlwgt: float = Field(None, description="Final weight")


class PredictionOutput(BaseModel):

    prediction: int
    probability: float
    imputed_features: List[str]


class ErrorOutput(BaseModel):

    error: str
    details: Union[Dict[str, str], List[str]]


@app.post("/predict", response_model=PredictionOutput, responses={400: {"model": ErrorOutput}})
async def predict(input_data: PredictionInput):
    """
    Predict income based on personal and professional features.

    This endpoint takes various personal and professional details as input and returns a prediction
    of whether the individual's income is above or below $50K per year, along with the probability
    of this prediction.

    API can handle missing values(""), but will return an error if more than 20% of features are missing.

    Args:
        input_data (PredictionInput): Personal and professional details of the individual.

    Returns:
        - prediction: Predicted income class (0: <=50K, 1: >50K)
        - probability: Probability of the prediction
        - imputed_features: List of features that were imputed due to missing values

    Raises:
        HTTPException: 
            - 400 Bad Request if input contains unknown categories or too many missing values.
            - 500 Internal Server Error for unexpected errors during prediction.

    Example of usage with missing values:
        ```
        curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{
            "age": 25,
            "hours_per_week": null,
            "workclass": "Private",
            "marital_status": "Married-civ-spouse",
            "occupation": "Machine-op-inspct",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "native_country": "United-States",
            "education_num": 10,
            "capital_gain": null,
            "capital_loss": 0,
            "fnlwgt": 200000
        }'
        ```
    """

    try:

        input_dict = input_data.dict()
        
        # Rename keys to match the expected input in preprocess_input
        input_dict['hours-per-week'] = input_dict.pop('hours_per_week')
        input_dict['marital-status'] = input_dict.pop('marital_status')
        input_dict['native-country'] = input_dict.pop('native_country')
        input_dict['education-num'] = input_dict.pop('education_num')
        input_dict['capital-gain'] = input_dict.pop('capital_gain')
        input_dict['capital-loss'] = input_dict.pop('capital_loss')
        logger.info(f"Input data: {input_dict}")

        # Process data
        processed_data, imputed_features = preprocess_input(input_dict)
        logger.info(f"Processed data: {processed_data}")
        logger.info(f"Imputed features: {imputed_features}")

        # Make prediction
        prediction = model.predict([processed_data])[0]
        probability = model.predict_proba([processed_data])[0].max()
        
        return PredictionOutput(prediction=int(prediction), probability=float(probability), imputed_features=imputed_features)
    
    except ValueError as e:

        logger.error(f"Error during prediction: {str(e)}")
        error_msg = str(e)

        if "Unknown categories detected:" in error_msg:
            details = eval(error_msg.split("Unknown categories detected: ")[1])
        elif "Too many missing values:" in error_msg:
            details = eval(error_msg.split("Too many missing values: ")[1])
        else:
            details = error_msg
        raise HTTPException(status_code=400, detail=ErrorOutput(error="Invalid input data", details=details).dict())
    
    except Exception as e:

        logger.error(f"Unexpected error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

