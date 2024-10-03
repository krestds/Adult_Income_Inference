import json
import numpy as np


# Load the label encoding dictionary, standardization dictionary and mode values for missing observations
with open('model_docs/label_mappings.json', 'r') as f:
    label_encoding = json.load(f)

with open('model_docs/standart_scaler.json', 'r') as f:
    standardization = json.load(f)

with open('model_docs/mode_values.json', 'r') as f:
    mode_values = json.load(f)


# functions for categorization Age and working hours
def work_time_category(hours):

    if hours <= 20:
        return 'Part-time'
    elif 20 < hours <= 40:
        return 'Full-time'
    elif 40 < hours <= 60:
        return 'Over-time'
    else:
        return 'Hard-time'

def age_group(age):

    if age < 20:
        return '0-20'
    elif 20 <= age < 30:
        return '20-30'
    elif 30 <= age < 40:
        return '30-40'
    elif 40 <= age < 50:
        return '40-50'
    elif 50 <= age < 60:
        return '50-60'
    elif 60 <= age < 70:
        return '60-70'
    else:
        return '70+'


# Label encoding for model: from features to labels
def encode_categorical(value, feature):

    if value is None or value == "":
        return label_encoding[feature][mode_values[feature]]
    if value not in label_encoding[feature]:
        raise ValueError(f"Unknown category '{value}' for feature '{feature}'")
    
    return label_encoding[feature][value]

# standartisation some numerical features
def standardize(value, feature):

    if value is None or value == "":
        return (mode_values[feature] - standardization[f"{feature}_mean"]) / np.sqrt(standardization[f"{feature}_var"])
    value = float(value)
    mean = standardization[f"{feature}_mean"]
    var = standardization[f"{feature}_var"]

    return (value - mean) / np.sqrt(var)

# Main preprocess function
def preprocess_input(data):
    """Main preprocess function. Provide inputing missing values and transformation for each feature.
       Prepare features for model inference"""

    processed = []
    imputed_features = []
    total_fields = len(data)

    for feature, value in data.items():
        if value is None or value == "":
            imputed_features.append(feature)

    if len(imputed_features) / total_fields > 0.2:
        raise ValueError(f"Too many missing values: {imputed_features}")

    # Age group
    if data['age'] is None or data['age'] == "":
        age_cat = mode_values['age']
        age_cat = age_group(age_cat)
    else:
        age_cat = age_group(data['age'])
    processed.append(encode_categorical(age_cat, 'age_group'))

    # Work time category
    if data['hours-per-week'] is None or data['hours-per-week'] == "":
        work_cat = mode_values["hours-per-week"]
        work_cat = work_time_category(work_cat)
    else:
        work_cat = work_time_category(data['hours-per-week'])
    processed.append(encode_categorical(work_cat, 'work_time_category'))

    # Other categorical features
    for feature in ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']:
        processed.append(encode_categorical(data[feature], feature))

    # Education-num (it's already numerical)
    if data['education-num'] is None or data['education-num'] == "":
        processed.append(float(mode_values['education-num']))
    else:
        processed.append(float(data['education-num']))

    # Standardized features
    for feature in ['capital-gain', 'capital-loss', 'fnlwgt']:
        processed.append(standardize(data[feature], feature))

    return processed, imputed_features


# Example usage
if __name__ == "__main__":
    sample_input = {
        'age': 25,
        'hours-per-week': 40,
        'workclass': 'Private',
        'marital-status': 'Never-married',
        'occupation': 'Sales',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'native-country': 'United-States',
        'education-num': 10,
        'capital-gain': 2000,
        'capital-loss': 0,
        'fnlwgt': 200000
    }
    print(preprocess_input(sample_input))
