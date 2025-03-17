# flight-price-prediction

# Flight Price Prediction

## Overview

This project builds a **Flight Price Prediction** model using **Machine Learning**. The dataset is sourced from Kaggle and contains information about airlines, flight details, departure/arrival times, stops, class, and ticket prices. The model is trained using a **Random Forest Regressor** to predict flight prices based on various factors.

## Dataset

- The dataset used is **Clean_Dataset.csv**, obtained from Kaggle:  
  [Flight Price Prediction Dataset](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)
- Features include:
  - `airline`: Name of the airline
  - `source_city`: Departure city
  - `departure_time`: Time of departure
  - `stops`: Number of stops
  - `arrival_time`: Time of arrival
  - `destination_city`: Arrival city
  - `class`: Flight class (Economy/Business)
  - `duration`: Flight duration
  - `days_left`: Days left for departure
  - `price`: Flight ticket price (Target variable)

## Installation

1. Clone the repository:
   ```sh
   git clone <repo_link>
   cd flight-price-prediction
   ```
2. Create a virtual environment and install dependencies:
   ```sh
   python -m venv myenv
   source myenv/bin/activate  # On Windows use: myenv\Scripts\activate
   pip install -r requirements.txt
   ```

## Training the Model

Run the following command to train and save the model:

```sh
python src/train.py
```

This will:

- Load the dataset
- Preprocess the data
- Train a **RandomForestRegressor**
- Save the trained model (`flight_price_predictor.pkl`)

## Making Predictions

To predict flight prices on new data:

```sh
python src/predict.py
```

This script:

- Loads `new_flight_data.csv`
- Preprocesses it like training data
- Loads the trained model
- Outputs predicted flight prices

## Notes

- **DO NOT upload `.pkl` files to GitHub** (they are ignored in `.gitignore`).
- You can retrain the model anytime using `train.py`.
- Adjust hyperparameters in `train.py` to improve performance.
