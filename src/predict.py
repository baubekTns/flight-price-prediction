import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import math

# Load the trained model
reg = joblib.load('flight_price_predictor.pkl')

# Load new flight data
new_data = pd.read_csv('data/new_flight_data.csv')  # Replace with actual file path

# Preprocess new data
def preprocess_data(df, reference_columns):
    df = df.drop(['Unnamed: 0', 'flight'], axis=1, errors='ignore')
    df['class'] = df['class'].apply(lambda x: 1 if x == 'Business' else 0)
    df['stops'] = pd.factorize(df['stops'])[0]

    categorical_cols = ['airline', 'source_city', 'destination_city', 'arrival_time', 'departure_time']
    for col in categorical_cols:
        df = df.join(pd.get_dummies(df[col], prefix=col)).drop(col, axis=1)
    
    # Ensure new data has same columns as training data
    df = df.reindex(columns=reference_columns, fill_value=0)
    return df

# Load reference columns from training data
X_train = joblib.load('X_train_columns.pkl')  # Saved during training
new_data = preprocess_data(new_data, X_train)

# Predict flight prices
predicted_prices = reg.predict(new_data)
print("Predicted Prices:", predicted_prices)

# Evaluate the model if actual prices are available
if 'price' in new_data.columns:
    true_prices = new_data['price']
    print('R2:', r2_score(true_prices, predicted_prices))
    print('MSE:', mean_squared_error(true_prices, predicted_prices))
    print('MAE:', mean_absolute_error(true_prices, predicted_prices))
    print('RMSE:', math.sqrt(mean_squared_error(true_prices, predicted_prices)))
    
# Save predictions
new_data['predicted_price'] = predicted_prices
new_data.to_csv('data/predicted_flight_prices.csv', index=False)
