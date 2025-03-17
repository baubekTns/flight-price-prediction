import pandas as pd
import joblib
import math
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('data/Clean_Dataset.csv')

# Preprocessing
df = df.drop(['Unnamed: 0', 'flight'], axis=1, errors='ignore')
df['class'] = df['class'].apply(lambda x: 1 if x == 'Business' else 0)
df['stops'] = pd.factorize(df['stops'])[0]

categorical_cols = ['airline', 'source_city', 'destination_city', 'arrival_time', 'departure_time']
for col in categorical_cols:
    df = df.join(pd.get_dummies(df[col], prefix=col)).drop(col, axis=1)

# Split dataset
X, y = df.drop('price', axis=1), df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train initial model
reg = RandomForestRegressor(n_jobs=-1, random_state=42)
reg.fit(X_train, y_train)

# Hyperparameter tuning
param_dist = {
    'n_estimators': randint(100, 300),
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5),
    'max_features': [None, 'sqrt', 'log2']
}

random_search = RandomizedSearchCV(
    estimator=reg,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    scoring='neg_mean_squared_error',
    verbose=2,
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)

best_regressor = random_search.best_estimator_

# Save the trained model and training data columns
joblib.dump(best_regressor, 'models/flight_price_predictor.pkl')
joblib.dump(X_train.columns, 'models/X_train_columns.pkl')

# Evaluate model
y_pred = best_regressor.predict(X_test)
print(f'Best Model R2: {r2_score(y_test, y_pred)}')
print(f'Best Model MSE: {mean_squared_error(y_test, y_pred)}')
print(f'Best Model MAE: {mean_absolute_error(y_test, y_pred)}')
print(f'Best Model RMSE: {math.sqrt(mean_squared_error(y_test, y_pred))}')

# Feature importance
importances = dict(zip(X_train.columns, best_regressor.feature_importances_))
sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)

plt.figure(figsize=(15, 6))
plt.bar([x[0] for x in sorted_importances[:10]], [x[1] for x in sorted_importances[:10]])
plt.xticks(rotation=45)
plt.title('Top 10 Feature Importances')
plt.show()
