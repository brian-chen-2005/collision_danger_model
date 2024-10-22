import joblib
import numpy as np
import pandas as pd

best_rf_regressor = joblib.load('random_forest_model.joblib')

def predict_danger_value(latitude, longitude, weather, roadcond, light_cond):
    input_data = pd.DataFrame([[latitude, longitude, weather, roadcond, light_cond]], 
                              columns=['latitude', 'longitude', 'weather', 'roadcond', 'light_cond'])

    
    predicted_value = best_rf_regressor.predict(input_data)

    #+round to 2 decimal places for assigned danger value
    predicted_value_rounded = np.round(predicted_value, 2)
    
    return predicted_value_rounded[0]

latitude = 47.6097
longitude = -122.3331
weather = 1
roadcond = 4
light_cond = 2

predicted_danger = predict_danger_value(latitude, longitude, weather, roadcond, light_cond)
print(f'predicted danger value: {predicted_danger}')
