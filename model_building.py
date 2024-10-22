import pandas as pd
import numpy as np
import joblib

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, validation_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn import metrics
import itertools
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 251249)
pd.set_option('display.max_columns', 50)


df = pd.read_csv('/Users/brian/trafficProj/SDOT_Collisions_All_Years.csv')

#+for convenience
df.rename(columns={'SEVERITYCODE': 'severity_code', 'x':'longitude', 'y': 'latitude',
                   'ADDRTYPE':'addr_type', 'LOCATION': 'location','SEVERITYDESC':'severity_desc', 'COLLISIONTYPE':'collision_type',
                   'PERSONCOUNT':'person_count', 'PEDCOUNT': 'ped_count', 'PEDCYLCOUNT': 'ped_cycle_count', 'VEHCOUNT': 'veh_count',
                   'INCDTTM': 'inc_dt', 'JUNCTIONTYPE': 'junc_type', 'SDOT_COLCODE': 'case_code', 'SDOT_COLDESC': 'case_desc',
                   'UNDERINFL':'under_infl', 'WEATHER': 'weather', 'ROADCOND': 'roadcond', 'LIGHTCOND': 'light_cond',
                   'STCOLCODE': 'st_code', 'ST_COLDESC': 'st_desc', 'HITPARKEDCAR':'hit_parked_car', 'SPEEDING':'speeding', 
                   'FATALITIES':'fatalities', 'INJURIES':'injuries', 'SERIOUSINJURIES':'serious_injuries'}, inplace=True)
#


#+too many null
df = df[['location','severity_code',
        'severity_desc','collision_type', 'person_count', 'ped_count', 'ped_cycle_count',
       'veh_count','inc_dt','addr_type', 'junc_type', 'case_code', 'case_desc','under_infl',
       'speeding', 'weather', 'roadcond', 'light_cond','st_code', 'st_desc',
       'hit_parked_car', 'injuries', 'serious_injuries', 'fatalities', 'longitude', 'latitude']]
#

#============================
#       Data Cleaning
#============================
#+useful data columns
df1 = df[['latitude', 'longitude', 'severity_code', 'weather', 'roadcond', 'light_cond', 
          'speeding', 'under_infl', 'person_count', 'ped_count', 'ped_cycle_count', 'veh_count', 
          'injuries', 'serious_injuries', 'severity_desc', 'fatalities']]
#


#+only y values in speeding column so Y->1 and nan->0
df1['speeding'].replace(np.nan,0,inplace=True)
df1['speeding'].replace('Y', 1, inplace=True)
#



#+clears unknown, other, empty
df1.replace(to_replace={'Unknown': np.nan, 
                        'Other':np.nan}, inplace=True)
df1.dropna(inplace=True)
#

#+sort all to 1,0
df1['under_infl'].replace(to_replace={'Y':1, 'N':0, '1':1, '0':0}, inplace=True)
#

#+severity codes 1,2,0,2b,3 correspond respectively to
#       1:property damage, 2:injury, 0:unknown, 2b:serious injury, 3:fatality
#replace 2b lable with 4 so its 1,2,0,4,3
#
df1['severity_code'].replace(to_replace={'2b':'4'}, inplace=True)

#assign danger values to weather, roadcond, light_cond

df1['weather'].replace('Clear',0,inplace=True)
df1['weather'].replace('Partly Cloudy', 1, inplace=True)
df1['weather'].replace('Overcast',2,inplace=True)
df1['weather'].replace('Fog/Smog/Smoke', 3, inplace=True)
df1['weather'].replace('Blowing Sand/Dirt',4,inplace=True)
df1['weather'].replace('Blowing Snow', 5, inplace=True)
df1['weather'].replace('Raining',6,inplace=True)
df1['weather'].replace('Snowing', 7, inplace=True)
df1['weather'].replace('Severe Crosswind',8,inplace=True)
df1['weather'].replace('Sleet/Hail/Freezing Rain', 9, inplace=True)

df1['roadcond'].replace('Dry', 0, inplace=True)
df1['roadcond'].replace('Sand/Mud/Dirt', 1, inplace=True)
df1['roadcond'].replace('Standing Water', 2, inplace=True)
df1['roadcond'].replace('Snow/Slush', 3, inplace=True)
df1['roadcond'].replace('Wet', 4, inplace=True)
df1['roadcond'].replace('Ice', 5, inplace=True)
df1['roadcond'].replace('Oil', 6, inplace=True)

df1['light_cond'].replace('Daylight', 0, inplace=True)
df1['light_cond'].replace('Dawn', 1, inplace=True)
df1['light_cond'].replace('Dusk', 2, inplace=True)
df1['light_cond'].replace('Dark - Unknown Lighting', 3, inplace=True)
df1['light_cond'].replace('Dark - Street Lights On', 4, inplace=True)
df1['light_cond'].replace('Dark - No Street Lights', 5, inplace=True)
df1['light_cond'].replace('Dark - Street Lights Off', 6, inplace=True)

#print(df1[df1['severity_code'] == 0].shape[0])
#no cases of unknown in severity code
#============================

df1.info()


#============================
#   Random Forest Regression
#============================    

#+severity codes to danger values mapping (arbitrary personal opinion on danger)
danger_mapping = {'1': 1, '2': 6, '4': 8, '3': 10}
df1['danger_value'] = df1['severity_code'].map(danger_mapping)

#+one-hot encode categorical variables (weather, roadcond, light_cond)
#categorical_cols = ['weather', 'roadcond', 'light_cond']
#encoder = OneHotEncoder(sparse_output=False, drop='first')
#encoded_features = pd.DataFrame(encoder.fit_transform(df1[categorical_cols]))
#encoded_features.columns = encoder.get_feature_names_out(categorical_cols)

#+combine features
#df_model = pd.concat([df1[['latitude', 'longitude']], encoded_features], axis=1)

#+split data into features (X) and target (y)
df_clean = df1.dropna(subset=['danger_value', 'latitude', 'longitude', 'weather', 'roadcond', 'light_cond'])

#+recreate features and target
X = df_clean[['latitude', 'longitude', 'weather', 'roadcond', 'light_cond']]
y = df_clean['danger_value']

#+polynomial features
#poly = PolynomialFeatures(degree=2, include_bias=False)
#X_poly = poly.fit_transform(X)

#+split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#+train random forest regressor
#param_grid = {
#    'n_estimators': [100, 200, 500],     # num trees in forest
#    'max_depth': [None, 10, 20, 30],     # max depth of tree
#    'min_samples_split': [2, 5, 10],     # min samples required to split a node
#    'min_samples_leaf': [1, 2, 4],       # min samples required to be at a leaf node
#    'bootstrap': [True, False]           # are bootstrap samples used when building trees
#}

#rf_regressor = RandomForestRegressor(random_state=42)

#+gridsearchcv with cross-validation
#grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, 
#                           cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

#+fit grid search to the training data
#grid_search.fit(X_train, y_train)

#+best hyperparameters
#print(f"best hyperparameters: {grid_search.best_params_}")

#hyperparameter result
best_rf_regressor = RandomForestRegressor(
    bootstrap=True,
    max_depth=20,
    min_samples_leaf=4,
    min_samples_split=10,
    n_estimators=500,
    random_state=42
)

#fit model to training data
best_rf_regressor.fit(X_train, y_train)

#+debug predictions
#print(y_pred_rounded[:10])

#+save model
joblib.dump(best_rf_regressor, '/Users/brian/trafficProj/random_forest_model.joblib')

#============================    

#============================
#   Testing Performance
#============================

#+predict danger values on test set
#y_pred = rf_regressor.predict(X_test)

#+calculate metrics
y_pred_best = best_rf_regressor.predict(X_test)

mae_best = mean_absolute_error(y_test, y_pred_best)
mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

print(f"Model Performance on Test Set:")
print(f"Mean Absolute Error (MAE): {mae_best:.2f}")
print(f"Mean Squared Error (MSE): {mse_best:.2f}")
print(f"R-squared (R²): {r2_best:.2f}")

#============================

#============================
#Residual plotting and error analysis
#============================

residuals = y_test - y_pred_best

#y_pred_best_reshaped = y_pred_best.reshape(-1, 1)
#linear_reg = LinearRegression().fit(y_pred_best_reshaped, y_test)
#slope = linear_reg.coef_[0]
#print(f"slope: {slope}")
#slope output was: 0.6890392014574755

#fig, axs = plt.subplots(3, 2, figsize=(14, 12))
#axs = axs.ravel()

#features = ['latitude', 'longitude']

#for i, feature in enumerate(features):
#    axs[i].scatter(X_test[feature], residuals, color='blue', edgecolor='k', alpha=0.6)
#    axs[i].set_title(f'Residuals vs {feature}')
#    axs[i].set_xlabel(feature)
#    axs[i].set_ylabel('Residuals')
#    axs[i].axhline(y=0, color='r', linestyle='--')

#fig.delaxes(axs[-1])

## Adjust layout
#plt.tight_layout()
#plt.show()

plt.figure(figsize=(10, 6))
plt.scatter((y_pred_best), residuals, color='blue', edgecolor='k', alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()


#============================

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


'''
#y_axis_cols = ['under_infl', 'speeding']
#x_axis_cols = ['injuries', 'serious_injuries', 'fatalities']
#df_filtered = df1[y_axis_cols + x_axis_cols].select_dtypes(include=[np.number])
#corr_matrix = df_filtered.corr()
#sns.heatmap(corr_matrix.loc[y_axis_cols, x_axis_cols], cmap='hot', annot=True)
#plt.show()

#df2 = pd.concat([df1.drop(['weather', 'roadcond', 'light_cond','severity_desc'], axis=1),
#                 pd.get_dummies(df1['weather']),
#                 pd.get_dummies(df1['roadcond']),
#                 pd.get_dummies(df1['light_cond'])], axis=1)

#print(df1.isnull().sum())

#labels = df2.columns
#fig, ax = plt.subplots()
#ax.set_xticks(np.arange(len(labels)))
#ax.set_xticklabels(labels,rotation=90, fontsize=6)
#ax.set_yticks(np.arange(len(labels)))
#ax.set_yticklabels(labels, fontsize=6)
#sns.heatmap(df2.corr(), cmap='hot', ax=ax)
#plt.show()
'''
