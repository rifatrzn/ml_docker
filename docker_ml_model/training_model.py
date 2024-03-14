import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load train and test datasets
train_df = pd.read_csv('./train_data.csv')
test_df = pd.read_csv('./test_data.csv')

# Convert 'timestamp' to datetime if it's not already
train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])

# Extract 'day_of_week' from 'timestamp'
train_df['day_of_week'] = train_df['timestamp'].dt.dayofweek
test_df['day_of_week'] = test_df['timestamp'].dt.dayofweek

# Assuming 'order_type' is categorical and 'order_quantity' and 'day_of_week' are numerical
categorical_features = ['order_type']
numerical_features = ['order_quantity', 'day_of_week']

# Create transformers for categorical and numerical features
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Combine transformers into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Prepare model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Extract features and target variable
X_train = train_df.drop('target_variable', axis=1)
y_train = train_df['target_variable']
X_test = test_df.drop('target_variable', axis=1)
y_test = test_df['target_variable']

# Train the model
model_pipeline.fit(X_train, y_train)

# Predict on test data
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
