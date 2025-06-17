import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("house_price_bd.csv")

# Clean the 'Price_in_taka' column
df['Price_in_taka'] = df['Price_in_taka'].replace('[৳,]', '', regex=True).astype(float)
df = df.dropna(subset=['Price_in_taka'])

# Drop irrelevant column
df = df.drop(columns=['Title'], errors='ignore')

# Handle missing values
df['Bedrooms'] = df['Bedrooms'].fillna(df['Bedrooms'].median())
df['Bathrooms'] = df['Bathrooms'].fillna(df['Bathrooms'].median())
df['Floor_area'] = df['Floor_area'].fillna(df['Floor_area'].median())
df['Floor_no'] = pd.to_numeric(df['Floor_no'], errors='coerce').fillna(-1)
df['Occupancy_status'] = df['Occupancy_status'].fillna('Unknown')
df['Location'] = df['Location'].fillna('Unknown')
df['City'] = df['City'].fillna('Unknown')

# Encode categorical columns
label_encoders = {}
for col in ['Occupancy_status', 'City', 'Location']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and Target
X = df.drop(columns=['Price_in_taka'])
y = df['Price_in_taka']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("✅ Model Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# --- Helper: Safe encoder for unseen values ---
def safe_transform(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    elif 'Unknown' in encoder.classes_:
        return encoder.transform(['Unknown'])[0]
    else:
        return encoder.transform([encoder.classes_[0]])[0]

# --- User Input Section ---
print("\n📊 Enter house details to predict price:\n")
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))
floor_no = int(input("Enter floor number: "))
floor_area = float(input("Enter floor area (in sq ft): "))
occupancy = input("Enter occupancy status (e.g. occupied/vacant): ").strip().lower()
city = input("Enter city (e.g. dhaka, chattogram): ").strip().lower()
location = input("Enter location (e.g. Banani): ").strip()

# Prepare input data in correct column order
input_values = [[
    bedrooms,
    bathrooms,
    floor_no,
    floor_area,
    safe_transform(label_encoders['Occupancy_status'], occupancy),
    safe_transform(label_encoders['City'], city),
    safe_transform(label_encoders['Location'], location)
]]

input_data = pd.DataFrame(input_values, columns=X.columns)

# Predict and output
predicted_price = model.predict(input_data)[0]
print(f"\n💰 Predicted House Price: ৳{predicted_price:,.0f}")
