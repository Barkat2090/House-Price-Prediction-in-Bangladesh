# House Price Prediction in Bangladesh

This project uses a machine learning model to predict house prices in Bangladesh based on a dataset containing property details such as bedrooms, bathrooms, floor area, and location. The model is built using a Random Forest Regressor and includes data preprocessing, model training, evaluation, and a user-friendly interface for price prediction.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to predict house prices in Bangladesh using features like:

- Number of bedrooms and bathrooms
- Floor number and area (in square feet)
- Occupancy status (e.g., occupied, vacant)
- City and specific location

The project includes:

- Data preprocessing (handling missing values, encoding categorical variables)
- Training a Random Forest Regressor model
- Evaluating model performance using Mean Squared Error (MSE) and R² Score
- A command-line interface for users to input house details and get price predictions

## Dataset

The dataset (`house_price_bd.csv`) contains property listings in Bangladesh with the following columns:

- **Title**: Description of the property (dropped during preprocessing)
- **Bedrooms**: Number of bedrooms
- **Bathrooms**: Number of bathrooms
- **Floor_no**: Floor number of the property
- **Occupancy_status**: Whether the property is occupied or vacant
- **Floor_area**: Size of the property in square feet
- **City**: City where the property is located (e.g., Dhaka, Chattogram)
- **Price_in_taka**: Price of the property in Bangladeshi Taka (target variable)
- **Location**: Specific area within the city (e.g., Banani, Gulshan)

The dataset is cleaned to handle missing values and convert the price column to a numeric format by removing the currency symbol (৳).

## Installation

To run this project, you need Python 3.x and the following libraries:

- pandas
- numpy
- scikit-learn

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/house-price-prediction-bd.git
   cd house-price-prediction-bd
Install dependencies:
bash

pip install pandas numpy scikit-learn
Ensure the dataset (house_price_bd.csv) is in the project directory.
Usage

Run the script:
bash
python house_price_prediction.py

The script will:
Load and preprocess the dataset
Train the Random Forest model
Display model evaluation metrics (MSE and R² Score)
Prompt you to enter house details for price prediction


Example input:

Enter number of bedrooms: 3

Enter number of bathrooms: 2

Enter floor number: 4

Enter floor area (in sq ft): 1500

Enter occupancy status (e.g. occupied/vacant): vacant

Enter city (e.g. dhaka, chattogram): dhaka

Enter location (e.g. Banani): Banani
Output:

💰 Predicted House Price: ৳15,000,000
Model Details

Algorithm: Random Forest Regressor

Parameters: 100 estimators, random state = 42

Preprocessing:
Removed currency symbols from Price_in_taka and converted to float
Dropped irrelevant Title column

Handled missing values:
Numeric columns (Bedrooms, Bathrooms, Floor_area): Filled with median
Floor_no: Converted to numeric, filled with -1 for non-numeric values

Categorical columns (Occupancy_status, City, Location): Filled with 'Unknown'
Encoded categorical variables using LabelEncoder
Train-Test Split: 80% training, 20% testing

Evaluation Metrics:
Mean Squared Error (MSE)
R² Score
Evaluation

The model's performance is evaluated using:

Mean Squared Error (MSE): Measures the average squared difference between actual and predicted prices.

R² Score: Indicates the proportion of variance in the target variable explained by the model.


Example output:

✅ Model Evaluation:

Mean Squared Error: [value]

R² Score: [value]

Contributing
Contributions are welcome! To contribute:

Fork the repository
Create a new branch:
bash
git checkout -b feature-branch
Make your changes
Commit and push:
bash
git commit -m "Add feature"
git push origin feature-branch
Open a Pull Request
Please ensure your code follows PEP 8 style guidelines and includes appropriate documentation.

License
This project is licensed under the MIT License. See the LICENSE file for details.
```
