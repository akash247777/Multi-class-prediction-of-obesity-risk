import streamlit as st 
import pandas as pd
from pycaret.classification import load_model, predict_model
from sklearn.preprocessing import LabelEncoder


# Function to load the trained model
def load_trained_model(model_path):
    return load_model(model_path)

# Load the trained model
model_path = "C:\\Users\\APL94855\\Downloads\\Multi-Class-Prediction-of-Obesity-Risk_May_2024-main\\final_model.pkl"  # Ensure the path is correct
model = load_trained_model(model_path)

# Dropdown options for user inputs
dropdown_options = {
    'id': [],
    'Gender': ['Female', 'Male'],
    'family_history_with_overweight': ['no', 'yes'],
    'frequent_consumption_of_high_caloric_food': ['no', 'yes'],
    'consumption_of_food_between_meals': ['Always', 'Frequently', 'Sometimes', 'no'],
    'SMOKE': ['no', 'yes'],
    'calories_consumption_monitoring': ['no', 'yes'],
    'consumption_of_alcohol': ['Frequently', 'Sometimes', 'no'],
    'transportation_used': ['Automobile', 'Bike', 'Motorbike', 'Public_Transportation', 'Walking'],
    'frequency_of_consumption_of_vegetables': ['Never', 'Sometimes', 'Frequently', 'Always'],
    'number_of_main_meals': ['1', '2', '3', '4', 'more than 4'],
    'consumption_of_water_daily': ['Less than a liter', '1-2 liters', 'more than 2 liters'],
    'physical_activity_frequency': ['None', '1-2 days', '2-4 days', '4-5 days', '6-7 days'],
    'time_using_technology_devices': ['0-2 hours', '3-5 hours', 'more than 5 hours']
}

# Streamlit app
st.title('Obesity Level Prediction')

# Create two columns
col1, col2 = st.columns(2)

# Collect user inputs using dropdowns, dividing the features between the two columns
user_input = {}

# Divide features equally into two lists
feature_list = list(dropdown_options.keys())
left_features = feature_list[:8]  # First 8 features
right_features = feature_list[8:]  # Remaining features

# Left column inputs (including Age, Height, and Weight)
with col1:
    st.subheader("Left Inputs")
    for feature in left_features:
        user_input[feature] = st.selectbox(f'Select {feature}', dropdown_options[feature])
    # Add Age, Height, and Weight to the left column
    user_input['Age'] = st.number_input('Age', min_value=1, max_value=100)
    

# Right column inputs
with col2:
    st.subheader("Right Inputs")
    for feature in right_features:
        user_input[feature] = st.selectbox(f'Select {feature}', dropdown_options[feature])
    user_input['Height'] = st.number_input('Height (in cm)', min_value=50, max_value=250)
    user_input['Weight'] = st.number_input('Weight (in kg)', min_value=10, max_value=200)    

# Convert user input into a DataFrame
input_df = pd.DataFrame([user_input])

# Display the user inputs
st.write('User Input:')
st.write(input_df)

# Encode categorical variables
label_encoders = {}
for feature in dropdown_options.keys():
    le = LabelEncoder()
    input_df[feature] = le.fit_transform(input_df[feature])
    label_encoders[feature] = le

# Display the encoded user inputs
st.write('Encoded User Input:')
st.write(input_df)

# Make predictions using the model
if st.button('Predict'):
    predictions = predict_model(model, data=input_df)
    
    # Debugging: Display the entire predictions DataFrame
    st.write('Predictions DataFrame:')
    st.write(predictions)
    
    # Extract the prediction label and score
    if 'prediction_label' in predictions.columns and 'prediction_score' in predictions.columns:
        prediction_label = predictions['prediction_label'][0]
        prediction_score = predictions['prediction_score'][0]
        st.write('Prediction Label:')
        st.write(prediction_label)
        st.write('Prediction Score:')
        st.write(prediction_score)
    else:
        st.write('Error: Expected columns "prediction_label" and "prediction_score" not found in predictions.')
