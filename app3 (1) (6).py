#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')


with open("/Users/sakethreddy/Downloads/label.pkl", 'rb') as file:
    label = pickle.load(file)
with open("/Users/sakethreddy/Downloads/model.pkl", 'rb') as file:
    model = pickle.load(file)
with open("/Users/sakethreddy/Downloads/scaler.pkl", 'rb') as file:
    scaler = pickle.load(file)
    
st.title("WORLD DEVELOPMENT MEASUREMENT PREDICTION")

all_features=[ 'CO2Emissions', 'GDP','InternetUsage','Energy Usage']


# Feature ranges (from your provided image)
feature_ranges = {
    'CO2Emissions': (7.0, 104443.0),
    'GDP': (63101270000.0, 227359000000.0),
    'InternetUsage': (0.0, 1.0),
    'Energy Usage' : (0.0, 1.0)
}

# All features (including Country)
all_features = list(feature_ranges.keys()) + ['Country']

# Load countries (as before)
df = pd.read_csv("/Users/sakethreddy/Downloads/Global_Development_Mesaurement (1).csv")
countries = sorted(df['Country'].unique())

st.markdown(
    """
    <style>
    
    /* Increase column width (adjust as needed) */
    .css-18e3th9 {
        max-width: 250px !important; # Adjust this value for width
        margin-right: 40px;        # Adjust for spacing between columns
    }

    .stButton > button {
            color: black;
            font-size: 16px;
            padding: 12px 24px;
            border-radius: 10px;
            border: 1px solid black;
            font-weight: bold;
            transition: background-color 0.3s ease;
            display: block;
            margin: 0 auto;
    } 

    .stButton > button:hover {
            background-color: blue;
            color: white;
    }

    .stMarkdown {
        font-size: 16px;
    }

    .stTitle {
            color: #4CAF50;
            text-align: center;
            font-weight: bold;
    }

    .prediction-box {
            border: 1px soild #4CAF50;
            padding: 20px;
            border-radius: 10px;
            background-color: #f9f9f9;
            color: #333;
            font-size: 18px;
            text-align: center;
            margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

def take_input(features, countries, feature_ranges):
    input_data = {}

    for i in range(0, len(features), 3):  # Iterate in steps of 3
        cols = st.columns(3)  # Create columns inside the loop
        col_index = 0
        features_in_row=features[i:min(i+3, len(features))]

        for feature in features_in_row: # Iterate on features_in_row
            if feature == 'Country':
                with cols[0]:
                    selected_country = st.selectbox("Select Country", countries)
                    input_data[feature] = selected_country
                    st.markdown("<br>", unsafe_allow_html=True)
            elif feature == 'GDP': # Special handling for GDP
                with cols[col_index]:
                    min_val, max_val = feature_ranges[feature]
                    gdp_billions = st.number_input(
                        f'Enter value for GDP (in billions, Range: {min_val/1e9:.2f} - {max_val/1e9:.2f})', # Display in billions
                        min_value=min_val/1e9, max_value=max_val/1e9, value=min_val/1e9, format="%.5f" 
                    )
                    input_data[feature] = gdp_billions * 1e9 # Store GDP in its original scale
                    st.markdown("<br>", unsafe_allow_html=True)

                col_index = (col_index + 1) % 3
            elif feature == 'PopulationTotal':  # Special handling for PopulationTotal
                with cols[col_index]:
                    min_val, max_val = feature_ranges[feature]
                    population_millions = st.number_input(
                        f'Enter value for Population Total (in millions, Range: {min_val/1e6:.2f} - {max_val/1e6:.2f})',
                        min_value=min_val/1e6, max_value=max_val/1e6, value=min_val/1e6, format="%.5f"  # Input, min, max in millions
                    )
                    input_data[feature] = population_millions * 1e6 # Store in original scale (millions)
                    st.markdown("<br>", unsafe_allow_html=True)

                col_index = (col_index + 1) % 3
            else:
                with cols[col_index]:
                    min_val, max_val = feature_ranges[feature]
                    feature_val = st.number_input(
                        f'Enter value for {feature} (Range: {min_val} - {max_val})',
                        min_value=min_val, max_value=max_val, value=min_val, format="%.5f"
                    )
                    input_data[feature] = feature_val
                    st.markdown("<br>", unsafe_allow_html=True)  # Add space below each input

                col_index +=1  # Increment col_index

    return input_data


def preprocess(input_data, all_features):
    input_list_numerical = []
    country = None

    for feature in all_features:
        if feature == 'Country':
            country = label.transform([input_data[feature]])[0]
        else:
            input_list_numerical.append(input_data[feature])

    # Scale numerical features only
    scaled_data_numerical = scaler.transform(np.array(input_list_numerical).reshape(1, -1))

    # Combine scaled numerical features and transformed country
    scaled_data = np.append(scaled_data_numerical, country)
    scaled_data=scaled_data.reshape(1,-1)
    return scaled_data


# Get user input
input_data = take_input(all_features, countries, feature_ranges)


cluster_interpretations = {
    0: "Developing Nations",
    1: "Developed Nations",
    2: "Emerging Economies or Developed Nations with Advanced Infrastructure",
}

# ... (take_input and preprocess functions - no changes)

if st.button("Predict"):
    scaled_data = preprocess(input_data, all_features)
    y_pred_test = model.predict(scaled_data)[0]

    # Get the interpretation from the dictionary
    interpretation = cluster_interpretations.get(y_pred_test, "Cluster interpretation not found.")  # Handle unknown clusters

    st.markdown(f'<div class="prediction-box"><b>Predicted Cluster</b>: {y_pred_test}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="prediction-box"><b>Interpretation</b>: {interpretation}</div>', unsafe_allow_html=True)


# In[ ]:




