import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

# Load the trained model
model = pk.load(open('rf_model.pkl', 'rb'))

st.header('Car Price Predictor')

# Load car details for reference
cars_data = pd.read_csv('pakwheels dataset.csv')

# Get unique car names and body types
unique_car_names = cars_data['Car_Name'].unique()

# Streamlit widgets for user input
name = st.selectbox('Select Car', unique_car_names)  # Car name selection
car_body = st.selectbox('Select Car Body Type', cars_data['Car_Body'].unique())  # Car body selection
year = st.slider('Year', 2014, 2025)  # Year selection
engine = st.slider('Engine CC', 660, 5000)  # Engine capacity slider
transmission = st.selectbox('Transmission type', cars_data['Transmission'].unique())  # Transmission type
fuel = st.selectbox('Fuel type', cars_data['Fueltype'].unique())  # Fuel type selection
mileage = st.slider('Car Mileage', 0, 200000)  # Mileage slider

if st.button("Predict"):
    # Prepare the input data for the model (remove 'Car_Name' as it's not used in the model)
    input_data_model = pd.DataFrame(
        [[car_body, name, year, engine, transmission, fuel, mileage]],
        columns=['Car_Body', 'Car_Name', 'Year', 'Engine', 'Transmission', 'Fueltype', 'Mileage']
    )

    # Preprocess categorical values to match the model's expected format
    input_data_model['Car_Body'].replace(['Hatchback', 'SUV', 'Crossover', 'Sedan', 'Micro Van', 'MPV',
                                          'Van', 'Compact SUV', 'Station Wagon', 'Mini Van', 'Double Cabin',
                                          'Convertible', 'Coupe', 'Pick Up', 'Mini Vehicles', 'High Roof',
                                          'Compact sedan', 'Compact hatchback', 'Truck', 'Single Cabin'],
                                         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], inplace=True)
    input_data_model['Car_Name'].replace(['Move', 'Land', 'Vitz', 'Alto', 'Passo', 'C-HR', 'Hustler', 'Corolla', 'Raize',
     'Fortuner', 'Vezel', 'N', 'Picanto', 'Yaris', 'Prius', 'Benz', 'Every', 'X70',
     'Grand', 'Wagon', 'Dayz', 'Mira', 'Prado', 'Flair', 'Clipper', 'Hiace', 'Vitara',
     'Aygo', 'Saga', 'HS', 'Fit', 'A3', 'BR-V', 'Sienta', 'Oshan', 'Rover', 'Celerio',
     'Sorento', 'LX', 'Cast', 'X1', 'Q2', 'A4', 'Note', 'V2', 'Ek', 'Cayenne', 'Grace',
     'A6', 'Boon', 'Aqua', 'Thor', 'Civic', 'Q7', 'Pearl', 'Hijet', 'Mehran', 'Xbee',
     'Tundra', 'Juke', 'Karvaan', 'Elantra', 'Harrier', 'Voxy', 'ZS', 'Stonic', 'Kizashi',
     'Freed', 'Crown', 'Glory', 'Sonata', 'Jimny', 'Premio', 'Roomy', 'Swift', 'Carol',
     'Camry', 'Insight', 'Alsvin', 'Tucson', 'CR-V', 'Tiggo', 'Sportage', 'MX', 'Jade',
     'RX', 'City', 'Tacoma', 'Moco', 'Taft', 'EK', 'Avanza', 'H6', 'Ciaz', 'Pixis', 'A5',
     'Hilux', 'Serena', 'S660', 'Tanto', 'Bravo', 'Spacia', 'Stella', 'F', 'A8', 'Roox',
     'X-PV', 'Cultus', 'BJ40', 'Rocky', 'Kicks', 'Sierra', 'Copen', 'Kona', 'Spike', 'Q3',
     'Tank', 'Mark', 'X2', 'X7', 'Noah', 'Jolion', 'Challenger', 'Terios', 'Panamera',
     'CT200h', 'Cx3', 'Esquire', 'Ignis', 'Bolan', 'MR', 'Rush', 'Accord', 'iQ', 'Wrangler',
     'D-Max', 'Ravi', 'Shehzore', 'Pleo', 'Vamos', 'Probox', 'Titan', 'X200', 'CR-Z', 'Z100',
     'Mirage', 'Mega', 'A7', 'Nx', 'HR-V', 'Niro', 'K07', 'Amarok', 'Gladiator', 'Patrol',
     'Atrai', 'Acty', 'Frontier', 'S', 'Otti', 'X', 'A800', 'Sirius', 'Ioniq', 'Life', 'Cerato',
     'Sambar', 'Gen', 'Minica'],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
     25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
     47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
     68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
     89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107,
     108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
     125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
     142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158,
     159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169], inplace=True)
    input_data_model['Fueltype'].replace(['Petrol', 'Hybrid', 'Diesel'], [1, 2, 3], inplace=True)
    input_data_model['Transmission'].replace(['Automatic', 'Manual'], [1, 2], inplace=True)
    


    # Ensure the column order matches the trained model
    input_data_model = input_data_model[['Car_Body','Car_Name', 'Year', 'Engine', 'Transmission', 'Fueltype', 'Mileage']]

    # Predict the car price using the trained model
    car_price = model.predict(input_data_model)

    # Show the predicted price
    st.markdown(f'Predicted Car Price: {car_price[0]:.2f}')