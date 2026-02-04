# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 17:55:59 2026

@author: HJNmusicBaby
"""

import streamlit as st
import joblib
import folium
import numpy as np
from streamlit_folium import st_folium
import geopy
from geopy.geocoders import Nominatim


#Load trained model
rf_model = joblib.load("rf_model.pkl") #Loading pre-trained model


#Title
st.title("🏡 House Price Prediction App")

#Input fields for users
income = st.number_input("Median Income", min_value=0)
income /= 10000 #income = income\10000

lat = st.number_input("Latitude", min_value=-32.0, max_value=180.0)

long = st.number_input("Longitude", min_value=-180.0, max_value=180.0)

rmse = 48000

#Reverse geocoding to get the city name from the coordinate
#Initialize the Nominatim API
try:
    geolocator = Nominatim(user_agent="my_geopy_app")
    location = geolocator.reverse(f"{lat},{long}")
    
    city = "Unknown"
    
    if location and "address" in location.raw:
        addr = location.raw["address"]
        #get city name
        city = addr["city"]



    #Create and display map
    m = folium.Map(location = [lat, long], zoom_start=10)
    folium.Marker([lat, long], popup=city).add_to(m)
    
    st_folium(m) #Displays of rendeers folium map in the streamlit app
except Exception as e:
    print(f"Sorry the map is not available at the moment. {e}")

if st.button("Predict"):
    features = np.array([[income, lat, long]])
    prediction = rf_model.predict(features)
    prediction *= 100000
    lower_range = prediction[0] - rmse
    upper_range = prediction[0] + rmse
    st.success(f"Estimated Predicted House Price $: {lower_range:,.2f} - {upper_range:,.2f}")
    

    
