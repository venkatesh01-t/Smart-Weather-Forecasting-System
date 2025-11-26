import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
import base64

# Load the pre-trained model
with open('model/weather_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Path to your background image
background_image_path = 'static/background/backgrouns.jpg'

# Encode the background image to base64
with open(background_image_path, 'rb') as image_file:
    background_image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

# Set background image using base64
st.markdown(
    f"""
    <style>
        body {{
            background-image: url('data:static/background/backgrouns.jpg;base64,{background_image_base64}');
            background-size: cover;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    st.title('Weather Prediction App')

    # Sidebar
    st.sidebar.title('Navigation')
    app_mode = st.sidebar.selectbox('Choose a page', ['Home', 'Graphical Visualization', 'Predict', 'Future Scope'])

    if app_mode == 'Home':
        show_homepage()
    elif app_mode == 'Graphical Visualization':
        show_graphical_visualization()
    elif app_mode == 'Predict':
        show_prediction()
    elif app_mode == 'Future Scope':
        show_future_scope()

def show_homepage():
    st.header('Abstract')
    paragraph = """
        <p style="text-align: justify;">
            The weather prediction project utilizes a Random Forest machine learning algorithm to forecast future weather conditions based on historical data. 
            The project involves preprocessing meteorological data, including variables such as precipitation, temperature, wind speed, and date. 
            The Random Forest model is trained on a labeled dataset, learning complex patterns and relationships within the data. 
            During the prediction phase, users input specific weather parameters for a given date, and the model generates predictions for the weather conditions. 
            The project aims to provide accurate and reliable weather forecasts, contributing to improved decision-making in various sectors, 
            such as agriculture, transportation, and event planning. 
            The Random Forest algorithm's ensemble of decision trees enhances the model's robustness and generalization capabilities, 
            making it well-suited for handling diverse weather patterns and improving forecast accuracy.
        </p>
        """
    
        # Render the HTML paragraph
    st.markdown(paragraph, unsafe_allow_html=True)
def show_graphical_visualization():
    st.header('Graphical Visualization')

    # Assuming you have 10 images saved as 'model/1.png', 'model/2.png', ..., 'model/10.png'
    images = [f'model/{i}.png' for i in range(1, 11)]

    # Display images in a 2x5 table with all images fitting inside the columns
    for i in range(0, len(images), 5):
        col1, col2, col3, col4, col5 = st.columns(5)
        for j in range(i, i+5):
            if j == i:
                with col1:
                    st.image(images[j], caption=f'Image {j+1}', width=150, use_column_width=True)
            elif j == i + 1:
                with col2:
                    st.image(images[j], caption=f'Image {j+1}', width=150, use_column_width=True)
            elif j == i + 2:
                with col3:
                    st.image(images[j], caption=f'Image {j+1}', width=150, use_column_width=True)
            elif j == i + 3:
                with col4:
                    st.image(images[j], caption=f'Image {j+1}', width=150, use_column_width=True)
            elif j == i + 4:
                with col5:
                    st.image(images[j], caption=f'Image {j+1}', width=150, use_column_width=True)

def show_prediction():

    # User input for future weather prediction
    st.header('Enter Future Weather Data:')

    # Create a 2x3 table for user input
    col1, col2, col3 = st.columns(3)

    with col1:
        date = st.date_input('Date:', min_value=datetime.today())
        precipitation = st.number_input('Precipitation (mm):', min_value=0.0, max_value=100.0, key='precipitation')

    with col2:
        temp_max = st.number_input('Max Temperature (°C):', min_value=-20.0, max_value=50.0, key='temp_max')
        wind = st.number_input('Wind Speed (m/s):', min_value=0.0, max_value=30.0, key='wind')

    with col3:
        temp_min = st.number_input('Min Temperature (°C):', min_value=-20.0, max_value=50.0, key='temp_min')

    # Predict button
    if st.button('Predict'):
        # Make predictions based on user input
        user_data = pd.DataFrame({
            'date': [pd.to_datetime(date, errors='coerce')],  # Handle invalid date input gracefully
            'precipitation': [precipitation],
            'temp_max': [temp_max],
            'temp_min': [temp_min],
            'wind': [wind]
        })
        
        # Drop rows with invalid dates
        user_data = user_data.dropna(subset=['date'])
        
        if user_data.empty:
            st.error('Invalid date input. Please provide a valid date.')
        else:
            user_data['dayofweek'] = user_data['date'].dt.dayofweek
            user_data['day'] = user_data['date'].dt.day
            user_data['month'] = user_data['date'].dt.month
            user_data['year'] = user_data['date'].dt.year
        
            user_data['weather_prediction'] = model.predict(user_data.drop('date', axis=1))[0]
        
            # Display the predicted weather
            st.success(f"The predicted weather for {date} is: {user_data['weather_prediction'].iloc[0]}")
            
            # Display sunny GIF if predicted weather is 'sun'
            if user_data['weather_prediction'].iloc[0].lower() == 'sun':
                st.image('static/sunny.gif', caption='Sunny Day', use_column_width=True)
            elif user_data['weather_prediction'].iloc[0].lower() == 'rain':
                st.image('static/rainy.gif', caption='Rainy Day', use_column_width=True)
            if user_data['weather_prediction'].iloc[0].lower() == 'snow':
                st.image('static/snow.gif', caption='Snow Day', use_column_width=True)
            if user_data['weather_prediction'].iloc[0].lower() == 'drizzle':
                st.image('static/drizzle.gif', caption='Drizzle Day', use_column_width=True)
            if user_data['weather_prediction'].iloc[0].lower() == 'fog':
                st.image('static/foggy.gif', caption='Foggy Day', use_column_width=True)

def show_future_scope():
    st.header('Future scope')
    paragraph = """
        <p style="text-align: justify;">
            The future scope of the weather prediction project employing Random Forest entails several promising avenues. First, continuous refinement and augmentation of the dataset can enhance the model's predictive capabilities, allowing it to adapt to evolving climate patterns. Integration with real-time weather data feeds and advanced sensor technologies can further improve the model's accuracy and responsiveness. Additionally, the project could explore incorporating geographical data and satellite imagery for a more comprehensive understanding of local weather phenomena. Collaboration with meteorological agencies and research institutions could facilitate the utilization of cutting-edge climate models and scientific advancements, enabling the development of a state-of-the-art forecasting system. Furthermore, exploring the integration of artificial intelligence techniques, such as neural networks, could open new horizons for capturing intricate weather patterns. Ultimately, the project's future endeavors aim to advance the accuracy and reliability of weather predictions, making substantial contributions to fields like disaster preparedness, resource management, and climate research.
        </p>
    """
        
    # Render the HTML paragraph with justified text
    st.markdown(paragraph, unsafe_allow_html=True)
if __name__ == '__main__':
    main()
