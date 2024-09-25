import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd
import os
import re

def main():

    # Set the title and description of the app
    st.title('UtabiriFarm Crop Disease Prediction')

    st.write("""
    Upload an image of a crop leaf to get a prediction of potential diseases.
    Supported crops: **Potato**, **Maize**, and **Tomato**.
    """)

    # URL of the Flask backend
    backend_url = 'http://127.0.0.1:5000/predict'  # Adjust the URL if needed

    # Path to the prediction history file
    prediction_file = '/home/sudotechpro/alx_learn/UtabiriFarm-Backend/predictions/predictions.txt'

    # Function to load the prediction history
    def load_prediction_history(file_path):
        if not os.path.exists(file_path):
            st.write("Prediction file does not exist.")
            return pd.DataFrame()

        # Regular expression pattern to match each line
        pattern = re.compile(
            r'^(?P<timestamp>[^,]+),\s*User ID:\s*(?P<user_id>[^,]+),\s*Image:\s*(?P<image>[^,]+),\s*Prediction:\s*(?P<prediction>[^,]+),\s*Confidence:\s*(?P<confidence>[\d\.]+)'
        )

        data = []

        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue  # Skip empty lines
                    match = pattern.match(line)
                    if match:
                        entry = match.groupdict()
                        data.append(entry)
                    else:
                        st.error(f"Line format incorrect: {line}")

            # Create DataFrame
            df = pd.DataFrame(data)
            if df.empty:
                st.write("No prediction history available.")
                return pd.DataFrame()

            # Convert data types
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['confidence'] = df['confidence'].astype(float)

            return df
        except Exception as e:
            st.error(f"Error reading prediction history: {e}")
            return pd.DataFrame()

    # Load the prediction history
    history_df = load_prediction_history(prediction_file)

    # Display the prediction history in the sidebar
    st.sidebar.title('Prediction History')

    if not history_df.empty:
        # Format the DataFrame
        history_df['timestamp'] = history_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        history_df['confidence'] = history_df['confidence'].apply(lambda x: f"{x * 100:.2f}%")
        
        # Display the DataFrame
        st.sidebar.table(history_df)
    else:
        st.sidebar.write('No prediction history available.')
    # Add a file uploader
    uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Read and display the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Add a predict button
        if st.button('Predict'):
            with st.spinner('Processing...'):
                # Convert the image to bytes
                buf = io.BytesIO()
                image.save(buf, format='JPEG')
                byte_im = buf.getvalue()

                # Prepare the files payload
                files = {'file': ('image.jpg', byte_im, 'image/jpeg')}

                try:
                    # Send the POST request to the backend
                    response = requests.post(backend_url, files=files)
                    response.raise_for_status()
                    
                    result = response.json()
                    # Display the prediction
                    st.success('Prediction Complete')
                    st.write(f"**Prediction:** {result['class']}")
                    st.write(f"**Confidence:** {result['confidence'] * 100:.2f}%")

                    # Reload the prediction history after a new prediction
                    history_df = load_prediction_history(prediction_file)

                    
                except requests.exceptions.RequestException as e:
                    st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()