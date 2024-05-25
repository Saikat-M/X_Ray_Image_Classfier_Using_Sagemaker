import streamlit as st
import json
import numpy as np
import boto3
from PIL import Image
import io


uploaded_file = st.file_uploader("Upload Image", accept_multiple_files=False)
if uploaded_file:
    st.image(uploaded_file)
    image = Image.open(uploaded_file)
    resized_image = image.resize((224, 224))

    # Display the resized image
    # st.image(np.asarray(resized_image))

    # Convert the resized image to bytes
    img_byte_arr = io.BytesIO()
    resized_image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    endpoint = 'Pneumonia-image-classifier'
    runtime = boto3.Session().client('sagemaker-runtime')
 
    # Send image via InvokeEndpoint API
    response = runtime.invoke_endpoint(EndpointName=endpoint, ContentType='application/x-image', Body=img_byte_arr)

    # Unpack response
    result = json.loads(response['Body'].read().decode())
    # st.write(result)
    predicted_class = np.argmax(result)
    st.write('predicted_class: ', predicted_class)
    if predicted_class == 0:
        st.write(f'The model classified the X-Ray image as Normal with probability score: {result[predicted_class]*100:.2f}')
    else:
        st.write(f'The model classified the X-Ray image having Pneumonia with probability score: {result[predicted_class]*100:.2f} ')