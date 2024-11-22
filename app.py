import os
# os.environ["KERAS_BACKEND"] = "tensorflow"


import streamlit as st
import tensorflow as tf
import numpy as np
import keras


def predection_model(test_image):
    
    model = tf.keras.models.load_model("newmodel.h5") #Loading the model
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64)) #Convert the image in the same format in which the model is trained
    input_arr = tf.keras.preprocessing.image.img_to_array(image) # Converting the image into array
    input_arr = np.array([input_arr]) #converting single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) # return index of max element


# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

# Main Page
if app_mode == "Home":
    st.header("FRUITS & VEGETABLES RECOGNITION SYSTEM")
    image_path = "pexels-photo-1132047.jpeg"
    st.image(image_path, width=300 )
    st.write("""
Welcome to the **Fruits & Vegetables Recognition System**! 🍎🥦  
This tool leverages the power of **Deep Learning** to identify fruits and vegetables from images.  
Whether you're exploring food datasets or building innovative applications, this system has you covered!
""")
    st.subheader("Key Features")
    st.markdown("""
- **Accurate Recognition**: Identify fruits and vegetables with high precision.  
- **User-Friendly Interface**: Simple and intuitive for easy navigation.  
- **Interactive Dashboard**: Upload images and view predictions instantly.  
- **Comprehensive Dataset**: Includes a wide range of fruits and vegetables.  
- **Future Scope**: Can be extended to include nutrition facts and recipes.
""")
    st.subheader("How to Use the App?")
    st.markdown("""
1. Navigate to the **Prediction** page using the sidebar.  
2. Upload an image of a fruit or vegetable.  
3. Click **Predict** to see the result.  
4. Explore the dataset in the **About Project** section.  
""")
    
    st.subheader("Did You Know?")
    st.info("""
- A **tomato** is technically a fruit.  
- **Carrots** were originally purple, not orange.  
- Eating fruits and vegetables daily can reduce the risk of chronic diseases by 30%.
""")
    st.write("### 'Healthy eating starts with knowing your food.' 🌱")






# About Project
elif app_mode == "About Project":
    # Main title with color
    st.markdown("<h1 style='color: #ff6347;'>About Project</h1>", unsafe_allow_html=True)
    
    # About Dataset Section
    st.markdown("<h2 style='color: #3b9a77;'>🍏 About Dataset</h2>", unsafe_allow_html=True)
    st.text("This dataset contains images of the following food items:")
    st.code("Fruits: banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.")
    st.code("Vegetables: cucumber, carrot, capsicum, onion, potato, lemon, tomato, radish, beetroot, cabbage, lettuce, spinach, soybean, cauliflower, bell pepper, chili pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalapeño, ginger, garlic, peas, eggplant.")
    
    # Tools and Technologies Section
    st.markdown("<h2 style='color: #3b9a77;'>🛠️ Tools and Technologies</h2>", unsafe_allow_html=True)
    st.markdown("""
    - **TensorFlow/Keras**: For deep learning model.  
    - **Streamlit**: For web application interface.  
    - **Python**: Programming language.  
    """)

    # Project Workflow Section
    st.markdown("<h2 style='color: #3b9a77;'>🔧 Project Workflow</h2>", unsafe_allow_html=True)
    st.write("""
    1. **Upload Image**: Upload an image of a fruit/vegetable.
    2. **Preprocess Image**: Resize and normalize image.
    3. **Predict**: Model predicts the image category.
    4. **Result**: Display predicted label.
    """)

    # Limitations Section
    st.markdown("<h2 style='color: #3b9a77;'>⚠️ Limitations</h2>", unsafe_allow_html=True)
    st.write("""
    - Limited to dataset.
    - May struggle with poor-quality images.
    - Requires more data to improve accuracy.
    """)

    # Banner Image for Sample Dataset
    st.markdown("### 📸 Sample Image from Dataset", unsafe_allow_html=True)
    image_path1 = "image.jpg"
    st.image(image_path1, caption="Example: Apple")







# Prediction Page
elif app_mode == "Prediction":
    # Header with styling
    st.markdown("<h1 style='text-align: center; color: #0073e6;'>🍎 Welcome to the Prediction Page 🍆</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Upload an image of a fruit or vegetable, and our model will identify it for you!</p>", unsafe_allow_html=True)
    
    # File uploader
    test_image = st.file_uploader("📤 Upload an image file (JPG, PNG, or JPEG):", type=["jpg", "png", "jpeg"])

    if test_image is not None:  # Check if an image has been uploaded
        # Display the uploaded image
        st.markdown("<h3 style='text-align: center;'>Uploaded Image Preview:</h3>", unsafe_allow_html=True)
        st.image(test_image, caption="Your Uploaded Image", use_container_width=False, width=300)

        # Center-aligned Predict button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            predict_button = st.button("🔍 Predict")

        if predict_button:  # Predict only when the button is clicked
            # Display loading spinner
            with st.spinner("Analyzing the image... Please wait!"):
                # Load and preprocess the image
                image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
                input_arr = tf.keras.preprocessing.image.img_to_array(image)
                input_arr = np.array([input_arr])  # Convert to batch format

                # Load the model and make predictions
                model = tf.keras.models.load_model("newmodel.h5")
                predictions = model.predict(input_arr)
                result_index = np.argmax(predictions)

                # Reading labels from the result file
                with open("result.txt") as f:
                    content = f.readlines()
                result = [i.strip() for i in content]

                # Display the prediction result
                st.success(f"🌟 The model predicts it is a **{result[result_index]}**!")
                st.balloons()  # Celebrate with balloons animation
    else:
        st.warning("⚠️ Please upload an image file for prediction.")




# if __name__ == '__main__':
#     main()
