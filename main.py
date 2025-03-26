import streamlit as st
import tensorflow as tf
import numpy as np

# Load TensorFlow Model
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=[128, 128])
    input_array = tf.keras.preprocessing.image.img_to_array(image)
    input_array = np.array([input_array])  # Convert single image to batch
    prediction = model.predict(input_array)
    result_index = np.argmax(prediction)
    return result_index

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🌿 Dashboard")
app_mode = st.sidebar.radio("📌 Select Page", ["🏠 Home", "🚀 Get Started", "ℹ️ About"])

# --- HOME PAGE ---
if app_mode == "🏠 Home":
    st.markdown("""
    <h1 style='text-align: center;'>🌿 Plant Disease Recognition System</h1>
    <p style='text-align: center; font-size:18px;'>An advanced AI-powered tool to detect plant diseases efficiently.</p>
    <hr>
    
    ### 🔍 How It Works  
    1️⃣ **Upload an Image**: Go to the **Get Started** page and upload a plant image.  
    2️⃣ **Automated Analysis**: Our AI will analyze the image to identify diseases.  
    3️⃣ **Get Results**: View the diagnosis and recommended actions.  

    ### 💡 Why Choose Us?  
    ✅ **High Accuracy**: State-of-the-art AI for precise detection.  
    ✅ **User-Friendly**: A seamless and intuitive experience.  
    ✅ **Fast & Efficient**: Get results within seconds!  

    ---
    🌱 Click on **Get Started** in the sidebar to begin your diagnosis!
    """, unsafe_allow_html=True)

# --- ABOUT PAGE ---
elif app_mode == "ℹ️ About":
    st.markdown("""
    <h1 style='text-align: center;'>ℹ️ About Us</h1>
    <p style='text-align: center; font-size:18px;'>Revolutionizing plant health with AI-powered disease detection.</p>
    <hr>

    - 🧠 **Cutting-edge Machine Learning** ensures accurate disease detection.  
    - 🌍 **Empowering Farmers & Gardeners** with quick and reliable insights.  
    - 🌱 **Promoting Sustainable Farming** through early detection and intervention.  

    Let's work together to build a healthier future for plants and crops! 🌿  
    """, unsafe_allow_html=True)

# --- GET STARTED PAGE ---
elif app_mode == "🚀 Get Started":
    st.markdown("<h1 style='text-align: center;'>🚀 Disease Recognition</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size:18px;'>Upload a plant image to diagnose potential diseases.</p><hr>", unsafe_allow_html=True)

    test_image = st.file_uploader("📤 **Upload an Image** (JPG, PNG)", type=["jpg", "png", "jpeg"])

    if test_image:
        st.image(test_image, caption="Uploaded Image", use_column_width=True, channels="RGB")

    # Predict button
    if st.button("🔍 Predict Disease"):
        with st.spinner("Analyzing... Please wait ⏳"):
            result_index = model_prediction(test_image)
            
        # Disease Labels
        class_name = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
            'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
            'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]

        disease_name = class_name[result_index]
        st.success(f"🌿 The model predicts: **{disease_name}**")
        st.balloons()