import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import os
from PIL import Image
import pickle

# Set page config
st.set_page_config(
    page_title="Melanoma Cancer Detection",
    page_icon="🩺",
    layout="wide"
)

# Load model and class indices (with caching)
@st.cache(allow_output_mutation=True)
def load_melanoma_model():
    try:
        model = load_model('models/melanoma_model_finetuned.keras')
        return model
    except:
        try:
            model = load_model('models/melanoma_model_final.keras')
            return model
        except:
            try:
                model = load_model('models/melanoma_model_best.keras')
                return model
            except:
                st.error("Model not found. Please train the model first.")
                return None

@st.cache(allow_output_mutation=True)
def load_class_indices():
    try:
        with open('models/class_indices.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        st.error("Class indices not found. Please train the model first.")
        return None

# Preprocess image
def preprocess_image(image, target_size=(224, 224)):
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image to match model input
    image = image.resize(target_size)
    # Convert to array and normalize
    img_array = img_to_array(image) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Main app
def main():
    st.title("🩺 Melanoma Cancer Detection")
    st.write("Upload a skin lesion image to classify it into one of 9 categories")
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info(
        "This app uses a deep learning model to classify skin lesions into 9 categories. "
        "Upload an image of a skin lesion for analysis. "
        "**Note: This is for educational purposes only and not a substitute for professional medical diagnosis.**"
    )
    
    # Load model and class indices
    model = load_melanoma_model()
    class_indices = load_class_indices()
    
    if model is not None and class_indices is not None:
        # Create reverse mapping from index to class name
        idx_to_class = {v: k for k, v in class_indices.items()}
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Preprocess and predict
            processed_image = preprocess_image(image)
            
            # Make prediction
            predictions = model.predict(processed_image)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = idx_to_class[predicted_class_idx]
            confidence = predictions[0][predicted_class_idx]
            
            # Display results
            st.subheader("Results")
            
            # Top prediction
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Class", predicted_class)
            with col2:
                st.metric("Confidence", f"{confidence:.4f}")
            
            # All predictions
            st.subheader("All Predictions")
            
            # Create dataframe for visualization
            results_df = pd.DataFrame({
                'Class': [idx_to_class[i] for i in range(len(idx_to_class))],
                'Confidence': predictions[0]
            }).sort_values('Confidence', ascending=False)
            
            # Display as table
            st.dataframe(results_df.style.format({'Confidence': '{:.4f}'}))
            
            # Display as bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(results_df['Class'], results_df['Confidence'] * 100)
            ax.set_xlabel('Confidence (%)')
            ax.set_title('Prediction Confidence for Each Class')
            
            # Add value labels on bars
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                       f'{width:.1f}%', ha='left', va='center')
            
            st.pyplot(fig)
            
            # Highlight if melanoma is detected
            if predicted_class.lower() == 'melanoma':
                st.error("⚠️ **Melanoma detected!** Please consult a dermatologist for proper diagnosis.")
            else:
                st.success("✅ **No melanoma detected.** However, please consult a doctor for any skin concerns.")
            
            # Disclaimer
            st.warning(
                "**Disclaimer:** This tool is for educational purposes only. "
                "It is not a substitute for professional medical advice, diagnosis, or treatment. "
                "Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition."
            )
            
            # Additional information
            with st.expander("Learn more about skin cancer types"):
                st.write("""
                **Common skin cancer types:**
                
                - **Actinic Keratosis**: Precancerous scaly spots caused by sun damage
                - **Basal Cell Carcinoma**: Most common skin cancer, rarely spreads
                - **Dermatofibroma**: Benign skin growth, usually harmless
                - **Melanoma**: Most serious skin cancer, can spread if not treated early
                - **Nevus**: Common mole, usually benign
                - **Pigmented Benign Keratosis**: Age spots or liver spots
                - **Seborrheic Keratosis**: Non-cancerous skin growths
                - **Squamous Cell Carcinoma**: Common skin cancer, can spread if not treated
                - **Vascular Lesion**: Blood vessel abnormalities
                
                **ABCDE rule for melanoma detection:**
                - **A**symmetry: Irregular shape
                - **B**order: Notched, irregular borders
                - **C**olor: Varied pigmentation
                - **D**iameter: Larger than 6mm
                - **E**volving: Changing in size, shape, or color
                
                If you notice any suspicious changes in your skin, consult a dermatologist.
                """)
    else:
        st.info("Please train the model first by running the Jupyter notebook.")

if __name__ == "__main__":
    main()