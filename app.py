import streamlit as st

# Title and description
st.set_page_config(page_title="SpineScope App", layout="wide")
st.title("🩻 SpineScope – Spinal Condition Classifier")
st.markdown(
    """
    Welcome to the **SpineScope Streamlit App**!  
    This tool allows you to explore spinal condition classification models
    and test predictions interactively.
    """
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Model Info", "🔮 Predictions"])

# Pages
if page == "🏠 Home":
    st.header("Home")
    st.write("This is the landing page. Use the sidebar to navigate.")
    
elif page == "📊 Model Info":
    st.header("Model Information")
    st.write("Here you’ll display summaries, charts, and evaluation metrics for your models.")
    
elif page == "🔮 Predictions":
    st.header("Make a Prediction")
    st.write("This will later allow users to input features and get predictions from trained models.")
    # Example input field
    age = st.number_input("Enter patient age:", min_value=0, max_value=120, value=30)
    st.write(f"Selected Age: {age}")
