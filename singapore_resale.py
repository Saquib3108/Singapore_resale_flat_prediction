import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
from PIL import Image

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1rem;
        color: #4A90E2;
    }
    .instructions {
        font-size: 1.1rem;
        color: #333;
        margin-bottom: 1rem;
    }
    .input-label {
        font-weight: bold;
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True
)

def town_mapping(town_map):
    town_dict = {
        'ANG MO KIO': 0, 'BEDOK': 1, 'BISHAN': 2, 'BUKIT BATOK': 3, 'BUKIT MERAH': 4,
        'BUKIT PANJANG': 5, 'BUKIT TIMAH': 6, 'CENTRAL AREA': 7, 'CHOA CHU KANG': 8,
        'CLEMENTI': 9, 'GEYLANG': 10, 'HOUGANG': 11, 'JURONG EAST': 12, 'JURONG WEST': 13,
        'KALLANG/WHAMPOA': 14, 'MARINE PARADE': 15, 'PASIR RIS': 16, 'PUNGGOL': 17,
        'QUEENSTOWN': 18, 'SEMBAWANG': 19, 'SENGKANG': 20, 'SERANGOON': 21, 'TAMPINES': 22,
        'TOA PAYOH': 23, 'WOODLANDS': 24, 'YISHUN': 25
    }
    return town_dict.get(town_map, -1)

def flat_type_mapping(flt_type):
    flat_type_dict = {
        '1 ROOM': 0, '2 ROOM': 1, '3 ROOM': 2, '4 ROOM': 3, '5 ROOM': 4,
        'EXECUTIVE': 5, 'MULTI-GENERATION': 6
    }
    return flat_type_dict.get(flt_type, -1)

def flat_model_mapping(fl_m):
    flat_model_dict = {
        '2-room': 0, '3Gen': 1, 'Adjoined flat': 2, 'Apartment': 3, 'DBSS': 4,
        'Improved': 5, 'Improved-Maisonette': 6, 'Maisonette': 7, 'Model A': 8,
        'Model A-Maisonette': 9, 'Model A2': 10, 'Multi Generation': 11,
        'New Generation': 12, 'Premium Apartment': 13, 'Premium Apartment Loft': 14,
        'Premium Maisonette': 15, 'Simplified': 16, 'Standard': 17, 'Terrace': 18,
        'Type S1': 19, 'Type S2': 20
    }
    return flat_model_dict.get(fl_m, -1)

def predict_price(year, town, flat_type, flr_area_sqm, flat_model, stry_start, stry_end, re_les_year, re_les_month, les_coms_dt):
    # Ensure valid values for logarithmic transformation
    if stry_start <= 0 or stry_end <= 0:
        raise ValueError("Storey Start and Storey End must be positive integers.")

    year_1 = int(year)
    town_2 = town_mapping(town)
    flt_ty_2 = flat_type_mapping(flat_type)
    flr_ar_sqm_1 = int(flr_area_sqm)
    flt_model_2 = flat_model_mapping(flat_model)
    str_str = np.log(int(stry_start))
    str_end = np.log(int(stry_end))
    rem_les_year = int(re_les_year)
    rem_les_month = int(re_les_month)
    lese_coms_dt = int(les_coms_dt)

    with open("Resale_Flat_Prices_Model_1.pkl", "rb") as f:
        regg_model = pickle.load(f)

    user_data = np.array([[year_1, town_2, flt_ty_2, flr_ar_sqm_1, flt_model_2, str_str, str_end, rem_les_year, rem_les_month, lese_coms_dt]])
    y_pred_1 = regg_model.predict(user_data)
    price = np.exp(y_pred_1[0])

    return round(price)


st.markdown('<div class="main-title">Singapore Resale Flat Price Predictor</div>', unsafe_allow_html=True)

with st.sidebar:
    select = option_menu("MAIN MENU", ["Home", "Price Prediction", "About"], 
                         icons=['house', 'graph-up', 'info-circle'], menu_icon="cast", default_index=0)

if select == "Home":
    img = Image.open("Resale_flat_prediction.jpg")
    st.image(img, use_column_width=True)

    st.markdown('<div class="section-header">HDB Flats</div>', unsafe_allow_html=True)
    st.write('''The majority of Singaporeans live in public housing provided by the HDB.
    HDB flats can be purchased either directly from the HDB as a new unit or through the resale market from existing owners.''')
    
    st.markdown('<div class="section-header">Resale Process</div>', unsafe_allow_html=True)
    st.write('''In the resale market, buyers purchase flats from existing flat owners, and the transactions are facilitated through the HDB resale process.
    The process involves a series of steps, including valuation, negotiations, and the submission of necessary documents.''')
    
    st.markdown('<div class="section-header">Valuation</div>', unsafe_allow_html=True)
    st.write('''The HDB conducts a valuation of the flat to determine its market value. This is important for both buyers and sellers in negotiating a fair price.''')
    
    st.markdown('<div class="section-header">Eligibility Criteria</div>', unsafe_allow_html=True)
    st.write("Buyers and sellers in the resale market must meet certain eligibility criteria, including citizenship requirements and income ceilings.")
    
    st.markdown('<div class="section-header">Resale Levy</div>', unsafe_allow_html=True)
    st.write("For buyers who have previously purchased a subsidized flat from the HDB, there might be a resale levy imposed when they purchase another flat from the HDB resale market.")
    
    st.markdown('<div class="section-header">Grant Schemes</div>', unsafe_allow_html=True)
    st.write("There are various housing grant schemes available to eligible buyers, such as the CPF Housing Grant, which provides financial assistance for the purchase of resale flats.")
    
    st.markdown('<div class="section-header">HDB Loan and Bank Loan</div>', unsafe_allow_html=True)
    st.write("Buyers can choose to finance their flat purchase through an HDB loan or a bank loan. HDB loans are provided by the HDB, while bank loans are obtained from commercial banks.")
    
    st.markdown('<div class="section-header">Market Trends</div>', unsafe_allow_html=True)
    st.write("The resale market is influenced by various factors such as economic conditions, interest rates, and government policies. Property prices in Singapore can fluctuate based on these factors.")
    
    st.markdown('<div class="section-header">Online Platforms</div>', unsafe_allow_html=True)
    st.write("There are online platforms and portals where sellers can list their resale flats, and buyers can browse available options.")

elif select == "Price Prediction":
    st.markdown('<div class="section-header">Predict Resale Price</div>', unsafe_allow_html=True)
    st.markdown('<div class="instructions">Please fill in the details of the flat below:</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="input-label">Year</div>', unsafe_allow_html=True)
        year = st.selectbox("", [str(i) for i in range(2015, 2025)])
        
        st.markdown('<div class="input-label">Town</div>', unsafe_allow_html=True)
        town = st.selectbox("", ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
                                                'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
                                                'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
                                                'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
                                                'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
                                                'TOA PAYOH', 'WOODLANDS', 'YISHUN'])
        
        st.markdown('<div class="input-label">Flat Type</div>', unsafe_allow_html=True)
        flat_type = st.selectbox("", ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION'])
        
        st.markdown('<div class="input-label">Floor Area (sqm)</div>', unsafe_allow_html=True)
        flr_area_sqm = st.number_input("", min_value=20, max_value=200)
    
    with col2:
        st.markdown('<div class="input-label">Flat Model</div>', unsafe_allow_html=True)
        flat_model = st.selectbox("", ['2-room', '3Gen', 'Adjoined flat', 'Apartment', 'DBSS', 'Improved',
                                       'Improved-Maisonette', 'Maisonette', 'Model A', 'Model A-Maisonette',
                                       'Model A2', 'Multi Generation', 'New Generation', 'Premium Apartment',
                                       'Premium Apartment Loft', 'Premium Maisonette', 'Simplified', 'Standard',
                                       'Terrace', 'Type S1', 'Type S2'])
        
        st.markdown('<div class="input-label">Storey Start</div>', unsafe_allow_html=True)
        stry_start = st.number_input("", min_value=1, max_value=50)
        
        st.markdown('<div class="input-label">Storey End</div>', unsafe_allow_html=True)
        stry_end = st.number_input("", min_value=1, max_value=100)
        
        st.markdown('<div class="input-label">Remaining Lease Year</div>', unsafe_allow_html=True)
        re_les_year = st.number_input("", min_value=1, max_value=99)
        
        st.markdown('<div class="input-label">Remaining Lease Month</div>', unsafe_allow_html=True)
        re_les_month = st.number_input("", min_value=0, max_value=11)
        
        st.markdown('<div class="input-label">Lease Commencement Date</div>', unsafe_allow_html=True)
        les_coms_dt = st.number_input("", min_value=1966, max_value=2017)
    
    if st.button("Predict Resale Price"):
        try:
            pre_price = predict_price(year, town, flat_type, flr_area_sqm, flat_model, stry_start, stry_end, re_les_year, re_les_month, les_coms_dt)
            st.success(f"Resale Price of Flat will be: ${pre_price}")
        except ValueError as e:
            st.error(e)

elif select == "About":
    st.markdown('<div class="section-header">About This Project</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Data Collection and Preprocessing</div>', unsafe_allow_html=True)
    st.write("Collect a dataset of resale flat transactions from the Singapore Housing and Development Board (HDB) for the years 1990 to Till Date. Preprocess the data to clean and structure it for machine learning.")
    
    st.markdown('<div class="section-header">Feature Engineering</div>', unsafe_allow_html=True)
    st.write("Extract relevant features from the dataset, including town, flat type, storey range, floor area, flat model, and lease commence date. Create any additional features that may enhance prediction accuracy.")
    
    st.markdown('<div class="section-header">Model Selection and Training</div>', unsafe_allow_html=True)
    st.write("Choose an appropriate machine learning model for regression (e.g., linear regression, decision trees, or random forests). Train the model on the historical data, using a portion of the dataset for training.")
    
    st.markdown('<div class="section-header">Model Evaluation</div>', unsafe_allow_html=True)
    st.write("Evaluate the model's predictive performance using regression metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), or Root Mean Squared Error (RMSE) and R2 Score.")
    
    st.markdown('<div class="section-header">Streamlit Web Application</div>', unsafe_allow_html=True)
    st.write("Develop a user-friendly web application using Streamlit that allows users to input details of a flat (town, flat type, storey range, etc.). Utilize the trained machine learning model to predict the resale price based on user inputs.")
    
    st.markdown('<div class="section-header">Deployment on Render</div>', unsafe_allow_html=True)
    st.write("Deploy the Streamlit application on the Render platform to make it accessible to users over the internet.")
    
    st.markdown('<div class="section-header">Testing and Validation</div>', unsafe_allow_html=True)
    st.write("Thoroughly test the deployed application to ensure it functions correctly and provides accurate predictions.")