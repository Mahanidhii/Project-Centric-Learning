# Streamlit Application
import streamlit as st

st.set_page_config(
    page_title="AI Vehicle Insurance System",
    page_icon="PCL",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Navigation Setup 
home_page = st.Page("pages/home.py", title="Home Page", default=True)
damage_page = st.Page("pages/damage_detection.py", title="Damage Detection",)
claim_page = st.Page("pages/claim_estimation.py", title="Claim Estimator")
pg = st.navigation([home_page, damage_page, claim_page])
pg.run()