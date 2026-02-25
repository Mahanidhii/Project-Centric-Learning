import streamlit as st

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        padding-bottom: 2rem;
    }
    .feature-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-header"> AI Vehicle Insurance Claim Assessment</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Upload a photo of vehicle damage and get an AI-powered claim estimate</p>',
    unsafe_allow_html=True,
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Damage Detection")
    st.markdown(
        """
        Upload a photo of the damaged vehicle and our AI will:
        - Detect if damage is present
        - Classify the severity
        - Provide a confidence score

        **Technology:** Deep Learning (MobileNetV2)
        """
    )
    if st.button("Go to Damage Detection →", use_container_width=True):
        st.switch_page("pages/damage_detection.py")

with col2:
    st.markdown("### Claim Estimation")
    st.markdown(
        """
        Based on the damage assessment and vehicle details:
        - Get an estimated claim range
        - View factors affecting the estimate
        - Download assessment report

        **Technology:** Machine Learning (Gradient Boosting)
        """
    )
    if st.button("Go to Claim Estimation →", use_container_width=True):
        st.switch_page("pages/claim_estimation.py")

st.divider()

# How it works Guide
st.markdown("###How It Works:")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("#### Step 1")
    st.markdown("**Upload** a photo of the damaged vehicle")

with col2:
    st.markdown("#### Step 2")
    st.markdown("**AI analyzes** the image and detects damage severity")

with col3:
    st.markdown("#### Step 3")
    st.markdown("**Enter** basic vehicle details (brand, year, fuel type)")

with col4:
    st.markdown("#### Step 4")
    st.markdown("**Receive** an estimated claim amount range")

st.divider()
st.caption(
    "⚠️ Disclaimer: This is an AI-assisted estimate for demonstration purposes. "
    "Final claim amounts are subject to physical inspection and policy terms."
)
