"""
Page 2: Insurance Claim Estimator
Takes damage severity + vehicle details → estimates claim range
"""
import streamlit as st
import numpy as np
import json
import os
import joblib
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Claim Estimation", page_icon="💰", layout="wide")
st.markdown("## Insurance Claim Estimation")
st.markdown("Enter vehicle details to get an AI-powered claim estimate.")
st.divider()

# Load Models & Encoders
@st.cache_resource
def load_claim_model():
    """Load the trained claim estimation model and encoders."""
    model_path = os.path.join("models", "claim_model.pkl")
    encoders_path = os.path.join("models", "label_encoders.pkl")
    features_path = os.path.join("models", "model_features.json")

    if not os.path.exists(model_path):
        return None, None, None

    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path) if os.path.exists(encoders_path) else {}

    with open(features_path, "r") as f:
        metadata = json.load(f)

    return model, encoders, metadata


model, encoders, metadata = load_claim_model()


# Get damage information from sesstion state
damage_severity = st.session_state.get("damage_severity", None)
damage_confidence = st.session_state.get("damage_confidence", None)

if damage_severity and damage_severity != "None":
    st.success(
        f" **Damage Analysis Imported:** Severity = **{damage_severity}** "
        f"(Confidence: {damage_confidence:.1f}%)"
    )
else:
    st.warning(
        "⚠️ No damage analysis found. You can either:\n"
        "- Go to **Damage Detection** page first, or\n"
        "- Manually select severity below"
    )

st.divider()

# Vehicle Detilas Collection
st.markdown("### Vehicle & Incident Details")

col1, col2 = st.columns(2)

with col1:
    # Vehicle Brand
    vehicle_brands = [
        "Maruti Suzuki", "Hyundai", "Tata", "Honda", "Toyota",
        "Mahindra", "Kia", "Ford", "Volkswagen", "BMW",
        "Mercedes-Benz", "Audi", "Chevrolet", "Nissan", "Renault",
    ]
    selected_brand = st.selectbox("🏭 Vehicle Brand", vehicle_brands)

    # Model Year
    current_year = datetime.now().year
    selected_year = st.selectbox(
        "Model Year",
        list(range(current_year, current_year - 20, -1)),
    )

    # Fuel Type
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric", "Hybrid"])

with col2:
    # Incident Severity (auto-filled or manual)
    severity_options = ["Minor", "Moderate", "Severe"]
    default_idx = (
        severity_options.index(damage_severity)
        if damage_severity in severity_options
        else 1
    )
    selected_severity = st.selectbox(
        "Damage Severity",
        severity_options,
        index=default_idx,
        help="Auto-filled from damage detection if available",
    )

    # Policy Type
    policy_type = st.selectbox(
        "Policy Type",
        ["Comprehensive", "Third-Party Only"],
    )

    # Incident Type
    incident_type = st.selectbox(
        "Incident Type",
        ["Collision", "Scratches/Dents", "Theft Recovery", "Natural Disaster", "Other"],
    )

st.divider()

# Claim Estimator
if st.button("🔍 Estimate Claim Amount", type="primary", use_container_width=True):

    with st.spinner("Calculating estimate..."):

        # Brand tier mapping
        premium_brands = ["BMW", "Mercedes-Benz", "Audi"]
        mid_brands = ["Honda", "Toyota", "Kia", "Volkswagen", "Hyundai"]
        budget_brands = ["Maruti Suzuki", "Tata", "Mahindra", "Renault", "Ford", "Chevrolet", "Nissan"]

        if selected_brand in premium_brands:
            brand_multiplier = 2.5
            brand_tier = "Premium"
        elif selected_brand in mid_brands:
            brand_multiplier = 1.5
            brand_tier = "Mid-Range"
        else:
            brand_multiplier = 1.0
            brand_tier = "Budget"

        # Vehicle age and depreciation
        vehicle_age = current_year - selected_year
        depreciation = min(vehicle_age * 8, 70)  # Max 70% depreciation

        # Fuel type multiplier
        fuel_multipliers = {"Petrol": 1.0, "Diesel": 1.1, "Electric": 1.8, "Hybrid": 1.5}
        fuel_mult = fuel_multipliers[fuel_type]

        # Base claim by severity
        base_claims = {"Minor": 15000, "Moderate": 55000, "Severe": 150000}
        base = base_claims[selected_severity]

        # Try to use the ML model if available
        if model is not None and metadata is not None:
            try:
                # Prepare input for ML model
                input_data = {}
                feature_cols = metadata["feature_columns"]

                for col in feature_cols:
                    if "make" in col.lower() or "brand" in col.lower():
                        if col in encoders:
                            try:
                                input_data[col] = encoders[col].transform([selected_brand])[0]
                            except ValueError:
                                input_data[col] = 0
                        else:
                            input_data[col] = 0
                    elif "year" in col.lower():
                        input_data[col] = selected_year
                    elif "severity" in col.lower():
                        if col in encoders:
                            severity_map = {
                                "Minor": "Minor Damage",
                                "Moderate": "Major Damage",
                                "Severe": "Total Loss",
                            }
                            try:
                                mapped = severity_map.get(selected_severity, selected_severity)
                                input_data[col] = encoders[col].transform([mapped])[0]
                            except ValueError:
                                input_data[col] = encoders[col].transform(
                                    [encoders[col].classes_[0]]
                                )[0]
                        else:
                            input_data[col] = severity_options.index(selected_severity)
                    elif "incident_type" in col.lower() or "collision" in col.lower():
                        if col in encoders:
                            try:
                                input_data[col] = encoders[col].transform([incident_type])[0]
                            except ValueError:
                                input_data[col] = 0
                        else:
                            input_data[col] = 0
                    else:
                        input_data[col] = 0

                input_df = pd.DataFrame([input_data])
                ml_prediction = model.predict(input_df)[0]
                estimated_claim = max(ml_prediction, 5000)  # Floor at 5000
                estimation_method = "ML Model"

            except Exception as e:
                # Fallback to rule-based
                estimated_claim = base * brand_multiplier * fuel_mult * (1 - depreciation / 100)
                estimation_method = "Rule-Based (ML model error)"
        else:
            # Rule-based estimation
            estimated_claim = base * brand_multiplier * fuel_mult * (1 - depreciation / 100)
            estimation_method = "Rule-Based"

        # Calculate range (±20%)
        claim_low = estimated_claim * 0.80
        claim_high = estimated_claim * 1.20

        # Policy adjustment
        if policy_type == "Third-Party Only":
            st.warning(
                "Third-Party Only policies typically do not cover own vehicle damage. "
                "The estimate below is for reference only."
            )

    # Display Results
    st.divider()
    st.markdown("### Claim Report")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Estimated Claim (Low)", f"₹{claim_low:,.0f}")
    with col2:
        st.metric("Estimated Claim (Mid)", f"₹{estimated_claim:,.0f}")
    with col3:
        st.metric("Estimated Claim (High)", f"₹{claim_high:,.0f}")

    st.divider()

    # Detailed breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Vehicle Summary:")
        st.markdown(f"""
        | Detail | Value |
        |--------|-------|
        | **Brand** | {selected_brand} ({brand_tier}) |
        | **Model Year** | {selected_year} |
        | **Vehicle Age** | {vehicle_age} years |
        | **Fuel Type** | {fuel_type} |
        | **Depreciation** | {depreciation}% |
        | **Policy Type** | {policy_type} |
        """)

    with col2:
        st.markdown("#### Damage Summary:")
        severity_icons = {"Minor": "🟡", "Moderate": "🟠", "Severe": "🔴"}
        st.markdown(f"""
        | Detail | Value |
        |--------|-------|
        | **Severity** | {severity_icons[selected_severity]} {selected_severity} |
        | **Incident Type** | {incident_type} |
        | **Estimation Method** | {estimation_method} |
        | **Brand Multiplier** | {brand_multiplier}x |
        | **Fuel Multiplier** | {fuel_mult}x |
        """)

    # Factors chart
    st.divider()
    st.markdown("#### Factors Affecting Your Claim:")

    factors = {
        "Base (Severity)": base,
        "Brand Adjustment": base * (brand_multiplier - 1),
        "Fuel Type Adjustment": base * brand_multiplier * (fuel_mult - 1),
        "Depreciation Reduction": -(base * brand_multiplier * fuel_mult * depreciation / 100),
    }

    import plotly.graph_objects as go

    fig = go.Figure(
        go.Waterfall(
            name="Claim Breakdown",
            orientation="v",
            measure=["absolute", "relative", "relative", "relative"],
            x=list(factors.keys()),
            y=list(factors.values()),
            textposition="outside",
            text=[f"₹{v:,.0f}" for v in factors.values()],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#2ecc71"}},
            decreasing={"marker": {"color": "#e74c3c"}},
            totals={"marker": {"color": "#3498db"}},
        )
    )

    fig.update_layout(
        title="Claim Amount Breakdown",
        showlegend=False,
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Disclaimer
    st.divider()
    st.caption(
        "⚠**Disclaimer:** This is an AI-assisted estimate for demonstration purposes. "
        "The actual claim amount will be determined after physical inspection by an authorized "
        "assessor and is subject to your policy terms, deductibles, and coverage limits."
    )