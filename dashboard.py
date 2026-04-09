import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================================
# 1. PAGE SETUP
# ==========================================
st.set_page_config(page_title="AI Orange Shelf-Life Tracker", page_icon="🍊", layout="wide")

st.title("🍊 Master AI Shelf-Life Dashboard")
st.markdown("Displaying a random sample of 50 oranges from the production line, showing predictions from the Master AI Brain.")

# ==========================================
# 2. DATA LOADING & SHUFFLING
# ==========================================
# We use st.cache_data so we don't reload the massive CSV from disk every single time,
# but we will sample randomly from it in the next step.
@st.cache_data
def load_data():
    try:
        # Load the fully predicted dataset from Colab
        return pd.read_csv("fully_labeled_by_teacher.csv")
    except FileNotFoundError:
        st.error("⚠️ Could not find 'fully_labeled_by_teacher.csv'. Please ensure it is in the same folder.")
        st.stop()

df = load_data()

# Create a button to manually trigger a reshuffle
if st.button("🔄 Resample 50 Oranges"):
    st.rerun()

# Shuffle and pick exactly 50 rows randomly
sample_df = df.sample(n=50).reset_index(drop=True)

# Map the numerical Spoilage Phase (0-3) to human-readable labels
# Based on your data, Phase 3 has ~11 days left, so it is the freshest.
phase_map = {
    3: "🟢 Phase 3: Fresh", 
    2: "🟡 Phase 2: Maturing", 
    1: "🟠 Phase 1: Fermenting", 
    0: "🔴 Phase 0: Spoilt"
}
sample_df['Phase_Label'] = sample_df['Spoilage_Phase'].map(phase_map)

# ==========================================
# 3. TOP LEVEL METRICS
# ==========================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Batch Avg Shelf Life", f"{sample_df['Shelf_Life_Days'].mean():.1f} Days")
col2.metric("Batch Avg Vision", f"{sample_df['Vision_Quality'].mean():.1f}%")
col3.metric("Batch Avg Gas", f"{sample_df['Gas_Quality'].mean():.1f}%")

# Count how many critical (spoilt/fermenting) oranges are in this batch of 50
critical_count = len(sample_df[sample_df['Spoilage_Phase'] <= 1])
col4.metric("Critical Action Needed", f"{critical_count} Oranges", delta_color="inverse")

st.divider()

# ==========================================
# 4. VISUALIZATIONS
# ==========================================
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Shelf Life Distribution")
    # A histogram showing how many oranges fall into which day ranges
    fig1 = px.histogram(
        sample_df, 
        x="Shelf_Life_Days", 
        color="Phase_Label",
        title="Days Remaining by Spoilage Phase", 
        nbins=15,
        color_discrete_sequence=["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
    )
    st.plotly_chart(fig1, use_container_width=True)

with col_chart2:
    st.subheader("Sensor Correlation")
    # Scatter plot checking if Vision Quality and Gas Quality agree with each other
    fig2 = px.scatter(
        sample_df, 
        x="Vision_Quality", 
        y="Gas_Quality", 
        color="Phase_Label",
        size="Shelf_Life_Days", 
        hover_data=["Shelf_Life_Days", "Temperature", "Humidity"],
        title="Vision vs. Gas Analysis",
        color_discrete_sequence=["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ==========================================
# 5. DATA TABLE
# ==========================================
st.subheader("📋 Sampled Orange Batch Details")

# Define a function to color-code the Shelf Life column dynamically
def color_shelf_life(val):
    if val >= 10:
        return 'color: #2ecc71; font-weight: bold' # Green
    elif val >= 5:
        return 'color: #f1c40f; font-weight: bold' # Yellow
    else:
        return 'color: #e74c3c; font-weight: bold' # Red

# Select specific columns and apply formatting
display_columns = ['Phase_Label', 'Shelf_Life_Days', 'Vision_Quality', 'Gas_Quality', 'Area', 'Temperature', 'Humidity']
styled_df = sample_df[display_columns].style.map(color_shelf_life, subset=['Shelf_Life_Days']).format({
    'Vision_Quality': "{:.1f}%",
    'Gas_Quality': "{:.1f}%",
    'Shelf_Life_Days': "{:.1f} days",
    'Temperature': "{:.1f} °C",
    'Humidity': "{:.1f}%",
    'Area': "{:,.0f}"
})

# Render the interactive dataframe
st.dataframe(styled_df, use_container_width=True, height=400)