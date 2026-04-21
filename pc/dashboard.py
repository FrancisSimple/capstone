import streamlit as st
import pandas as pd
import plotly.express as px
import os
import time
import requests

st.set_page_config(page_title="Industrial Orange Sorter", layout="wide", page_icon="🍊")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "live_telemetry.csv")
GAS_FILE = os.path.join(BASE_DIR, "data", "gas_telemetry.csv")
GAS_RAW_FILE = os.path.join(BASE_DIR, "data", "gas_raw.csv")
VISION_RAW_FILE = os.path.join(BASE_DIR, "data", "vision_raw.csv")

# Pi Bridge Configuration
PI_IP = "10.73.56.103" # CHANGE THIS to your Pi's IP
PI_BRIDGE_URL = f"http://{PI_IP}:8001/trigger"

st.title("🍊 Orange Sorter Pro | Master AI Dashboard")

# --- DATA LOADERS ---
def load_vision_data():
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame(columns=["Timestamp", "Fruit_ID", "Quality", "Status", "Area", "Gas_Env_Score", "Days_Left", "Life_Status"])
    try:
        df = pd.read_csv(DATA_FILE)
        return df
    except:
        return pd.DataFrame(columns=["Timestamp", "Fruit_ID", "Quality", "Status", "Area", "Gas_Env_Score", "Days_Left", "Life_Status"])

def load_vision_raw_data():
    cols = ["Timestamp", "Fruit_ID", "Quality", "Status", "Area"]
    if not os.path.exists(VISION_RAW_FILE):
        return pd.DataFrame(columns=cols)
    try:
        # Peak at the first line to see if it's a header
        df_peak = pd.read_csv(VISION_RAW_FILE, nrows=0)
        # If the first column name contains a colon (like a timestamp 12:00:00), it's data, not a header
        if len(df_peak.columns) > 0 and ":" in str(df_peak.columns[0]):
            return pd.read_csv(VISION_RAW_FILE, names=cols, header=None)
        return pd.read_csv(VISION_RAW_FILE)
    except:
        return pd.DataFrame(columns=cols)

def load_gas_raw_data():
    cols = ["Timestamp", "Gas_Quality", "Temp", "Hum"]
    if not os.path.exists(GAS_RAW_FILE):
        return pd.DataFrame(columns=cols)
    try:
        df_peak = pd.read_csv(GAS_RAW_FILE, nrows=0)
        if len(df_peak.columns) > 0 and ":" in str(df_peak.columns[0]):
            return pd.read_csv(GAS_RAW_FILE, names=cols, header=None)
        return pd.read_csv(GAS_RAW_FILE)
    except:
        return pd.DataFrame(columns=cols)

def load_gas_data():
    cols = ["Timestamp", "Gas_Quality", "Temp", "Hum"]
    if not os.path.exists(GAS_FILE):
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(GAS_FILE)
        return df
    except:
        return pd.DataFrame(columns=cols)

# --- CONTROL SIDEBAR ---
with st.sidebar:
    st.header("⚙️ System Control")
    
    # Get current status
    try:
        resp = requests.get(f"http://localhost:8000/status", timeout=0.5).json()
        v_active = resp["vision_active"]
        g_active = resp["gas_active"]
    except:
        v_active = False
        g_active = False
    
    # Configuration
    st.subheader("🌐 Network Config")
    PI_IP = st.text_input("Raspberry Pi IP", value=PI_IP)
    PI_BRIDGE_URL = f"http://{PI_IP}:8001"
    RECEIVER_URL = "http://localhost:8000"

    with st.expander("🔍 Network Diagnostics (Check here if Timeout occurs)", expanded=True):
        st.caption(f"Bridge Target: {PI_BRIDGE_URL}")
        st.caption(f"Local Receiver: {RECEIVER_URL}")
        
        try:
            r_ping = requests.get(f"{PI_BRIDGE_URL}/ping", timeout=3)
            st.success(f"✅ Pi Bridge: Reachable ({r_ping.json().get('message')})")
        except Exception as e:
            st.error("❌ Pi Bridge: UNREACHABLE")
            st.warning("1. Check Pi IP Address above\n2. Verify bridge is running on Pi\n3. Ensure PC & Pi are on the same WiFi")
            
        try:
            requests.get(f"{RECEIVER_URL}/ping", timeout=1)
            st.success("✅ Local Receiver: Reachable")
        except:
            st.error("❌ Local Receiver: UNREACHABLE")

    new_v = st.toggle("👁️ Activate Vision Scanner", value=v_active)
    if new_v != v_active:
        try:
            requests.post(f"http://localhost:8000/toggle/vision/{str(new_v).lower()}", timeout=1)
            st.rerun()
        except:
            st.error("Failed to connect to Receiver.")

    new_g = st.toggle("🌬️ Activate Gas Sensor", value=g_active)
    if new_g != g_active:
        try:
            requests.post(f"http://localhost:8000/toggle/gas/{str(new_g).lower()}", timeout=1)
            st.rerun()
        except:
            st.error("Failed to connect to Receiver.")

    st.divider()
    auto_refresh = st.toggle("🔄 Live Monitoring (Auto-Refresh)", value=True, help="Turn off to stop the dashboard from reloading every second.")

    st.divider()
    st.subheader("On-Demand Intelligence")
    
    # Calculate next index to match
    df_results = load_vision_data()
    next_index = len(df_results)
    
    # Try to peek at vision_raw to show the Target ID
    df_raw_v = load_vision_raw_data()
    target_id = "Unknown"
    if next_index < len(df_raw_v):
        target_id = f"#{df_raw_v.iloc[next_index]['Fruit_ID']}"
    
    st.info(f"🎯 Target: {target_id} (Index {next_index})")

    if st.button("🚀 CAPTURE & ANALYZE", use_container_width=True, type="primary"):
        with st.status(f"Processing Fruit {target_id}...", expanded=True) as p_status:
            st.write(f"1. Triggering Pi for Index {next_index}...")
            try:
                # Call Pi Bridge with Index - Increased timeout to 30s
                resp = requests.post(f"{PI_BRIDGE_URL}/trigger?index={next_index}", timeout=30)
                if resp.status_code == 200:
                    result = resp.json()
                    
                    # 2. Check for Logic Errors / Sync Errors from Pi
                    if "error" in result:
                        st.error(f"❌ Pi Bridge Error: {result['error']}")
                        p_status.update(label="Inference Logic Error", state="error")
                        st.stop()

                    st.write("2. Edge Inference Complete.")
                    # Robust key access
                    display_status = result.get('Life_Status', result.get('Status', 'Unknown'))
                    st.write(f"   -> Result: {display_status} ({result.get('Days_Left', 0)} days)")
                    
                    st.write("3. Saving result to PC database...")
                    # Save to PC Receiver for persistence
                    try:
                        record_resp = requests.post("http://localhost:8000/record_final", json=result, timeout=5)
                        if record_resp.status_code == 200:
                            p_status.update(label="Analysis Success!", state="complete", expanded=False)
                            st.success(f"Fruit #{result['Fruit_ID']} Processed & Recorded!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"PC Receiver Error: {record_resp.text}")
                    except Exception as pc_err:
                        st.error(f"Local PC Error (Saving failed): {pc_err}")
                else:
                    st.error(f"⚠️ Pi Error ({resp.status_code}): {resp.text}")
                    p_status.update(label="Inference Failed", state="error")
            except requests.exceptions.Timeout:
                st.error("⌛ Request Timed Out. Is the Pi Bridge running?")
                p_status.update(label="Timeout Error", state="error")
            except Exception as e:
                st.error(f"❌ Connection Error: {e}")
                p_status.update(label="Connection Error", state="error")

    st.divider()
    st.subheader("Master AI Actions")
    if st.button("[AI] Run Master AI Prediction", use_container_width=True):
        with st.spinner("Analyzing synced logs..."):
            analyze_resp = requests.post("http://localhost:8000/analyze").json()
            if "success" in analyze_resp["message"]:
                st.success(f"Processed {analyze_resp['items_processed']} items!")
                time.sleep(1)
                st.rerun()
            else:
                st.error(analyze_resp["message"])

tab1, tab2 = st.tabs(["Vision and Shelf-Life", "Environment Gas"])

# ==========================================
# TAB 1: VISION & SHELF-LIFE (The "Master" View)
# ==========================================
with tab1:
    df_v = load_vision_data()
    
    # Header Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Scanned", len(df_v))
    
    avg_life = df_v["Days_Left"].mean() if len(df_v) > 0 else 0
    col2.metric("Avg Shelf Life", f"{avg_life:.1f} Days")
    
    reject_count = len(df_v[df_v["Status"] == "REJECT"])
    col3.metric("Critical Rejects", reject_count, delta_color="inverse")
    
    excellent_count = len(df_v[df_v["Life_Status"] == "EXCELLENT"])
    col4.metric("Premium Grade (%)", f"{(excellent_count/len(df_v)*100):.0f}%" if len(df_v) > 0 else "0%")

    st.divider()

    if len(df_v) > 0:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("Predictive Analytics")
            fig = px.scatter(df_v, x="Quality", y="Days_Left", color="Life_Status",
                           size="Area", hover_name="Fruit_ID",
                           color_discrete_map={"EXCELLENT": "#2ecc71", "CONSUME SOON": "#f1c40f", "DISCARD": "#e74c3c"},
                           title="Shelf Life Prediction vs Vision Quality")
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.subheader("Final Grade tracking")
            # Style the dataframe
            display_df = df_v[['Fruit_ID', 'Days_Left', 'Life_Status', 'Quality']].iloc[::-1].head(15)
            st.table(display_df)
    else:
        st.info("🕒 Waiting for Master AI analysis. (Click 'Run Master AI Prediction' in sidebar once data is collected)")

    st.divider()
    
    # --- SYNC COUNTERS ---
    df_rv = load_vision_raw_data()
    df_rg = load_gas_raw_data()
    v_count = len(df_rv)
    g_count = len(df_rg)
    
    c1, c2 = st.columns(2)
    c1.metric("Fruits Scanned (Vision)", v_count)
    c2.metric("Snapshots Captured (Gas)", g_count, delta=g_count - v_count, delta_color="inverse" if g_count != v_count else "normal")
    
    if v_count != g_count:
        st.warning(f"⚠️ Sync Mismatch: You have {v_count} vision records but {g_count} gas snapshots. They must match for analysis!")

    st.subheader("📑 Live Session Scans (Raw Data)")
    if v_count > 0:
        st.dataframe(df_rv.iloc[::-1], use_container_width=True)
    else:
        st.caption("No scans recorded in this session yet.")

# ==========================================
# TAB 2: ENVIRONMENT GAS
# ==========================================
with tab2:
    df_g = load_gas_data()
    
    if len(df_g) > 0:
        latest = df_g.iloc[-1]
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Current Gas Health", f"{latest['Gas_Quality']:.1f}%")
        m2.metric("Storage Temperature", f"{latest['Temp']}°C")
        m3.metric("Storage Humidity", f"{latest['Hum']}%")
        
        st.divider()
        
        g_left, g_right = st.columns(2)
        with g_left:
            st.subheader("Environmental Trends")
            df_plot = df_g.tail(50)
            fig_health = px.line(df_plot, x="Timestamp", y="Gas_Quality", title="Gas Quality Score (%)")
            st.plotly_chart(fig_health, use_container_width=True)
        with g_right:
            st.subheader("Storage Conditions")
            df_env = df_g.tail(50).melt(id_vars=["Timestamp"], value_vars=["Temp", "Hum"])
            fig_env = px.line(df_env, x="Timestamp", y="value", color="variable", title="Temp & Humidity")
            st.plotly_chart(fig_env, use_container_width=True)
    else:
        st.info("Waiting for MQTT Gas sensor packets...")

# Refresh logic
if auto_refresh:
    time.sleep(1.5)
    st.rerun()
