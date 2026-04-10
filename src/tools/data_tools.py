import pandas as pd
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.tools import tool
import sqlite3

import joblib
import os

# Define the model path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'failure_predictor.pkl')

@tool
def predict_failure(air_temp: float, process_temp: float, speed: float, torque: float, tool_wear: float):
    """
    Predicts the probability of a machine failure based on sensor inputs.
    Inputs: Air Temp (K), Process Temp (K), Speed (rpm), Torque (Nm), Tool Wear (min).
    """
    if not os.path.exists(MODEL_PATH):
        return "Failure prediction model not found. Please train the model first."
    
    # Load the model
    model = joblib.load(MODEL_PATH)
    
    # Format the data for the model
    features = [[air_temp, process_temp, speed, torque, tool_wear]]
    
    # Predict probability (returns [prob_no_failure, prob_failure])
    probability = model.predict_proba(features)[0][1]
    
    risk_level = "CRITICAL" if probability > 0.8 else "WARNING" if probability > 0.5 else "STABLE"
    
    return f"Failure Probability: {probability:.2%}. Risk Status: {risk_level}."

def get_failed_machines():
    """Returns a list of machines that have failed and the type of failure."""
    df = pd.read_csv('data/maintenance.csv')
    # Filter for failures
    failures = df[df['Machine failure'] == 1]
    
    # Identify which specific failure occurred
    # (TWF, HDF, PWF, OSF, RNF)
    failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    result = []
    
    for _, row in failures.iterrows():
        active_failures = [f for f in failure_types if row[f] == 1]
        result.append({
            "product_id": row['Product ID'],
            "failure_type": active_failures[0] if active_failures else "Unknown",
            "air_temp": row['Air temperature [K]']
        })
    return result

def get_commodity_price(commodity_name):
    """Gets the latest nominal price for a specific commodity (e.g., 'Aluminum')."""
    df = pd.read_csv('data/commodity.csv')
    # Filter by name and get the most recent entry
    target = df[df['commodity_name'].str.contains(commodity_name, case=False)]
    if not target.empty:
        latest = target.iloc[-1]
        return {
            "price": latest['price_nominal_usd'],
            "unit": latest['unit'],
            "date": latest['date']
        }
    return "Commodity not found."

def get_supplier_info():
    """Checks inventory and lead times from the industrial dataset."""
    df = pd.read_csv('data/industrial_cleaned.csv')
    # Let's look for suppliers with high reliability and low lead time
    best_suppliers = df[df['Reliability_Score'] > 0.8].sort_values(by='Lead_Time_Supplier')
    return best_suppliers[['Supplier_ID', 'Lead_Time_Supplier', 'Reliability_Score']].head(3).to_dict()

def query_dataframe(query, df_type="maintenance"):
    """Provides a summary of the requested dataframe for the AI to answer questions."""
    if df_type == "maintenance":
        df = pd.read_csv('data/maintenance.csv')
    elif df_type == "industrial":
        df = pd.read_csv('data/industrial_cleaned.csv')
    else:
        df = pd.read_csv('data/commodity.csv')
    
    # We send a summary and the first few rows so the AI understands the structure
    summary = df.describe().to_string()
    head = df.head(5).to_string()
    return f"Data Summary:\n{summary}\n\nRecent Samples:\n{head}"

def search_manual(query):
    """Searches the indexed PDF manual for technical repair steps."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="data/chroma_db", embedding_function=embeddings)
    
    # Retrieve the top 3 most relevant paragraphs
    results = db.similarity_search(query, k=3)
    
    context = "\n".join([res.page_content for res in results])
    return context



def calculate_risk_scores():
    """Predicts future failures by calculating proximity to known danger thresholds."""
    df = pd.read_csv('data/maintenance.csv')
    
    # Constants from the AI4I 2020 Documentation
    # We calculate how 'close' we are to a failure (1.0 = Failure Imminent)
    
    # 1. Heat Dissipation Risk
    temp_diff = df['Process temperature [K]'] - df['Air temperature [K]']
    df['heat_risk'] = np.where((temp_diff < 8.6) & (df['Rotational speed [rpm]'] < 1380), 1.0, 0.2)
    
    # 2. Power Failure Risk (Torque * Speed in rad/s)
    power = df['Torque [Nm]'] * (df['Rotational speed [rpm]'] * (2 * np.pi / 60))
    df['power_risk'] = np.where((power < 3500) | (power > 9000), 1.0, 0.1)
    
    # 3. Overstrain Risk (Tool Wear * Torque)
    overstrain = df['Tool wear [min]'] * df['Torque [Nm]']
    df['overstrain_risk'] = overstrain / 11000 # Normalized to the 'L' type threshold
    
    # Final Health Score (100 is perfect, 0 is failing)
    df['health_score'] = (1 - df[['heat_risk', 'power_risk', 'overstrain_risk']].max(axis=1)) * 100
    df['health_score'] = df['health_score'].clip(0, 100)
    
    return df[['Product ID', 'health_score', 'heat_risk', 'power_risk', 'overstrain_risk']].tail(10)

from langchain.tools import tool

@tool
def check_maintenance_sensors(product_id: str):
    """Use this tool to get real-time sensor data (temp, speed, torque) for a specific machine ID."""
    df = pd.read_csv('data/maintenance.csv')
    data = df[df['Product ID'] == product_id]
    return data.to_dict() if not data.empty else "Machine ID not found."

@tool
def consult_technical_manual(query: str):
    """Use this tool to search the WEG Motor Manual for repair steps, installation guides, or error code meanings."""
    return search_manual(query) # Calls your existing vector search

@tool
def check_market_prices(commodity: str):
    """Use this tool to see the latest global price for materials like Copper, Aluminum, or Steel."""
    return get_commodity_price(commodity)

@tool
def analyze_sensor_trends(product_id: str, sensor_name: str):
    """
    Calculates the average and max for a specific sensor (e.g., 'Torque [Nm]') 
    for a machine to see if current levels are abnormal.
    """ 
    df = pd.read_csv('data/maintenance.csv')
    machine_data = df[df['Product ID'] == product_id]
    
    if machine_data.empty:
        return "Machine ID not found."
    
    avg_val = df[sensor_name].mean()
    current_val = machine_data[sensor_name].iloc[-1]
    
    diff_pct = ((current_val - avg_val) / avg_val) * 100
    
    return f"Current {sensor_name}: {current_val}. Factory Average: {avg_val:.2f}. Deviation: {diff_pct:.2f}%"


@tool
def run_sql_query(query: str):
    """
    Use this to ask complex questions about the factory data. 
    Tables: 'maintenance' (sensors), 'commodities' (prices), 'logistics' (suppliers).
    Example: 'SELECT AVG(Torque) FROM maintenance WHERE Machine_failure = 1'
    """
    conn = sqlite3.connect('data/industrial_ai.db')
    try:
        result = pd.read_sql_query(query, conn)
        return result.to_string()
    except Exception as e:
        return f"Error in SQL query: {e}"
    finally:
        conn.close()

@tool
def get_market_news(query: str):
    """
    Search for real-time supply chain disruptions, strikes, or price forecasts.
    Use this to decide if we should buy parts now or wait.
    """
    from tools.search_tools import get_live_market_news # Import your existing Tavily logic
    return get_live_market_news(query)

@tool
def predict_machine_health(product_id: str):
    """Predicts the probability of a machine failure based on its current sensor readings."""
    # 1. Get current sensors from SQL
    # 2. Feed them into the .pkl model
    # 3. Return: "90% Probability of Failure"
    pass