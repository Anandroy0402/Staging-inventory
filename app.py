import streamlit as st
import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="Advanced AI Inventory Auditor", layout="wide", page_icon="🛡️")

# --- ADVANCED LOGIC: TRAP DETECTION ---
# Define keywords that denote a completely different SKU even if names match
SPEC_TRAPS = {
    "Gender": ["MALE", "FEMALE"],
    "Connection": ["BW", "SW", "THD", "THREADED", "FLGD", "FLANGED", "SORF", "WNRF", "BLRF"],
    "Type": ["ELBOW", "TEE", "REDUCER", "UNION"],
    "Rating": ["150#", "300#", "600#", "PN10", "PN16", "PN25", "PN40"],
    "Schedule": ["SCH 10", "SCH 40", "SCH 80", "SCH 160", "SDR-11", "SDR-II"]
}

def get_spec_fingerprint(text):
    """Extracts critical technical specs to distinguish variants."""
    text = str(text).upper()
    specs = {}
    
    # 1. Numeric Fingerprint (Size)
    specs['numbers'] = set(re.findall(r'\d+(?:[./]\d+)?', text))
    
    # 2. Mutually Exclusive Keyword Check
    # We find which specific keyword from our trap lists exists in the text
    specs['traps'] = []
    for category, keywords in SPEC_TRAPS.items():
        found = [k for k in keywords if re.search(rf'\b{k}\b', text)]
        if found:
            specs['traps'].append(set(found))
            
    return specs

@st.cache_data
def run_ai_pipeline(file_path):
    df = pd.read_csv(file_path)
    df.columns = ['Index', 'Item_No', 'Description', 'UoM']
    
    def clean(text):
        text = str(text).upper()
        text = re.sub(r'"+', '', text)
        return " ".join(text.split()).strip()

    df['Clean_Desc'] = df['Description'].apply(clean)
    # Generate the Technical Fingerprint
    df['Tech_DNA'] = df['Clean_Desc'].apply(get_spec_fingerprint)

    # NLP Clustering (Standard logic)
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['Clean_Desc'])
    model_kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    df['Cluster_ID'] = model_kmeans.fit_predict(tfidf_matrix)
    
    # Confidence Scoring
    distances = model_kmeans.transform(tfidf_matrix)
    df['Confidence_Score'] = (1 - (np.min(distances, axis=1) / np.max(distances))).round(4)
    
    terms = vectorizer.get_feature_names_out()
    cluster_labels = {i: " ".join([terms[ind] for ind in model_kmeans.cluster_centers_[i].argsort()[-2:]]).upper() for i in range(8)}
    df['AI_Category'] = df['Cluster_ID'].map(cluster_labels)

    # Anomaly Detection
    df['Char_Length'] = df['Clean_Desc'].apply(len)
    model_iso = IsolationForest(contamination=0.04, random_state=42)
    df['Anomaly_Flag'] = model_iso.fit_predict(df[['Char_Length', 'Cluster_ID']])
    
    # SMART DUPLICATE IDENTIFICATION (The Trap Solver)
    conflict_results = []
    recs = df.to_dict('records')
    for i in range(len(recs)):
        for j in range(i + 1, min(i + 150, len(recs))):
            r1, r2 = recs[i], recs[j]
            sim = SequenceMatcher(None, r1['Clean_Desc'], r2['Clean_Desc']).ratio()
            
            if sim > 0.85:
                # TRAP LOGIC 1: Numeric Mismatch (Size)
                num_mismatch = r1['Tech_DNA']['numbers'] != r2['Tech_DNA']['numbers']
                
                # TRAP LOGIC 2: Keyword Conflict (Gender/Ends/Rating)
                # Check if they have different keywords within the same category
                keyword_conflict = False
                for idx, trap_set_a in enumerate(r1['Tech_DNA']['traps']):
                    # Check if r2 has a DIFFERENT keyword from the same technical category
                    for trap_set_b in r2['Tech_DNA']['traps']:
                        # If the sets contain keywords from the same SPEC_TRAPS list but differ
                        if trap_set_a != trap_set_b:
                            # Verify if these are from the same category (e.g. one is MALE, one is FEMALE)
                            keyword_conflict = True
                            break
                
                status = "🛠️ Variant (Tech Specs Differ)" if (num_mismatch or keyword_conflict) else "🚨 Potential Duplicate"
                
                conflict_results.append({
                    'Item A': r1['Item_No'], 'Item B': r2['Item_No'],
                    'Desc A': r1['Clean_Desc'], 'Desc B': r2['Clean_Desc'],
                    'Similarity': f"{sim:.1%}", 'Status': status
                })
                
    return df, pd.DataFrame(conflict_results)

# --- APP FLOW ---
try:
    df, conflicts = run_ai_pipeline('raw_data.csv')
except:
    st.error("Ensure 'raw_data.csv' is in your GitHub repo.")
    st.stop()

st.title("🛡️ Spec-Aware AI Inventory Auditor")
st.markdown("---")

tabs = st.tabs(["📍 Categorization", "🚨 Anomalies", "👯 Conflict Manager (Traps)", "📈 Insights"])

with tabs[0]:
    st.header("AI Product Classification")
    st.dataframe(df[['Item_No', 'Description', 'AI_Category', 'Confidence_Score']], use_container_width=True)

with tabs[1]:
    st.header("Anomaly Detection")
    st.dataframe(df[df['Anomaly_Flag'] == -1][['Item_No', 'Description', 'AI_Category']], use_container_width=True)

with tabs[2]:
    st.header("The Conflict Manager")
    st.info("Advanced Logic: This tab identifies **Technical Variants** by looking for 'Mutually Exclusive' specifications (like Male vs Female).")
    
    if not conflicts.empty:
        # Highlighting logic for the UI
        def color_status(val):
            return 'color: #1f77b4' if 'Variant' in val else 'color: #d62728; font-weight: bold'
        
        st.dataframe(conflicts.style.applymap(color_status, subset=['Status']), use_container_width=True)
    else:
        st.success("No conflicts found.")

with tabs[3]:
    st.header("Inventory Health Report")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.pie(df, names='AI_Category', title="Inventory Cluster Split"), use_container_width=True)
    with c2:
        health_score = (len(df[df['Anomaly_Flag'] == 1]) / len(df)) * 100
        st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=health_score, title={'text': "Data Integrity %"})), use_container_width=True)
