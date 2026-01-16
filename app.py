import streamlit as st
import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Inventory Intelligence Pro", layout="wide", page_icon="🛡️")

# --- DOMAIN KNOWLEDGE CONFIGURATION (The "Anti-Trap" Engine) ---
CORE_NOUNS = [
    "TRANSMITTER", "VALVE", "FLANGE", "PIPE", "GASKET", "STUD", "ELBOW", "TEE", 
    "REDUCER", "BEARING", "SEAL", "GAUGE", "CABLE", "CONNECTOR", "BOLT", "NUT", 
    "WASHER", "UNION", "COUPLING", "HOSE", "PUMP", "MOTOR", "FILTER", "ADAPTOR", 
    "BRUSH", "TAPE", "SPANNER", "O-RING", "GLOVE", "CHALK", "BATTERY"
]

SPEC_TRAPS = {
    "Gender": ["MALE", "FEMALE"],
    "Connection": ["BW", "SW", "THD", "THREADED", "FLGD", "FLANGED", "SORF", "WNRF", "BLRF"],
    "Rating": ["150#", "300#", "600#", "PN10", "PN16", "PN25", "PN40"],
    "Material": ["SS316", "SS304", "MS", "PVC", "UPVC", "CPVC", "GI", "CS", "BRASS"]
}

# --- CORE UTILITY FUNCTIONS ---
def get_tech_dna(text):
    text = str(text).upper()
    dna = {"numbers": set(re.findall(r'\d+(?:[./]\d+)?', text)), "attributes": {}}
    for cat, keywords in SPEC_TRAPS.items():
        found = [k for k in keywords if re.search(rf'\b{k}\b', text)]
        if found: dna["attributes"][cat] = set(found)
    return dna

def intelligent_noun_extractor(text):
    text = str(text).upper()
    for noun in CORE_NOUNS:
        if re.search(rf'\b{noun}\b', text): return noun
    words = text.split()
    fillers = ["SS", "GI", "MS", "PVC", "UPVC", "SIZE", "DIA", "INCH", "MM", "CPVC"]
    for word in words:
        clean = re.sub(r'[^A-Z]', '', word)
        if clean and clean not in fillers and len(clean) > 2: return clean
    return "GENERAL"

# --- MAIN AI PIPELINE ---
@st.cache_data
def execute_ai_audit(file_path):
    # 1. ETL & CLEANING
    df = pd.read_csv(file_path)
    df.columns = ['Index', 'Item_No', 'Description', 'UoM']
    df['Clean_Desc'] = df['Description'].str.upper().str.replace('"', '', regex=False).str.strip()
    
    # 2. INTELLIGENT CATEGORIZATION (NMF Topic Modeling + Heuristics)
    tfidf = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1,2))
    tfidf_matrix = tfidf.fit_transform(df['Clean_Desc'])
    nmf = NMF(n_components=12, random_state=42)
    nmf_features = nmf.fit_transform(tfidf_matrix)
    
    # Mapping Topics
    feature_names = tfidf.get_feature_names_out()
    topic_labels = {i: " ".join([feature_names[ind] for ind in nmf.components_[i].argsort()[-2:][::-1]]).upper() for i in range(12)}
    
    df['Extracted_Noun'] = df['Clean_Desc'].apply(intelligent_noun_extractor)
    df['AI_Topic'] = nmf_features.argmax(axis=1).map(topic_labels)
    df['Category'] = df.apply(lambda r: r['AI_Topic'] if r['Extracted_Noun'] in r['AI_Topic'] else f"{r['Extracted_Noun']} ({r['AI_Topic']})", axis=1)
    
    # 3. DATA CLUSTERING & CONFIDENCE
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    df['Cluster_ID'] = kmeans.fit_predict(tfidf_matrix)
    dists = kmeans.transform(tfidf_matrix)
    df['Confidence'] = (1 - (np.min(dists, axis=1) / np.max(dists))).round(4)

    # 4. ANOMALY DETECTION
    df['Complexity'] = df['Clean_Desc'].apply(len)
    iso = IsolationForest(contamination=0.04, random_state=42)
    df['Anomaly_Flag'] = iso.fit_predict(df[['Complexity', 'Cluster_ID']])

    # 5. DUPLICATE & FUZZY LOGIC (The Trap Solver)
    exact_dups = df[df.duplicated(subset=['Clean_Desc'], keep=False)]
    df['Tech_DNA'] = df['Clean_Desc'].apply(get_tech_dna)
    
    fuzzy_results = []
    recs = df.to_dict('records')
    for i in range(len(recs)):
        for j in range(i + 1, min(i + 150, len(recs))):
            r1, r2 = recs[i], recs[j]
            sim = SequenceMatcher(None, r1['Clean_Desc'], r2['Clean_Desc']).ratio()
            if sim > 0.85:
                dna1, dna2 = r1['Tech_DNA'], r2['Tech_DNA']
                conflict = dna1['numbers'] != dna2['numbers']
                for cat in SPEC_TRAPS.keys():
                    if cat in dna1['attributes'] and cat in dna2['attributes']:
                        if dna1['attributes'][cat] != dna2['attributes'][cat]: conflict = True; break
                
                fuzzy_results.append({
                    'Item A': r1['Item_No'], 'Item B': r2['Item_No'],
                    'Desc A': r1['Clean_Desc'], 'Desc B': r2['Clean_Desc'],
                    'Similarity': f"{sim:.1%}", 'Status': "🛠️ Variant" if conflict else "🚨 Duplicate"
                })

    return df, exact_dups, pd.DataFrame(fuzzy_results)

# --- APP UI ---
st.title("🛡️ Enterprise AI Data Auditor")
st.markdown("---")

try:
    df, exact_dups, fuzzy_df = execute_ai_audit('raw_data.csv')
except:
    st.error("Upload 'raw_data.csv' to begin.")
    st.stop()

# 7 SEPARATE TABS PER ASSIGNMENT REQUIREMENTS
tabs = st.tabs([
    "📍 1. Categorization", "🎯 2. Clustering", "🚨 3. Anomaly Detection", 
    "👯 4. Exact Duplicates", "⚡ 5. Fuzzy Matches", "🧠 6. AI Methodology", "📈 7. Business Insights"
])

with tabs[0]:
    st.header("Product Categorization & Classification")
    st.dataframe(df[['Item_No', 'Clean_Desc', 'Category', 'Confidence']].sort_values('Confidence', ascending=False), use_container_width=True)

with tabs[1]:
    st.header("Data Clustering with Confidence")
    
    fig2 = px.scatter(df, x='Cluster_ID', y='Confidence', color='Category', size='Confidence', hover_data=['Clean_Desc'])
    st.plotly_chart(fig2, use_container_width=True)

with tabs[2]:
    st.header("Anomaly Identification")
    
    anomalies = df[df['Anomaly_Flag'] == -1]
    st.warning(f"Detected {len(anomalies)} statistical outliers.")
    st.dataframe(anomalies[['Item_No', 'Description', 'Category']])

with tabs[3]:
    st.header("Exact Duplicate Detection")
    if not exact_dups.empty: st.error(f"Found {len(exact_dups)} exact duplicates."); st.dataframe(exact_dups[['Item_No', 'Description']])
    else: st.success("No exact duplicates detected.")

with tabs[4]:
    st.header("Fuzzy Duplicate Identification")
    
    st.info("Differentiation Logic: >85% similarity but conflicting Technical DNA (e.g. Male vs Female) = Variant.")
    st.dataframe(fuzzy_df, use_container_width=True)

with tabs[5]:
    st.header("AI & NLP Methodology")
    st.markdown("""
    - **Categorization:** Hybrid NMF Topic Modeling + Heuristic Noun Anchoring.
    - **Clustering:** K-Means with Euclidean-based Confidence Scoring.
    - **Anomaly Detection:** Isolation Forest analyzing description complexity.
    - **Fuzzy Matching:** Levenshtein Sequence Matching with Spec-Aware conflict overrides.
    """)

with tabs[6]:
    st.header("Business Insights & Reporting")
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(px.pie(df, names='Extracted_Noun', title="Inventory by Component Type"), use_container_width=True)
    with c2: 
        health = (len(df[df['Anomaly_Flag'] == 1]) / len(df)) * 100
        st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=health, title={'text': "Data Accuracy %"})), use_container_width=True)
