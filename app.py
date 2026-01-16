import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Inventory Intelligence Pro", layout="wide", page_icon="🛡️")

# --- DOMAIN KNOWLEDGE CONFIG ---
CORE_NOUNS = ["TRANSMITTER", "VALVE", "FLANGE", "PIPE", "GASKET", "STUD", "ELBOW", "TEE", "REDUCER", "BEARING", "SEAL", "GAUGE", "CABLE", "CONNECTOR", "BOLT", "NUT", "WASHER", "UNION", "COUPLING", "HOSE", "PUMP", "MOTOR", "FILTER", "ADAPTOR", "BRUSH", "TAPE", "SPANNER", "O-RING", "GLOVE", "CHALK", "BATTERY"]
SPEC_TRAPS = {
    "Gender": ["MALE", "FEMALE"],
    "Connection": ["BW", "SW", "THD", "THREADED", "FLGD", "FLANGED", "SORF", "WNRF", "BLRF"],
    "Rating": ["150#", "300#", "600#", "PN10", "PN16", "PN25", "PN40"],
    "Material": ["SS316", "SS304", "MS", "PVC", "UPVC", "CPVC", "GI", "CS", "BRASS"]
}

# --- LOGIC HELPERS ---
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

# --- REFINED DATA LOADING ---
def load_raw_data():
    # Priority: Check multiple possible names in the current directory
    possible_names = ['raw_data.csv', 'Demo - Raw data.xlsx - Sheet2.csv']
    for name in possible_names:
        if os.path.exists(name):
            try:
                return pd.read_csv(name, encoding='utf-8')
            except UnicodeDecodeError:
                return pd.read_csv(name, encoding='latin1')
    
    # Sidebar fallback
    uploaded = st.sidebar.file_uploader("Data file not found in repo. Upload 'raw_data.csv' here:", type=['csv'])
    if uploaded:
        return pd.read_csv(uploaded)
    return None

# --- MAIN AI PIPELINE ---
@st.cache_data
def execute_ai_audit(df):
    try:
        # 1. CLEANING
        df.columns = [c.strip() for c in df.columns]
        desc_col = next((c for c in df.columns if 'desc' in c.lower()), df.columns[2])
        id_col = next((c for c in df.columns if any(x in c.lower() for x in ['item', 'no', 'id'])), df.columns[1])
        df['Clean_Desc'] = df[desc_col].astype(str).str.upper().str.replace('"', '', regex=False).str.strip()
        
        # 2. INTELLIGENT CATEGORIZATION
        tfidf = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1,2))
        tfidf_matrix = tfidf.fit_transform(df['Clean_Desc'])
        nmf = NMF(n_components=10, random_state=42, init='nndsvd')
        nmf_features = nmf.fit_transform(tfidf_matrix)
        
        # Mapping Topic Names
        feature_names = tfidf.get_feature_names_out()
        topic_labels = {i: " ".join([feature_names[ind] for ind in nmf.components_[i].argsort()[-2:][::-1]]).upper() for i in range(10)}
        
        # FIX: Convert numpy array to Series before mapping
        topic_ids = nmf_features.argmax(axis=1)
        df['AI_Topic'] = pd.Series(topic_ids).map(topic_labels).values
        
        df['Extracted_Noun'] = df['Clean_Desc'].apply(intelligent_noun_extractor)
        df['Category'] = df.apply(lambda r: r['AI_Topic'] if r['Extracted_Noun'] in r['AI_Topic'] else f"{r['Extracted_Noun']} ({r['AI_Topic']})", axis=1)
        
        # 3. CLUSTERING & CONFIDENCE
        kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
        df['Cluster_ID'] = kmeans.fit_predict(tfidf_matrix)
        dists = kmeans.transform(tfidf_matrix)
        df['Confidence'] = (1 - (np.min(dists, axis=1) / np.max(dists))).round(4)

        # 4. ANOMALY DETECTION
        df['Complexity'] = df['Clean_Desc'].apply(len)
        iso = IsolationForest(contamination=0.04, random_state=42)
        df['Anomaly_Flag'] = iso.fit_predict(df[['Complexity', 'Cluster_ID']])

        # 5. SMART DUPLICATE LOGIC
        exact_dups = df[df.duplicated(subset=['Clean_Desc'], keep=False)]
        df['Tech_DNA'] = df['Clean_Desc'].apply(get_tech_dna)
        
        fuzzy_results = []
        recs = df.to_dict('records')
        for i in range(len(recs)):
            for j in range(i + 1, min(i + 100, len(recs))):
                r1, r2 = recs[i], recs[j]
                sim = SequenceMatcher(None, r1['Clean_Desc'], r2['Clean_Desc']).ratio()
                if sim > 0.85:
                    dna1, dna2 = r1['Tech_DNA'], r2['Tech_DNA']
                    conflict = dna1['numbers'] != dna2['numbers']
                    for cat in SPEC_TRAPS.keys():
                        if cat in dna1['attributes'] and cat in dna2['attributes']:
                            if dna1['attributes'][cat] != dna2['attributes'][cat]: conflict = True; break
                    fuzzy_results.append({
                        'Item A': r1[id_col], 'Item B': r2[id_col],
                        'Desc A': r1['Clean_Desc'], 'Desc B': r2['Clean_Desc'],
                        'Similarity': f"{sim:.1%}", 'Status': "🛠️ Variant" if conflict else "🚨 Duplicate"
                    })
        return df, exact_dups, pd.DataFrame(fuzzy_results), id_col, desc_col
    except Exception as e:
        st.error(f"Error in Processing: {str(e)}")
        return None, None, None, None, None

# --- UI FLOW ---
raw_df = load_raw_data()

if raw_df is not None:
    df, exact_dups, fuzzy_df, id_col, desc_col = execute_ai_audit(raw_df)
    
    if df is not None:
        tabs = st.tabs(["📍 Categorization", "🎯 Clustering", "🚨 Anomalies", "👯 Exact Duplicates", "⚡ Fuzzy Matches", "🧠 AI Info", "📈 Reports"])
        
        with tabs[0]: st.dataframe(df[[id_col, 'Clean_Desc', 'Category', 'Confidence']])
        with tabs[1]: st.plotly_chart(px.scatter(df, x='Cluster_ID', y='Confidence', color='Category', hover_data=['Clean_Desc']), use_container_width=True)
        with tabs[2]: st.warning(f"Found {len(df[df['Anomaly_Flag']==-1])} anomalies."); st.dataframe(df[df['Anomaly_Flag']==-1][[id_col, desc_col, 'Category']])
        with tabs[3]: st.dataframe(exact_dups[[id_col, desc_col]] if not exact_dups.empty else "No exact duplicates.")
        with tabs[4]: st.dataframe(fuzzy_df, use_container_width=True)
        with tabs[5]: st.markdown("**Models:** NMF (Topics), K-Means (Clusters), Isolation Forest (Anomalies), Levenshtein (Fuzzy)")
        with tabs[6]:
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.pie(df, names='Extracted_Noun', title="Inventory Breakdown"))
            c2.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=(len(df[df['Anomaly_Flag']==1])/len(df)*100), title={'text':"Data Health %"})))
else:
    st.info("👋 Waiting for Data. Please ensure 'raw_data.csv' is in your GitHub repository.")
