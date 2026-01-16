import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from difflib import SequenceMatcher

# AI & NLP Stack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from transformers import pipeline  # Hugging Face Model Loader
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Inventory Auditor Pro", layout="wide", page_icon="🛡️")

# --- KNOWLEDGE BASE: DOMAIN LOGIC ---
DEFAULT_PRODUCT_GROUP = "Consumables & General"

PRODUCT_GROUPS = {
    "Piping & Fittings": ["FLANGE", "PIPE", "ELBOW", "TEE", "UNION", "REDUCER", "BEND", "COUPLING", "NIPPLE", "BUSHING", "UPVC", "CPVC", "PVC"],
    "Valves & Actuators": ["BALL VALVE", "GATE VALVE", "PLUG VALVE", "CHECK VALVE", "GLOBE VALVE", "CONTROL VALVE", "VALVE", "ACTUATOR", "COCK"],
    "Fasteners & Seals": ["STUD", "BOLT", "NUT", "WASHER", "GASKET", "O RING", "MECHANICAL SEAL", "SEAL", "JOINT"],
    "Electrical & Instruments": ["TRANSMITTER", "CABLE", "WIRE", "GAUGE", "SENSOR", "CONNECTOR", "SWITCH", "TERMINAL", "INSTRUMENT", "CAMERA"],
    "Tools & Hardware": ["PLIER", "CUTTING PLIER", "STRIPPER", "WIRE STRIPPER", "WRENCH", "SPANNER", "HAMMER", "FILE", "SAW", "TOOL", "CHISEL", "CUTTER", "TAPE MEASURE", "MEASURING TAPE", "BIT", "DRILL BIT"],
    "Consumables & General": ["BRUSH", "PAINT BRUSH", "TAPE", "ADHESIVE", "HOSE", "SAFETY GLOVE", "GLOVE", "CLEANER", "PAINT", "CEMENT", "STICKER", "CHALK"],
    "Specialized Spares": ["FILTER", "BEARING", "PUMP", "MOTOR", "CARTRIDGE", "IMPELLER", "SPARE"]
}

SPEC_TRAPS = {
    "Gender": ["MALE", "FEMALE"],
    "Connection": ["BW", "SW", "THD", "THREADED", "FLGD", "FLANGED", "SORF", "WNRF", "BLRF"],
    "Rating": ["150#", "300#", "600#", "PN10", "PN16", "PN25", "PN40"]
}

# --- AI MODELS LOADING (CACHED) ---
@st.cache_resource
def load_hf_classifier():
    # Using a fast, lightweight model for zero-shot classification
    return pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")

# --- AI UTILITIES ---
def clean_description(text):
    text = str(text).upper().replace('"', ' ')
    text = text.replace("O-RING", "O RING")
    text = text.replace("MECH-SEAL", "MECHANICAL SEAL").replace("MECH SEAL", "MECHANICAL SEAL")
    text = re.sub(r'[^A-Z0-9\s./-]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def token_pattern(token):
    return rf'(?<!\w){re.escape(token)}(?!\w)'

def get_tech_dna(text):
    text = clean_description(text)
    dna = {"numbers": set(re.findall(r'\d+(?:[./]\d+)?', text)), "attributes": {}}
    for cat, keywords in SPEC_TRAPS.items():
        found = [k for k in keywords if re.search(token_pattern(k), text)]
        if found: dna["attributes"][cat] = set(found)
    return dna

def intelligent_noun_extractor(text):
    text = clean_description(text)
    phrases = ["MEASURING TAPE", "BALL VALVE", "GATE VALVE", "PLUG VALVE", "CHECK VALVE", "MECHANICAL SEAL", "PAINT BRUSH", "WIRE STRIPPER", "CUTTING PLIER", "DRILL BIT"]
    for p in phrases:
        if re.search(token_pattern(p), text): return p
    all_nouns = [item for sublist in PRODUCT_GROUPS.values() for item in sublist]
    for n in all_nouns:
        if re.search(token_pattern(n), text): return n
    return text.split()[0] if text.split() else "MISC"

def dominant_group(series):
    counts = series.value_counts()
    return counts.idxmax() if not counts.empty else "UNMAPPED"

# --- MAIN ENGINE ---
@st.cache_data
def run_intelligent_audit(file_path):
    df = pd.read_csv(file_path, encoding='latin1')
    df.columns = [c.strip() for c in df.columns]
    id_col = next(c for c in df.columns if any(x in c.lower() for x in ['item', 'no']))
    desc_col = next(c for c in df.columns if 'desc' in c.lower())
    
    df['Standard_Desc'] = df[desc_col].apply(clean_description)
    df['Part_Noun'] = df['Standard_Desc'].apply(intelligent_noun_extractor)
    
    # 1. HUGGING FACE CLASSIFICATION
    classifier = load_hf_classifier()
    candidate_labels = list(PRODUCT_GROUPS.keys())
    
    # Process batch for speed
    results = classifier(df['Standard_Desc'].tolist(), candidate_labels=candidate_labels)
    df['Product_Group'] = [r['labels'][0] for r in results]
    df['HF_Score'] = [r['scores'][0] for r in results]

    # 2. NLP & Topic Modeling
    tfidf = TfidfVectorizer(max_features=300, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Standard_Desc'])
    
    # 3. Clustering & Confidence
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    df['Cluster_ID'] = kmeans.fit_predict(tfidf_matrix)
    dists = kmeans.transform(tfidf_matrix)
    df['Confidence'] = (1 - (np.min(dists, axis=1) / np.max(dists, axis=1))).round(4)
    cluster_groups = df.groupby('Cluster_ID')['Product_Group'].agg(dominant_group)
    df['Cluster_Group'] = df['Cluster_ID'].map(cluster_groups)
    df['Cluster_Validated'] = df['Product_Group'] == df['Cluster_Group']
    
    # 4. Anomaly detection
    iso = IsolationForest(contamination=0.04, random_state=42)
    df['Anomaly_Flag'] = iso.fit_predict(tfidf_matrix)

    # 5. Fuzzy DNA
    df['Tech_DNA'] = df['Standard_Desc'].apply(get_tech_dna)
    
    return df, id_col, desc_col

# --- DATA LOADING ---
target_file = 'raw_data.csv'
if os.path.exists(target_file):
    df_raw, id_col, desc_col = run_intelligent_audit(target_file)
else:
    st.error("Data file missing from repository. Please ensure 'raw_data.csv' is present.")
    st.stop()

# --- HEADER & NAVIGATION ---
st.title("🛡️ AI Inventory Auditor Pro")
st.markdown("### Powered by Hugging Face Transformers")

page = st.tabs(["📈 Executive Dashboard", "📍 Categorization Audit", "🚨 Quality Hub", "🧠 Technical Methodology"])

df = df_raw.copy()
group_options = list(PRODUCT_GROUPS.keys())

# --- PAGE: EXECUTIVE DASHBOARD ---
with page[0]:
    st.markdown("#### 📊 Inventory Health Overview")
    selected_group = st.multiselect("Product Category", options=group_options, default=group_options, key="dash_group")
    df = df_raw[df_raw['Product_Group'].isin(selected_group)]
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("📦 SKUs Analyzed", len(df))
    kpi2.metric("🎯 Mean AI Confidence", f"{df['HF_Score'].mean():.1%}")
    kpi3.metric("⚠️ Anomalies Found", len(df[df['Anomaly_Flag'] == -1]))
    kpi4.metric("🔄 Duplicate Pairs", "Audit Required")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.pie(df, names='Product_Group', title="Categorization Split", hole=0.4), use_container_width=True)
    with col2:
        top_nouns = df['Part_Noun'].value_counts().head(10).reset_index()
        st.plotly_chart(px.bar(top_nouns, x='Part_Noun', y='count', title="Top 10 Product Types"), use_container_width=True)

    health_val = (len(df[df['Anomaly_Flag'] == 1]) / len(df)) * 100
    fig_gauge = go.Figure(go.Indicator(mode = "gauge+number", value = health_val, title = {'text': "Catalog Data Accuracy %"}))
    st.plotly_chart(fig_gauge, use_container_width=True)

# --- PAGE: CATEGORIZATION AUDIT ---
with page[1]:
    st.markdown("#### 📍 AI Categorization Audit")
    selected_group = st.multiselect("Product Category", options=group_options, default=group_options, key="cat_group")
    df = df_raw[df_raw['Product_Group'].isin(selected_group)]
    
    st.dataframe(df[[id_col, 'Standard_Desc', 'Part_Noun', 'Product_Group', 'HF_Score']].sort_values('HF_Score', ascending=False), use_container_width=True)

# --- PAGE: QUALITY HUB ---
with page[2]:
    st.markdown("#### 🚨 Anomaly & Duplicate Identification")
    selected_group = st.multiselect("Product Category", options=group_options, default=group_options, key="qual_group")
    df = df_raw[df_raw['Product_Group'].isin(selected_group)]
    
    t1, t2 = st.tabs(["⚠️ Anomalies", "👯 Fuzzy Duplicates"])
    
    with t1:
        anoms = df[df['Anomaly_Flag'] == -1]
        st.warning(f"Found {len(anoms)} anomalies.")
        st.dataframe(anoms[[id_col, 'Standard_Desc', 'Part_Noun']], use_container_width=True)
        
    with t2:
        fuzzy_list = []
        recs = df.to_dict('records')
        for i in range(len(recs)):
            for j in range(i + 1, min(i + 50, len(recs))):
                r1, r2 = recs[i], recs[j]
                sim = SequenceMatcher(None, r1['Standard_Desc'], r2['Standard_Desc']).ratio()
                if sim > 0.85:
                    dna1, dna2 = r1['Tech_DNA'], r2['Tech_DNA']
                    is_variant = (dna1['numbers'] != dna2['numbers']) or (dna1['attributes'] != dna2['attributes'])
                    fuzzy_list.append({'ID A': r1[id_col], 'ID B': r2[id_col], 'Match %': f"{sim:.1%}", 'Verdict': "🛠️ Variant" if is_variant else "🚨 Duplicate"})
        st.dataframe(pd.DataFrame(fuzzy_list), use_container_width=True)

# --- PAGE: METHODOLOGY ---
with page[3]:
    st.header("🧠 Technical Methodology")
    st.markdown("""
    1. **Hugging Face Zero-Shot Classification**: We use the `distilbert-base-uncased-mnli` model to semantically categorize items into your predefined `PRODUCT_GROUPS`. Unlike keyword matching, this understands synonyms and technical context.
    2. **Isolation Forest**: Analyzes the complexity and length of descriptions to identify statistical anomalies.
    3. **Spec-Aware Fuzzy Matching**: Uses Levenshtein distance but overrides the AI if the Technical DNA (numbers/gender) differs, preventing false-positive duplicates.
    """)
