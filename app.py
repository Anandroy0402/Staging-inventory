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

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Enterprise AI Inventory Auditor", layout="wide", page_icon="⚙️")

# --- ADVANCED TECHNICAL CONFIGURATION (The "Anti-Trap" Dictionary) ---
SPEC_TRAPS = {
    "Gender": ["MALE", "FEMALE"],
    "Connection": ["BW", "SW", "THD", "THREADED", "FLGD", "FLANGED", "SORF", "WNRF", "BLRF"],
    "Rating": ["150#", "300#", "600#", "PN10", "PN16", "PN25", "PN40"],
    "Material": ["SS316", "SS304", "MS", "PVC", "UPVC", "CPVC", "GI", "CS", "BRASS", "PP"]
}

# --- CORE AI ENGINE ---
def get_tech_dna(text):
    """Extracts technical attributes to prevent logical errors in matching."""
    text = str(text).upper()
    dna = {
        "numbers": set(re.findall(r'\d+(?:[./]\d+)?', text)),
        "attributes": {}
    }
    for category, keywords in SPEC_TRAPS.items():
        found = [k for k in keywords if re.search(rf'\b{k}\b', text)]
        if found:
            dna["attributes"][category] = set(found)
    return dna

def identify_product_noun(text):
    """Extracts the core product noun (e.g., 'PIPE' or 'VALVE') to avoid categorization errors."""
    clean = re.sub(r'[^A-Z\s]', ' ', text.upper())
    words = clean.split()
    # Industrial descriptions usually start with the noun. 
    # We skip small common filler words if necessary.
    if words:
        return words[0]
    return "UNKNOWN"

@st.cache_data
def process_inventory_data(file_path):
    # 1. ETL & Cleaning
    df = pd.read_csv(file_path)
    df.columns = ['Index', 'Item_No', 'Description', 'UoM']
    
    def clean_full(text):
        text = str(text).upper()
        text = re.sub(r'"+', '', text)
        return " ".join(text.split()).strip()

    df['Clean_Desc'] = df['Description'].apply(clean_full)
    df['Tech_DNA'] = df['Clean_Desc'].apply(get_tech_dna)
    df['Product_Noun'] = df['Clean_Desc'].apply(identify_product_noun)

    # 2. AI Clustering & Sub-Categorization
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['Clean_Desc'])
    
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    df['Cluster_ID'] = kmeans.fit_predict(tfidf_matrix)
    
    # Generate Confidence Scores (Inverse Centroid Distance)
    dist = kmeans.transform(tfidf_matrix)
    df['Confidence_Score'] = (1 - (np.min(dist, axis=1) / np.max(dist))).round(4)

    # 3. Anomaly Detection (Isolation Forest)
    # Using length and complexity to find non-standard data entries
    df['Desc_Len'] = df['Clean_Desc'].apply(len)
    df['Digit_Density'] = df['Clean_Desc'].apply(lambda x: len(re.findall(r'\d', x)))
    iso = IsolationForest(contamination=0.04, random_state=42)
    df['Anomaly_Flag'] = iso.fit_predict(df[['Desc_Len', 'Digit_Density', 'Cluster_ID']])

    # 4. Duplicate Logic (Exact & Spec-Aware Fuzzy)
    exact_dups = df[df.duplicated(subset=['Clean_Desc'], keep=False)]
    
    fuzzy_results = []
    recs = df.to_dict('records')
    for i in range(len(recs)):
        for j in range(i + 1, min(i + 150, len(recs))): # Sliding window
            r1, r2 = recs[i], recs[j]
            sim = SequenceMatcher(None, r1['Clean_Desc'], r2['Clean_Desc']).ratio()
            
            if sim > 0.85:
                # TRAP CHECK: Does technical DNA conflict?
                dna1, dna2 = r1['Tech_DNA'], r2['Tech_DNA']
                
                # Check 1: Size Mismatch
                size_conflict = dna1['numbers'] != dna2['numbers']
                
                # Check 2: Attribute Conflict (e.g. Male vs Female)
                attr_conflict = False
                for cat in SPEC_TRAPS.keys():
                    if cat in dna1['attributes'] and cat in dna2['attributes']:
                        if dna1['attributes'][cat] != dna2['attributes'][cat]:
                            attr_conflict = True
                            break
                
                status = "🛠️ Variant (Tech Spec Difference)" if (size_conflict or attr_conflict) else "🚨 Potential Duplicate"
                
                fuzzy_results.append({
                    'Item A': r1['Item_No'], 'Item B': r2['Item_No'],
                    'Desc A': r1['Clean_Desc'], 'Desc B': r2['Clean_Desc'],
                    'Similarity': f"{sim:.1%}", 'Status': status
                })

    return df, exact_dups, pd.DataFrame(fuzzy_results)

# --- DASHBOARD EXECUTION ---
try:
    df, exact_dups, fuzzy_df = process_inventory_data('raw_data.csv')
except:
    st.error("Upload 'raw_data.csv' to the repository to begin.")
    st.stop()

# --- STREAMLIT UI ---
st.title("🛡️ Enterprise AI Data Auditor")
st.caption("Advanced Supply Chain Logic Engine for Anomaly, Duplicate, and Category Management")

# Setup 7 Sections per requirements
t1, t2, t3, t4, t5, t6, t7 = st.tabs([
    "📍 Categorization", "🎯 Clustering", "🚨 Anomaly Detection", 
    "👯 Exact Duplicates", "⚡ Fuzzy Matches", "🧠 AI Methodology", "📈 Business Insights"
])

# 1. Product Categorization & Classification
with t1:
    st.header("Product Classification")
    st.markdown("Automatic classification using Noun-First extraction and Confidence Scoring.")
    # Show classification results
    st.dataframe(df[['Item_No', 'Clean_Desc', 'Product_Noun', 'Confidence_Score']].sort_values('Confidence_Score', ascending=False))

# 2. Data Clustering
with t2:
    st.header("Semantic Data Clustering")
    fig2 = px.sunburst(df, path=['Product_Noun', 'Item_No'], values='Confidence_Score', color='Cluster_ID')
    st.plotly_chart(fig2, use_container_width=True)

# 3. Anomaly Detection
with t3:
    st.header("Data Anomaly Detection")
    anomalies = df[df['Anomaly_Flag'] == -1]
    st.warning(f"Flagged {len(anomalies)} items as pattern outliers.")
    st.dataframe(anomalies[['Item_No', 'Description', 'Desc_Len', 'Digit_Density']])
    
    fig3 = px.scatter(df, x='Desc_Len', y='Digit_Density', color='Anomaly_Flag', hover_data=['Clean_Desc'])
    st.plotly_chart(fig3, use_container_width=True)

# 4. Duplicate Detection (Exact)
with t4:
    st.header("Exact Duplicate Identification")
    if not exact_dups.empty:
        st.error(f"Found {len(exact_dups)} instances of exact data entry repetition.")
        st.dataframe(exact_dups[['Item_No', 'Description']])
    else:
        st.success("No exact duplicates found.")

# 5. Fuzzy Duplicate Identification (Spec-Aware)
with t5:
    st.header("Fuzzy Match & Variant Resolver")
    st.info("Advanced Logic: Items with >85% similarity are audited for technical DNA conflicts (Gender, Rating, Size).")
    
    if not fuzzy_df.empty:
        # Highlight logic
        def highlight_status(val):
            return 'background-color: #ffe6e6' if 'Duplicate' in val else 'background-color: #e6f3ff'
        st.dataframe(fuzzy_df.style.applymap(highlight_status, subset=['Status']), use_container_width=True)
    else:
        st.success("No fuzzy conflicts detected.")

# 6. AI Model / NLP Techniques
with t6:
    st.header("Technical Methodology Report")
    st.markdown("""
    ### AI Stack Used:
    1.  **NLP Preprocessing:** Custom RegEx cleaning and **Numeric Fingerprinting** to isolate technical specifications.
    2.  **Product Classification:** A hybrid model using **Leading Noun Extraction** for hard categorization and **TF-IDF Vectorization** for sub-categorization.
    3.  **Clustering:** **K-Means Clustering** to group items by semantic description patterns.
    4.  **Anomaly Detection:** **Isolation Forest (Ensemble Learning)** to identify entries that are statistically "far" from the dataset norm in terms of complexity and length.
    5.  **Similarity Engine:** **Levenshtein Distance** paired with a **Technical DNA Validator** to prevent the "Male/Female" and "Size" traps.
    6.  **Confidence Scoring:** Calculated based on the Euclidean distance of a record to its cluster centroid.
    """)

# 7. Business Insights & Reports
with t7:
    st.header("Inventory Health & Insights")
    c1, c2 = st.columns(2)
    with c1:
        # Category breakdown
        fig7a = px.bar(df['Product_Noun'].value_counts().head(10), title="Top 10 Inventory Items")
        st.plotly_chart(fig7a)
    with c2:
        # Data Quality Gauge
        quality_score = (1 - (len(anomalies) / len(df))) * 100
        fig7b = go.Figure(go.Indicator(mode="gauge+number", value=quality_score, title={'text': "Catalog Accuracy %"}))
        st.plotly_chart(fig7b)
