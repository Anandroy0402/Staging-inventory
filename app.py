import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import socket
import time
from pathlib import Path
from difflib import SequenceMatcher

# Advanced AI/ML Imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

# Hugging Face Hub Integration
import huggingface_hub

# Correct Import Paths for Newer Versions (v0.17+)
try:
    from huggingface_hub import InferenceClient
    from huggingface_hub.errors import (
        InferenceTimeoutError, 
        HfHubHTTPError
    )
    
    # Define a dummy alias for RateLimitError since it doesn't exist 
    # (We handle it via HfHubHTTPError 429 later)
    RateLimitError = HfHubHTTPError 
    
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.info("💡 Solution: Your 'huggingface_hub' version might be outdated or the import paths are incorrect.")
    st.stop()
except Exception as e:
    st.error(f"Unexpected Error: {e}")
    st.stop()

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Inventory Auditor Pro", layout="wide", page_icon="🛡️")

# --- CONFIGURATION & CONSTANTS ---
DEFAULT_PRODUCT_GROUP = "Consumables & General"
COMPARISON_WINDOW_SIZE = 50
FUZZY_SIMILARITY_THRESHOLD = 0.85
SEMANTIC_SIMILARITY_THRESHOLD = 0.9
HF_BATCH_SIZE = 16
HF_ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_INFERENCE_TIMEOUT = 30
HF_CONNECTION_CACHE_TTL = 30
HF_API_HOSTNAME = "api-inference.huggingface.co"

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

# --- UTILITY FUNCTIONS ---
def get_streamlit_secrets():
    secrets_path = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return {}
    try:
        with secrets_path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return {}

def resolve_setting(key, default=None, as_bool=False):
    value = os.getenv(key)
    if value is None:
        secrets = get_streamlit_secrets()
        value = secrets.get(key)
        if value is None:
            try:
                value = st.secrets.get(key)
            except (AttributeError, KeyError):
                value = None
    
    if value is None:
        return default
    
    if as_bool:
        if isinstance(value, bool): return value
        return str(value).strip().lower() == "true"
    return str(value).strip()

# --- AI CONNECTIVITY BACKEND (REFACTORED) ---
class InventoryAIBackend:
    """
    Encapsulates all Hugging Face API interactions, connection checks, 
    token management, and retry logic.
    """
    def __init__(self):
        self.enabled = resolve_setting("ENABLE_HF_MODELS", default=False, as_bool=True)
        self.token = self._resolve_token()
        self.client = InferenceClient(token=self.token, timeout=HF_INFERENCE_TIMEOUT) if self.token else None
        self.status = {"zero_shot": False, "embeddings": False, "reason": "init", "detail": ""}

    def _resolve_token(self):
        keys = ["HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_API_TOKEN"]
        for k in keys:
            t = resolve_setting(k)
            if t and t.startswith("hf_") and len(t) >= 20:
                return t
        return None

    def _check_network(self):
        """Low-level network diagnostic."""
        try:
            socket.gethostbyname(HF_API_HOSTNAME)
        except Exception:
            return False, "DNS Resolution Failed"
        
        try:
            with socket.create_connection((HF_API_HOSTNAME, 443), timeout=5):
                return True, "OK"
        except Exception as e:
            return False, f"TCP Connect Failed: {str(e)}"

    def _execute_with_retry(self, func, retries=2, delay=2):
        """Exponential backoff for transient API errors."""
        try:
            return func()
        except (InferenceTimeoutError, RateLimitError, HfHubHTTPError) as e:
            should_retry = False
            if isinstance(e, (InferenceTimeoutError, RateLimitError)):
                should_retry = True
            elif isinstance(e, HfHubHTTPError) and e.response.status_code in [429, 503, 504]:
                should_retry = True
            
            # Check for loading state messages
            err_str = str(e).lower()
            if "loading" in err_str:
                should_retry = True
            
            if should_retry and retries > 0:
                time.sleep(delay)
                return self._execute_with_retry(func, retries - 1, delay * 2)
            raise e

    @st.cache_data(ttl=HF_CONNECTION_CACHE_TTL)
    def check_health(_self):  # _self prevents hashing the object
        """Comprehensive health check returns a dict for the UI."""
        if not _self.enabled:
            return {"enabled": False, "status": "disabled", "reason": "disabled"}
        if not _self.token:
            return {"enabled": False, "status": "missing_token", "reason": "Token not found or invalid"}

        net_ok, net_msg = _self._check_network()
        if not net_ok:
            return {"enabled": False, "status": "network_error", "reason": net_msg}

        # Functional Tests
        zs_ok, emb_ok = False, False
        test_text = "Inventory Audit Check"
        
        try:
            _self._execute_with_retry(lambda: _self.client.feature_extraction(test_text, model=HF_EMBEDDING_MODEL))
            emb_ok = True
        except Exception: pass

        try:
            _self._execute_with_retry(lambda: _self.client.zero_shot_classification(test_text, list(PRODUCT_GROUPS.keys())[:3]))
            zs_ok = True
        except Exception: pass

        if zs_ok and emb_ok:
            status = "connected"
        elif zs_ok or emb_ok:
            status = "partial"
        else:
            status = "api_error"

        return {
            "enabled": True,
            "status": status,
            "zero_shot": zs_ok,
            "embeddings": emb_ok,
            "reason": "All systems go" if status == "connected" else "Some models failed"
        }

    def get_zero_shot(self, texts, labels):
        if not self.token: return None
        if isinstance(texts, str): texts = [texts]
        
        results = []
        try:
            for i in range(0, len(texts), HF_BATCH_SIZE):
                batch = texts[i:i + HF_BATCH_SIZE]
                batch_res = self._execute_with_retry(
                    lambda: self.client.zero_shot_classification(
                        batch, labels, hypothesis_template="This industrial inventory item is {}"
                    )
                )
                if isinstance(batch_res, dict): batch_res = [batch_res]
                results.extend(batch_res)
            return results
        except Exception as e:
            st.warning(f"Zero-Shot classification failed: {str(e)}")
            return None

    def get_embeddings(self, texts):
        if not self.token: return None
        try:
            raw_embeds = []
            for i in range(0, len(texts), HF_BATCH_SIZE):
                batch = texts[i:i + HF_BATCH_SIZE]
                res = self._execute_with_retry(
                    lambda: self.client.feature_extraction(batch, model=HF_EMBEDDING_MODEL)
                )
                # Handle single vs batch return
                if isinstance(res, list) and res and isinstance(res[0], float):
                    res = [res]
                raw_embeds.extend(res)
            
            # Normalize
            embeddings = np.array(raw_embeds, dtype=float)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            return embeddings / norms
        except Exception as e:
            st.warning(f"Embedding generation failed: {str(e)}")
            return None

# Initialize Backend
ai_backend = InventoryAIBackend()

# --- BUSINESS LOGIC (PRESERVED) ---
def clean_description(text):
    text = str(text).upper().replace('"', ' ')
    text = text.replace("O-RING", "O RING").replace("MECH-SEAL", "MECHANICAL SEAL").replace("MECH SEAL", "MECHANICAL SEAL")
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
    # Priority phrases
    phrases = ["MEASURING TAPE", "BALL VALVE", "GATE VALVE", "PLUG VALVE", "CHECK VALVE", "MECHANICAL SEAL", "PAINT BRUSH", "WIRE STRIPPER", "CUTTING PLIER", "DRILL BIT"]
    for p in phrases:
        if re.search(token_pattern(p), text): return p
    # Noun Knowledge Base
    all_nouns = [item for sublist in PRODUCT_GROUPS.values() for item in sublist]
    for n in all_nouns:
        if re.search(token_pattern(n), text): return n
    return text.split()[0] if text.split() else "MISC"

def map_product_group(noun):
    for group, keywords in PRODUCT_GROUPS.items():
        if noun in keywords: return group
    for group, keywords in PRODUCT_GROUPS.items():
        for keyword in keywords:
            if re.search(token_pattern(keyword), noun): return group
    return DEFAULT_PRODUCT_GROUP

def dominant_group(series):
    counts = series.value_counts()
    return counts.idxmax() if not counts.empty else "UNMAPPED"

def apply_distance_floor(distances, min_threshold=1e-8):
    max_dist = np.max(distances, axis=1)
    return np.where(max_dist == 0, min_threshold, max_dist)

def normalize_confidence_scores(scores):
    if scores.empty: return scores
    min_s, max_s = scores.min(), scores.max()
    if max_s >= 0.8: return scores
    if max_s == min_s: return pd.Series(np.full(len(scores), max(max_s, 0.6)), index=scores.index)
    return ((scores - min_s) / (max_s - min_s) * (0.98 - 0.6) + 0.6).round(4)

def build_fuzzy_duplicates(df, id_col):
    fuzzy_list = []
    recs = df.to_dict('records')
    for i in range(len(recs)):
        for j in range(i + 1, min(i + COMPARISON_WINDOW_SIZE, len(recs))):
            r1, r2 = recs[i], recs[j]
            desc1, desc2 = r1.get('Standard_Desc', ''), r2.get('Standard_Desc', '')
            sim = SequenceMatcher(None, desc1, desc2).ratio()
            if sim > FUZZY_SIMILARITY_THRESHOLD:
                dna1 = r1.get('Tech_DNA', {'numbers': set(), 'attributes': {}})
                dna2 = r2.get('Tech_DNA', {'numbers': set(), 'attributes': {}})
                is_variant = (dna1['numbers'] != dna2['numbers']) or (dna1['attributes'] != dna2['attributes'])
                fuzzy_list.append({
                    'ID A': r1[id_col], 'ID B': r2[id_col],
                    'Desc A': desc1, 'Desc B': desc2,
                    'Match %': f"{sim:.1%}", 'Verdict': "🛠️ Variant" if is_variant else "🚨 Duplicate"
                })
    return fuzzy_list

# --- MAIN ENGINE ---
@st.cache_data
def run_intelligent_audit(file_path, _ai_health):
    df = pd.read_csv(file_path, encoding='latin1')
    df.columns = [c.strip() for c in df.columns]
    
    # ID/Desc Column Discovery
    id_col = next(c for c in df.columns if any(x in c.lower() for x in ['item', 'no']))
    desc_col = next(c for c in df.columns if 'desc' in c.lower())
    
    # 1. Processing
    df['Standard_Desc'] = df[desc_col].apply(clean_description)
    df['Tech_DNA'] = df['Standard_Desc'].apply(get_tech_dna)
    df['Part_Noun'] = df['Standard_Desc'].apply(intelligent_noun_extractor)
    df['Product_Group'] = df['Part_Noun'].apply(map_product_group)

    # 2. Local TF-IDF Clustering
    tfidf = TfidfVectorizer(max_features=300, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Standard_Desc'])
    
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    df['Cluster_ID'] = kmeans.fit_predict(tfidf_matrix)
    dists = kmeans.transform(tfidf_matrix)
    max_tfidf_dist = apply_distance_floor(dists)
    df['Confidence'] = (1 - (np.min(dists, axis=1) / max_tfidf_dist)).round(4)
    
    cluster_groups = df.groupby('Cluster_ID')['Product_Group'].agg(dominant_group)
    df['Cluster_Group'] = df['Cluster_ID'].map(cluster_groups)
    df['Cluster_Validated'] = df['Product_Group'] == df['Cluster_Group']
    
    # 3. Local Anomaly (TF-IDF)
    iso = IsolationForest(contamination=0.04, random_state=42)
    df['Anomaly_Flag'] = iso.fit_predict(tfidf_matrix)

    # 4. Hugging Face Integation
    enable_zs = _ai_health.get("zero_shot", False)
    enable_emb = _ai_health.get("embeddings", False)

    # Prepare inputs
    hf_inputs = df['Part_Noun'].fillna('').str.cat(df['Standard_Desc'].fillna(''), sep=' ').tolist() if enable_zs else None
    
    # Zero-Shot Call
    hf_results = ai_backend.get_zero_shot(hf_inputs, list(PRODUCT_GROUPS.keys())) if enable_zs else None
    
    if hf_results:
        df['HF_Product_Group'] = [res['labels'][0] for res in hf_results]
        df['HF_Product_Confidence'] = [round(res['scores'][0], 4) for res in hf_results]
    else:
        df['HF_Product_Group'] = df['Product_Group']
        df['HF_Product_Confidence'] = df['Confidence']
    
    df['HF_Product_Confidence'] = normalize_confidence_scores(df['HF_Product_Confidence'])

    # Embedding Call
    embeddings = ai_backend.get_embeddings(df['Standard_Desc'].tolist()) if enable_emb else None
    
    if embeddings is not None:
        kmeans_hf = KMeans(n_clusters=8, random_state=42, n_init=10)
        df['HF_Cluster_ID'] = kmeans_hf.fit_predict(embeddings)
        hf_dists = kmeans_hf.transform(embeddings)
        max_dist = apply_distance_floor(hf_dists)
        df['HF_Cluster_Confidence'] = (1 - (np.min(hf_dists, axis=1) / max_dist)).round(4)
        
        iso_hf = IsolationForest(contamination=0.04, random_state=42)
        df['HF_Anomaly_Flag'] = iso_hf.fit_predict(embeddings)
        df['HF_Embedding'] = list(embeddings)
    else:
        df['HF_Cluster_ID'] = df['Cluster_ID']
        df['HF_Cluster_Confidence'] = df['Confidence']
        df['HF_Anomaly_Flag'] = df['Anomaly_Flag']
        df['HF_Embedding'] = [None] * len(df)

    return df, id_col, desc_col

# --- DATA LOADING & HEALTH ---
ai_health = ai_backend.check_health()
target_file = 'raw_data.csv'

if os.path.exists(target_file):
    df_raw, id_col, desc_col = run_intelligent_audit(target_file, ai_health)
else:
    st.error("Data file missing from repository. Please ensure 'raw_data.csv' is present.")
    st.stop()

# --- HEADER & NAVIGATION ---
st.title("🛡️ AI Inventory Auditor Pro")
st.markdown("### Advanced Inventory Intelligence & Quality Management")

# Status Banner
status_cols = st.columns([3, 1])
with status_cols[0]:
    if ai_health["status"] == "connected":
        st.success(f"✅ AI Connected (Zero-Shot & Embeddings Active)")
    elif ai_health["status"] == "partial":
        st.warning(f"⚠️ Partial AI Connectivity: {ai_health.get('reason')}")
    elif ai_health["status"] == "disabled":
        st.info("ℹ️ Hosted AI Disabled (Using Local ML)")
    else:
        st.error(f"❌ AI Connection Failed: {ai_health.get('reason')}")

group_options = list(PRODUCT_GROUPS.keys())
page = st.tabs(["📈 Executive Dashboard", "📍 Categorization Audit", "🚨 Quality Hub", "🧠 Technical Methodology", "🧭 My Approach"])

# --- PAGE 1: EXECUTIVE DASHBOARD ---
with page[0]:
    st.markdown("#### 📊 Inventory Health Overview")
    selected_group = st.multiselect("Product Category", options=group_options, default=group_options, key="dash_group")
    df = df_raw[df_raw['Product_Group'].isin(selected_group)]
    st.markdown("---")
    
    fuzzy_list = build_fuzzy_duplicates(df, id_col)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("📦 SKUs Analyzed", len(df))
    kpi2.metric("🎯 Mean HF Confidence", f"{df['HF_Product_Confidence'].mean():.1%}")
    kpi3.metric("⚠️ HF Anomalies", len(df[df['HF_Anomaly_Flag'] == -1]))
    kpi4.metric("🔄 Duplicate Pairs", len(fuzzy_list))

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.pie(df, names='HF_Product_Group', title="Inventory Distribution", hole=0.4), use_container_width=True)
    with col2:
        top_nouns = df['Part_Noun'].value_counts().head(10).reset_index()
        st.plotly_chart(px.bar(top_nouns, x='Part_Noun', y='count', title="Top 10 Products"), use_container_width=True)

    # Health Gauge
    health_val = (len(df[df['HF_Anomaly_Flag'] == 1]) / len(df)) * 100
    st.plotly_chart(go.Figure(go.Indicator(
        mode = "gauge+number", value = health_val, title = {'text': "Catalog Data Accuracy %"},
        gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#00cc96"}}
    )), use_container_width=True)

# --- PAGE 2: CATEGORIZATION AUDIT ---
with page[1]:
    st.markdown("#### 📍 AI Categorization Audit")
    selected_group = st.multiselect("Product Category", options=group_options, default=group_options, key="cat_group")
    df = df_raw[df_raw['Product_Group'].isin(selected_group)]
    
    st.dataframe(df[[id_col, 'Standard_Desc', 'Part_Noun', 'Product_Group', 'HF_Product_Group', 'HF_Product_Confidence']].sort_values('HF_Product_Confidence'), use_container_width=True)
    st.plotly_chart(px.histogram(df, x="HF_Product_Confidence", nbins=20, title="Confidence Distribution"), use_container_width=True)

# --- PAGE 3: QUALITY HUB ---
with page[2]:
    st.markdown("#### 🚨 Quality Hub")
    selected_group = st.multiselect("Product Category", options=group_options, default=group_options, key="qual_group")
    df = df_raw[df_raw['Product_Group'].isin(selected_group)]
    
    t1, t2, t3 = st.tabs(["⚠️ Anomalies", "👯 Fuzzy Duplicates", "🧠 Semantic Duplicates"])
    
    with t1:
        st.dataframe(df[df['HF_Anomaly_Flag'] == -1][[id_col, desc_col, 'HF_Product_Group', 'HF_Cluster_Confidence']], use_container_width=True)
        
    with t2:
        fuzzy = build_fuzzy_duplicates(df, id_col)
        if fuzzy: st.dataframe(pd.DataFrame(fuzzy), use_container_width=True)
        else: st.success("No fuzzy duplicates found.")
        
    with t3:
        if df['HF_Embedding'].iloc[0] is None:
            st.info("Embeddings unavailable.")
        else:
            sem_list = []
            recs = df.to_dict('records')
            embeds = list(df['HF_Embedding'])
            for i in range(len(recs)):
                for j in range(i + 1, min(i + COMPARISON_WINDOW_SIZE, len(recs))):
                    sim = float(np.dot(embeds[i], embeds[j]))
                    if sim > SEMANTIC_SIMILARITY_THRESHOLD:
                        sem_list.append({
                            'ID A': recs[i][id_col], 'ID B': recs[j][id_col],
                            'Desc A': recs[i]['Standard_Desc'], 'Desc B': recs[j]['Standard_Desc'],
                            'Match %': f"{sim:.1%}"
                        })
            if sem_list: st.dataframe(pd.DataFrame(sem_list), use_container_width=True)
            else: st.success("No semantic duplicates found.")

# --- PAGE 4: METHODOLOGY ---
with page[3]:
    st.markdown("#### 🧠 Technical Methodology")
    st.markdown("### Connection Diagnostics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Status", ai_health["status"])
    c2.metric("Zero-Shot", "✅" if ai_health["zero_shot"] else "❌")
    c3.metric("Embeddings", "✅" if ai_health["embeddings"] else "❌")
    
    if ai_health["status"] != "connected":
        st.error(f"Reason: {ai_health.get('reason')}")
        
    st.markdown("### Stack Info")
    st.json({
        "Timeout": f"{HF_INFERENCE_TIMEOUT}s",
        "ZeroShot Model": HF_ZERO_SHOT_MODEL,
        "Embedding Model": HF_EMBEDDING_MODEL,
        "Backend": "Hugging Face Inference API" if ai_health["enabled"] else "Local Scikit-Learn"
    })

# --- PAGE 5: MY APPROACH ---
with page[4]:
    st.markdown("#### 🧭 My Approach")
    st.markdown("""
    This app uses a **Hybrid Intelligence Model**:
    1. **Heuristic Layer:** Regex & Knowledge Base for exact parsing of technical specs.
    2. **Statistical Layer:** TF-IDF & K-Means for local pattern recognition.
    3. **Neural Layer:** Hugging Face Transformers for semantic understanding.
    """)
