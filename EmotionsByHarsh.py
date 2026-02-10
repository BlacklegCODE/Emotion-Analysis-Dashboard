import os
import io
import time
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
import plotly.express as px

# ---------------- Config ----------------
st.set_page_config(page_title="Emotion Studio", layout="wide", initial_sidebar_state="expanded")
CSV_PATH = "emotions.csv.gz"
AUTHOR_TEXT = "Code Author — Harsh Raundal"
AUTHOR_LINK = "https://github.com/BlacklegCODE"
EMOTION_MAP = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
CLASS_LIST = np.array([0, 1, 2, 3, 4, 5])

# ---------------- Polished CSS + Animations ----------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, .reportview-container, .main, .block-container {
        background: #030617 !important;
        color: #e6eef8;
        font-family: 'Inter', system-ui, -apple-system;
    }

    /* ===== Heavy result hero ===== */
    .result-hero {
        font-size:72px;
        text-align:center;
        padding:18px;
        border-radius:18px;
        background: linear-gradient(135deg,#7C4DFF22,#00C2FF22);
        border:1px solid rgba(124,77,255,0.25);
        box-shadow: 0 0 40px rgba(124,77,255,0.25);
        animation: popIn .4s ease;
    }

    @keyframes popIn {
        from {transform:scale(.9); opacity:0;}
        to {transform:scale(1); opacity:1;}
    }

    .emotion-chip {
        display:inline-block;
        padding:6px 14px;
        border-radius:999px;
        background:#111a33;
        border:1px solid #2a3a66;
        font-weight:700;
        margin:4px;
        box-shadow: 0 0 12px rgba(124,77,255,0.25);
    }

    /* ===== Sidebar ===== */
    section[data-testid="stSidebar"] {
        min-width: 300px;
        padding-top: 3cm;
        padding-bottom: 3cm;
        display:flex;
        align-items:center;
        justify-content:center;
        background: linear-gradient(180deg, rgba(14,16,32,0.78), rgba(6,8,18,0.62));
        border-right: 1px solid rgba(120,80,255,0.08);
        backdrop-filter: blur(8px);
    }

    section[data-testid="stSidebar"] .sidebar-content {
        width: 86%;
        display:flex;
        gap:12px;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        height: calc(100% - 6cm);
    }

    /* ===== Neon buttons ===== */
    section[data-testid="stSidebar"] .stButton>button {
        width:100%;
        padding:12px;
        border-radius:12px;
        font-weight:700;
        background: linear-gradient(90deg,#06102a,#081324);
        color:#ecf6ff;
        border: 1px solid rgba(130,90,255,0.24);
        box-shadow: 0 8px 28px rgba(100,60,230,0.12);
        transition: transform .12s, box-shadow .12s, border-color .12s;
        cursor:pointer;
    }

    section[data-testid="stSidebar"] .stButton>button:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 30px 70px rgba(100,60,230,0.25);
        border-color: rgba(200,150,255,0.6);
    }

    /* ===== Cards ===== */
    .card {
        border-radius:12px;
        padding:18px;
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        box-shadow: 0 18px 48px rgba(2,6,23,0.75);
        border: 1px solid rgba(120,80,255,0.08);
        color: #eaf3ff;
        animation: fadeUp .45s ease both;
    }

    @keyframes fadeUp {
        from { opacity:0; transform: translateY(8px); }
        to { opacity:1; transform: translateY(0px); }
    }

    /* ===== Examples slide ===== */
    .examples-box {
        max-height:60vh;
        overflow:auto;
        padding-right:8px;
        animation: slideLeft .45s ease both;
    }

    @keyframes slideLeft {
        from { opacity:0; transform: translateX(12px); }
        to { opacity:1; transform: translateX(0px); }
    }

    /* ===== Neon border ===== */
    .neon-border {
        border-radius:14px;
        padding:8px;
        border: 1px solid rgba(120,80,255,0.08);
        animation: neonPulse 4.5s ease-in-out infinite;
    }

    @keyframes neonPulse {
        0% { box-shadow: 0 0 6px rgba(124,77,255,0.05); }
        50% { box-shadow: 0 0 26px rgba(124,77,255,0.18); }
        100% { box-shadow: 0 0 6px rgba(124,77,255,0.05); }
    }

    @media (max-width:900px) {
        section[data-testid="stSidebar"] { display:none; }
        .main > div.block-container { padding-left: 18px; }
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Sidebar (centered buttons + signature) ----------------
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)

    if "nav" not in st.session_state:
        st.session_state.nav = "home"

    if st.button("Test Emotions !", key="sb_test"):
        st.session_state.nav = "test"
    if st.button("Examples of Emotions", key="sb_examples"):
        st.session_state.nav = "examples"
    if st.button("Statistical Analysis", key="sb_stats"):
        st.session_state.nav = "stats"
    if st.button("Help & Navigation", key="sb_help"):
        st.session_state.nav = "help"

    # signature in sidebar
    st.markdown(f'<div class="author-link">{AUTHOR_TEXT} · <a href="{AUTHOR_LINK}" target="_blank">GitHub</a></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
# ---------------- Data loader (supports .csv and .csv.gz) ----------------
@st.cache_data(show_spinner=False)
def load_csv(path):
    if not os.path.exists(path):
        return None
    try:
        # works for both .csv and .csv.gz
        df = pd.read_csv(path, compression="infer", encoding="utf-8")
        # if file has no header, enforce names
        if df.shape[1] == 2:
            df.columns = ["text", "label"]
        df = df.dropna().astype({"text": str, "label": int})
        return df
    except Exception:
        return None


df = load_csv(CSV_PATH)
if df is None or df.shape[0] < 6:
    sample = [
        ("i just feel really helpless and heavy hearted", 4),
        ("i gave up my internship and am feeling distraught", 4),
        ("i dont know i feel so lost", 0),
        ("i feel so stupid that i realise it so late", 0),
        ("i feel like i am actually getting something useful out of it", 1),
        ("i feel anger when i see a parent beating a child", 3),
        ("what a surprising turn of events!", 5),
        ("wow that shocked me", 5),
    ]
    df = pd.DataFrame(sample, columns=["text", "label"])

@st.cache_resource(show_spinner=False)
def build_fast_model_safe(df_local, n_features=2**16, max_train_samples=20000):
    """
    Robust fast model builder:
    - Samples up to max_train_samples for speed
    - Tries HashingVectorizer + SGDClassifier (class_weight='balanced')
    - If that fails, falls back to TF-IDF + LogisticRegression (small) — better than Dummy
    Returns: (vectorizer, clf, present_classes, model_kind)
    model_kind in {'hash_sgd', 'tfidf_lr', 'dummy', 'none'}
    """
    try:
        df_local = df_local.dropna(subset=["text", "label"])
        df_local["text"] = df_local["text"].astype(str)
        df_local["label"] = df_local["label"].astype(int)
    except Exception:
        return None, None, np.array([]), "none"

    # speed: sample if too large
    if df_local.shape[0] > max_train_samples:
        df_sample = df_local.sample(max_train_samples, random_state=42)
    else:
        df_sample = df_local

    texts = df_sample["text"].tolist()
    y = df_sample["label"].values
    if len(texts) == 0:
        return None, None, np.array([]), "none"

    present_classes = np.unique(y)

    # TRY 1: Hashing + SGD (fast), with class_weight balanced to avoid majority bias
    try:
        vec_h = HashingVectorizer(n_features=n_features, alternate_sign=False, ngram_range=(1,2))
        clf_sgd = SGDClassifier(loss="log", max_iter=1000, tol=1e-3, random_state=42, class_weight="balanced")
        X = vec_h.transform(texts)
        clf_sgd.partial_fit(X, y, classes=present_classes)
        return vec_h, clf_sgd, present_classes, "hash_sgd"
    except Exception:
        # TRY 2: TF-IDF + LogisticRegression (smaller, more stable)
        try:
            tf = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
            Xtf = tf.fit_transform(texts)
            lr = LogisticRegression(max_iter=600)
            lr.fit(Xtf, y)
            return tf, lr, present_classes, "tfidf_lr"
        except Exception:
            # TRY 3: Dummy fallback (most frequent) — last resort
            try:
                vec_h2 = HashingVectorizer(n_features=n_features, alternate_sign=False, ngram_range=(1,2))
                dummy = DummyClassifier(strategy="most_frequent")
                X2 = vec_h2.transform(texts)
                dummy.fit(X2, y)
                return vec_h2, dummy, present_classes, "dummy"
            except Exception:
                return None, None, present_classes, "none"

# Build model (cached) and show debug info
vec, clf, used_classes, model_kind = build_fast_model_safe(df)

def fast_predict(text):
    """
    Predict with whichever vectorizer+clf returned.
    Returns (label_int, label_name) or None if model missing.
    """
    if vec is None or clf is None:
        return None
    try:
        Xq = vec.transform([text])
        lab = int(clf.predict(Xq)[0])
        return lab, EMOTION_MAP.get(lab, str(lab))
    except Exception:
        return None

# Faster similarity: smaller TF-IDF, cached
@st.cache_data(show_spinner=False)
def build_tfidf_for_similarity(texts, max_features=4000):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    if len(texts) == 0:
        vec.fit(["dummy"])
        return vec
    vec.fit(texts)
    return vec

@st.cache_data(show_spinner=False)
def similar_sentences_tfidf(query, texts, k=3, max_features=4000):
    if not texts:
        return []
    tvec = build_tfidf_for_similarity(texts, max_features=max_features)
    A = tvec.transform(texts)
    q = tvec.transform([query])
    sims = cosine_similarity(q, A)[0]
    idx = np.argsort(-sims)[:k]
    return [texts[i] for i in idx if i < len(texts)]
# ---------------- TF-IDF builder for similarity (cached) ----------------
@st.cache_data(show_spinner=False)
def build_tfidf_for_similarity(texts):
    vec = TfidfVectorizer(max_features=8000, ngram_range=(1,2))
    if len(texts) == 0:
        vec.fit(["dummy"])
        return vec
    vec.fit(texts)
    return vec

@st.cache_data(show_spinner=False)
def similar_sentences_tfidf(query, texts, k=3):
    if len(texts) == 0:
        return []
    tvec = build_tfidf_for_similarity(texts)
    A = tvec.transform(texts)
    q = tvec.transform([query])
    sims = cosine_similarity(q, A)[0]
    idx = np.argsort(-sims)[:k]
    return [texts[i] for i in idx if i < len(texts)]

# ---------------- prediction history for export ----------------
if "pred_history" not in st.session_state:
    st.session_state.pred_history = []  # list of dicts: {"text","label","label_name","ts"}

# ---------------- UI header + small author badge ----------------
st.markdown('<div class="neon-border" style="padding:10px; margin-bottom:10px;">', unsafe_allow_html=True)
st.markdown("<h2 style='margin:0'>Emotion Studio — Fast & Animated</h2>", unsafe_allow_html=True)
st.markdown("<p style='margin:0; color:#9fbbe0'>Dark • Neon • Smooth transitions • Exportable</p>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# top-right tiny signature
st.markdown(f'<div style="position:fixed; right:18px; top:12px; color:#9aaed3; font-size:12px">{AUTHOR_TEXT} · <a href="{AUTHOR_LINK}" target="_blank" style="color:#7c9cff">GitHub</a></div>', unsafe_allow_html=True)

# ---------------- Navigation ----------------
nav = st.session_state.get("nav", "home")
AUTHOR_TEXT = "Code Author — Harsh Raundal"
AUTHOR_LINK = "https://github.com/BlacklegCODE"
# ---------------- PAGES ----------------
if nav == "test":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Test Emotions !")
    st.write("Type a sentence and press Check. Prediction is fast (Hashing + SGD).")
    user_text = st.text_area("Your sentence", height=160, key="input_text")
    if st.button("Check", key="btn_check"):
        if not user_text or user_text.strip() == "":
            st.warning("Write a sentence first.")
        else:
            res = fast_predict(user_text)
            if res is None:
                st.info("Model not available. Showing closest examples.")
                sims = similar_sentences_tfidf(user_text, df["text"].astype(str).tolist(), k=3)
                st.write("Closest examples:")
                for s in sims:
                    st.write("• " + s)
            else:
                lab, name = res
                st.success(f"Predicted: **{name.upper()}**")
                sims = similar_sentences_tfidf(user_text, df[df["label"] == lab]["text"].astype(str).tolist(), k=3)
                if not sims:
                    sims = similar_sentences_tfidf(user_text, df["text"].astype(str).tolist(), k=3)
                st.markdown("**Similar examples:**")
                for s in sims:
                    st.write("• " + s)
                # record to history with timestamp
                st.session_state.pred_history.append({
                    "text": user_text,
                    "label": int(lab),
                    "label_name": name,
                    "ts": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                })
    # export predictions history button
    if st.session_state.pred_history:
        hist_df = pd.DataFrame(st.session_state.pred_history)
        csv_bytes = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions (CSV)", data=csv_bytes, file_name="predictions_history.csv", mime="text/csv")
    # author small footer here
    st.markdown(f'<div style="margin-top:10px; font-size:12px; color:#9aaed3">{AUTHOR_TEXT} · <a href="{AUTHOR_LINK}" target="_blank" style="color:#7c9cff">GitHub</a></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif nav == "examples":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Examples of Emotions")
    st.write("Click a center button to show examples on the right. Surprise included.")
    c1, c2, c3 = st.columns([0.18, 0.24, 0.58])
    emo_buttons = [("Sadness", 0), ("Joy", 1), ("Love", 2), ("Anger", 3), ("Fear", 4), ("Surprise", 5)]
    with c2:
        for nm, id_ in emo_buttons:
            if st.button(nm, key=f"emo_button_{id_}"):
                st.session_state.chosen_emo = int(id_)
    with c3:
        st.markdown('<div class="examples-box card">', unsafe_allow_html=True)
        chosen = st.session_state.get("chosen_emo", None)
        if chosen is None:
            st.info("Pick an emotion from center to see examples.")
        else:
            examples = df[df["label"] == chosen]["text"].astype(str).tolist()
            if not examples:
                st.write("No examples in dataset for this emotion.")
            else:
                st.markdown(f"### Top {min(10, len(examples))} — {EMOTION_MAP[chosen].upper()}")
                # show and allow selection for export
                for i, ex in enumerate(examples[:10]):
                    st.write(f"{i+1}. {ex}")
                # download displayed examples
                disp_df = pd.DataFrame({"example": examples[:10]})
                csv_bytes = disp_df.to_csv(index=False).encode("utf-8")
                st.download_button(f"Download {EMOTION_MAP[chosen]} examples (CSV)", data=csv_bytes,
                                   file_name=f"examples_{EMOTION_MAP[chosen]}.csv", mime="text/csv")
        st.markdown('</div>', unsafe_allow_html=True)
    # small signature
    st.markdown(f'<div style="margin-top:10px; font-size:12px; color:#9aaed3">{AUTHOR_TEXT} · <a href="{AUTHOR_LINK}" target="_blank" style="color:#7c9cff">GitHub</a></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif nav == "stats":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Statistical Analysis")
    counts = df["label"].value_counts().reindex(sorted(EMOTION_MAP.keys())).fillna(0).astype(int)
    counts.index = [EMOTION_MAP[k].capitalize() for k in counts.index]

    fig_bar = px.bar(x=counts.index, y=counts.values, labels={"x": "Emotion", "y": "Count"}, title="Emotion Counts")
    fig_bar.update_traces(marker_color=['#7C4DFF', '#00C2FF', '#FF6B6B', '#FFB86B', '#7EE787', '#FFD86B'],
                          hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>")
    st.plotly_chart(fig_bar, use_container_width=True)

    fig_pie = px.pie(names=counts.index, values=counts.values, title="Distribution", hole=0.35)
    fig_pie.update_traces(textinfo='percent+label', hoverinfo='label+value')
    st.plotly_chart(fig_pie, use_container_width=True)

    df["word_count"] = df["text"].astype(str).apply(lambda s: len(s.split()))
    wc = df.groupby("label")["word_count"].mean().reset_index().rename(columns={"word_count": "avg_words"})
    wc["emotion"] = wc["label"].map(EMOTION_MAP)
    fig_wc = px.bar(wc, x="emotion", y="avg_words", title="Avg words per sentence by emotion")
    fig_wc.update_traces(hovertemplate="<b>%{x}</b><br>Avg words: %{y:.2f}<extra></extra>")
    st.plotly_chart(fig_wc, use_container_width=True)

    # allow download of full counts & wc as CSV
    counts_df = pd.DataFrame({"emotion": counts.index, "count": counts.values})
    csv_counts = counts_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download counts (CSV)", data=csv_counts, file_name="emotion_counts.csv", mime="text/csv")

    st.markdown(f'<div style="margin-top:10px; font-size:12px; color:#9aaed3">{AUTHOR_TEXT} · <a href="{AUTHOR_LINK}" target="_blank" style="color:#7c9cff">GitHub</a></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if nav == "help":
        st.subheader("Help & Navigation")
        st.write("""
        • Sidebar: use the four centered buttons.  
        • Test Emotions: type → Check → fast prediction + similar examples.  
        • Examples: click center button → top examples appear right + download.  
        • Stats: bar, pie, avg words + download counts.
        """)
        st.write(f"Data loaded from: `{CSV_PATH}`")
        # prominent author block on help page
        st.markdown(f'<div style="margin-top:8px; padding:10px; border-radius:8px; background:rgba(120,80,255,0.02); color:#cfe6ff;">{AUTHOR_TEXT} — <a href="{AUTHOR_LINK}" target="_blank" style="color:#7c9cff">GitHub</a></div>', unsafe_allow_html=True)
    else:
        st.subheader("Welcome — Emotion Studio")
        st.write("Use the sidebar to navigate. Sidebar buttons are centered and styled neon.")
    st.markdown('</div>', unsafe_allow_html=True)
AUTHOR_TEXT = "Code Author — Harsh Raundal"
AUTHOR_LINK = "https://github.com/BlacklegCODE"
# ---------------- footer signature ----------------
st.markdown(
    f"""
    <div style="position:fixed; right:18px; bottom:8px; color:#9aaed3; font-size:12px">
        {AUTHOR_TEXT} • <a href="{AUTHOR_LINK}" target="_blank" style="color:#7c9cff">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)
# ------------- End -------------



