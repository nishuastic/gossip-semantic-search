import os
import requests
import streamlit as st
from datetime import datetime
import json
from pathlib import Path

st.set_page_config(
    page_title="Gossip Semantic Search",
    page_icon="🔎",
    layout="wide",
)

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_PATH = DATA_DIR / "ui_search_history.json"


def load_history_from_disk():
    try:
        if HISTORY_PATH.exists():
            return json.loads(HISTORY_PATH.read_text())
    except Exception:
        pass
    return []


def save_history_to_disk(history):
    try:
        HISTORY_PATH.write_text(json.dumps(history[:50], ensure_ascii=False, indent=2))
    except Exception:
        pass


# Sidebar controls
with st.sidebar:
    st.markdown("## Controls")
    st.markdown("Search across VSD & Public with semantic relevance.")
    top_k = st.slider("Results to show", min_value=3, max_value=20, value=8, step=1)
    categories = st.multiselect(
        "Categories",
        options=[
            "vsd_people",
            "vsd_tv",
            "vsd_company",
            "vsd_culture",
            "vsd_leisure",
            "public_news",
            "public_people",
            "public_tv",
            "public_fashion",
            "public_royalty",
        ],
        help="Filter by source category.",
    )
    st.markdown("---")
    if st.button("Clear recent searches", use_container_width=True):
        st.session_state.history = []
        save_history_to_disk([])
        st.success("History cleared")
    st.caption("Tip: Try queries like 'royal wedding outfit' or 'TV host controversy'.")

# Header
st.markdown("### 🔎 Gossip Semantic Search")
st.caption(
    "A fast, premium search experience over curated celebrity and entertainment news."
)

# Query input
col_q1, col_q2 = st.columns([6, 1])
with col_q1:
    query = st.text_input(
        "What are you looking for?",
        placeholder="e.g., latest on celebrity weddings",
        label_visibility="collapsed",
    )
with col_q2:
    search_clicked = st.button("Search", use_container_width=True)

# Search history state (load from disk on first run)
if "history" not in st.session_state:
    st.session_state.history = load_history_from_disk()


def render_metrics(metrics):
    m1, m2, m3 = st.columns(3)
    m1.metric("Results returned", metrics.get("top_k", 0))
    m2.metric("Indexed articles", metrics.get("total_vectors", 0))
    m3.metric("Search time", f"{metrics.get('elapsed_ms', 0)} ms")


def render_result_card(item):
    with st.container(border=True):
        st.markdown(f"**[{item['title']}]({item['url']})**")
        st.caption(f"{item.get('category', 'unknown')} • {item.get('published', '')}")
        st.write(item.get("summary", ""))


if search_clicked and query:
    with st.spinner("Searching the index..."):
        try:
            payload = {"query": query, "top_k": top_k}
            if categories:
                payload["categories"] = categories
            response = requests.post(
                "http://localhost:8000/search", json=payload, timeout=60
            )
            response.raise_for_status()
            data = response.json()

            # Metrics row
            render_metrics(data.get("metrics", {}))

            # Results grid
            results = data.get("results", [])
            if not results:
                st.info(
                    "No results found. Try broadening your query or removing filters."
                )
            else:
                left, right = st.columns(2)
                for i, item in enumerate(results):
                    (left if i % 2 == 0 else right).markdown(" ")
                    with left if i % 2 == 0 else right:
                        render_result_card(item)

            # Append to history
            st.session_state.history.insert(
                0,
                {
                    "query": query,
                    "when": datetime.now().strftime("%H:%M:%S"),
                    "count": len(results),
                    "elapsed_ms": data.get("metrics", {}).get("elapsed_ms", 0),
                },
            )
            st.session_state.history = st.session_state.history[:10]
            save_history_to_disk(st.session_state.history)
        except requests.RequestException as e:
            st.error(f"Backend error: {e}")
elif search_clicked and not query:
    st.warning("Please enter a query.")

# Recent searches section
with st.expander("Recent searches", expanded=False):
    if not st.session_state.history:
        st.caption("No searches yet.")
    else:
        for item in st.session_state.history:
            st.write(
                f"{item['when']} • '{item['query']}' — {item['count']} results in {item['elapsed_ms']} ms"
            )
