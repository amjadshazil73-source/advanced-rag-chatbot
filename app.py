import streamlit as st
import os
import httpx
import json
import time

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:10001"
TIMEOUT = 180.0 # Increased for complex RAG tasks

st.set_page_config(
    page_title="Industry RAG Dashboard",
    page_icon="🤖",
    layout="wide"
)



# --- STYLING ---
st.markdown("""
<style>
    .main { background-color: #F8FAFB; }
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    .source-card { 
        background-color: white; 
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid #E3E8ED;
        font-size: 0.85rem;
        margin-top: 5px;
    }
    .metric-bubble {
        background-color: #4BAE96;
        color: white;
        padding: 5px 10px;
        border-radius: 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- APP STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Persistent HTTP Client for efficiency
@st.cache_resource
def get_http_client():
    # In single-container deployment, the API is always at 127.0.0.1:8000
    return httpx.Client(base_url=API_URL, timeout=TIMEOUT)

client = get_http_client()

# --- SIDEBAR: METRICS & UPLOAD ---
with st.sidebar:
    st.title("🛡️ RAG Ops Center")
    st.markdown("---")
    
    # 1. Upload Section
    st.subheader("📁 Ingest Documents")
    uploaded_file = st.file_uploader("Upload a PDF to the system", type="pdf")
    if st.button("Index Document", use_container_width=True):
        if uploaded_file:
            with st.spinner("Processing PDF and updating Vector DB..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                    resp = client.post("/ingest", files=files)
                    if resp.status_code == 200:
                        st.success("Ingestion Complete!")
                    else:
                        st.error(f"Error: {resp.text}")
                except Exception as e:
                    st.error(f"API Connection Error: {e}")
        else:
            st.warning("Please select a file first.")

    st.markdown("---")

    # 2. Results Dashboard
    st.subheader("📊 Evaluation Metrics")
    try:
        metrics_resp = client.get("/metrics")
        if metrics_resp.status_code == 200:
            results = metrics_resp.json()
            if results:
                # Calculate simple averages
                avg_faith = sum(r["scores"]["FaithfulnessMetric"] for r in results) / len(results)
                avg_rel = sum(r["scores"]["AnswerRelevancyMetric"] for r in results) / len(results)
                
                st.metric("Faithfulness", f"{avg_faith:.2f}", delta="Target 0.8+")
                st.metric("Relevancy", f"{avg_rel:.2f}", delta="Target 0.8+")
                st.info(f"Based on {len(results)} automated test cases.")
            else:
                st.info("No evaluation data yet. Run Phase 4 to see metrics here.")
    except Exception:
        st.caption("Unable to fetch live metrics.")

# --- MAIN CHAT STAGE ---
st.title("Advanced RAG Assistant")

def check_api_health():
    try:
        # Internal loopback on port 10001
        response = httpx.get("http://127.0.0.1:10001/health", timeout=2.0)
        return response.status_code == 200, "Healthy"
    except Exception as e:
        # Diagnostic: Try to read the internal api logs if they exist
        log_content = ""
        if os.path.exists("/app/api.log"):
            with open("/app/api.log", "r") as f:
                log_content = f.read()[-2000:] # Last 2000 chars
        return False, f"{str(e)}\n\nAPI LOGS:\n{log_content}"

# Heartbeat check
is_healthy, error_msg = check_api_health()
if not is_healthy:
    st.info("🔄 System is waking up...")
    st.caption("Waiting for Backend API...")
    st.error(f"Diagnostic Report:\n{error_msg}")
    time.sleep(5)
    st.rerun()

st.caption("Production pipeline with Hybrid Search + Reranking + Gemini 1.5")

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("View 8 Internal Sources"):
                for i, src in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1}**")
                    st.markdown(f"*{src['metadata'].get('source', 'Unknown')}*")
                    st.code(src["content"][:200] + "...", language="text")

# Chat Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Clear visual state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        sources = []
        
        try:
            # Use streaming API
            with client.stream("POST", "/ask_stream", json={"question": prompt}) as r:
                if r.status_code != 200:
                    error_text = r.read().decode()
                    st.error(f"Chat failed (Backend Error): {error_text}")
                    return
                
                for line in r.iter_lines():
                    if not line: continue
                    data = json.loads(line)
                    
                    if data["type"] == "metadata":
                        sources = data["source_chunks"]
                    elif data["type"] == "content":
                        full_response += data["delta"]
                        placeholder.markdown(full_response + "▌")
                    elif data["type"] == "end":
                        placeholder.markdown(full_response)
        
            # Add to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "sources": sources
            })
            
            # Show sources after completion
            if sources:
                # Add to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "sources": sources
                })
                
                # Show sources after completion
                if sources:
                    with st.expander("View 8 Internal Sources"):
                        for i, src in enumerate(sources):
                            st.markdown(f"**Source {i+1}**")
                            st.markdown(f"*{src['metadata'].get('source', 'Unknown')}*")
                            st.markdown(f"```text\n{src['content'][:300]}...\n```")

        except Exception as e:
            st.error(f"Chat failed: {e}")
            if "RESOURCE_EXHAUSTED" in str(e):
                st.warning("API Quota Reached! Please try again tomorrow.")
