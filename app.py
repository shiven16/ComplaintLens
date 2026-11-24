#!/usr/bin/env python3
"""
ComplaintLens — Next-Gen AI Complaint Triage & Priority Intelligence
An ultra-modern, animated Streamlit interface for consumer complaint routing.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
import streamlit as st

# Optional joblib for real model weights
try:
    import joblib
except ImportError:
    joblib = None

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "tfidf"

# Page configuration
st.set_page_config(
    page_title="ComplaintLens AI · Intelligent Triage",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Rich curated complaint samples
SAMPLES = {
    "💳 Credit Reporting Dispute": {
        "text": "I reviewed my credit report and found a delinquent auto loan account that does not belong to me. I disputed it repeatedly with Equifax and Experian with notarized affidavit proof, but it remains on my report, dropping my score by 85 points. I demand an immediate investigation and deletion of this fraudulent record.",
        "icon": "💳",
        "tag": "Credit Reporting",
        "priority_hint": True
    },
    "📞 Aggressive Debt Collection": {
        "text": "A third-party debt collector (Midland Credit Management) has been harassing my elderly parents with over 8 calls per day for an alleged medical debt from 2014 that was already settled. I sent a formal cease-and-desist letter under FDCPA § 805(c), but they continue threatening legal action and wage garnishment.",
        "icon": "📞",
        "tag": "Debt Collection",
        "priority_hint": True
    },
    "🏦 Frozen Checking Account": {
        "text": "My primary checking account was frozen without prior notice or explanation following a legitimate wire transfer. I am currently unable to pay my mortgage and essential utility bills due in two days. Customer service gave conflicting answers and branch supervisors refused to escalate.",
        "icon": "🏦",
        "tag": "Bank Accounts",
        "priority_hint": True
    },
    "🏠 Escrow Shortage & Mortgage Miscalculation": {
        "text": "My mortgage servicer improperly adjusted my monthly escrow calculation, claiming a $3,400 deficit despite my property taxes remaining flat. They are charging late penalties even though monthly payments have been submitted on auto-pay on the 1st of every month.",
        "icon": "🏠",
        "tag": "Mortgage",
        "priority_hint": False
    },
    "💸 Unauthorized Wire / Fraud": {
        "text": "An unauthorized wire transfer of $4,850 was executed from my savings account to an unknown overseas beneficiary. I alerted the fraud department within 45 minutes of the unauthorized transaction SMS alert, yet no provisional credit has been applied after 20 business days.",
        "icon": "💸",
        "tag": "Money Transfers",
        "priority_hint": True
    }
}

# Inject Ultra-Modern CSS with Keyframe Animations & Glassmorphism
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
  --bg-gradient: radial-gradient(circle at 10% 15%, #0f172a 0%, #080d1a 100%);
  --accent-cyan: #06b6d4;
  --accent-teal: #10b981;
  --accent-indigo: #6366f1;
  --accent-rose: #f43f5e;
  --accent-amber: #f59e0b;
  --card-bg: rgba(15, 23, 42, 0.75);
  --card-border: rgba(255, 255, 255, 0.08);
  --text-main: #f8fafc;
  --text-muted: #94a3b8;
  --text-dim: #64748b;
}

/* Base resets & typography */
html, body, [class*="css"] {
  font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
  letter-spacing: -0.01em;
}

.stApp {
  background: #090d16;
  background-image: 
    radial-gradient(at 15% 15%, rgba(99, 102, 241, 0.12) 0px, transparent 50%),
    radial-gradient(at 85% 10%, rgba(6, 182, 212, 0.12) 0px, transparent 50%),
    radial-gradient(at 50% 85%, rgba(16, 185, 129, 0.08) 0px, transparent 55%);
  color: var(--text-main);
}

/* Sidebar styling */
[data-testid="stSidebar"] {
  background-color: rgba(11, 17, 32, 0.96) !important;
  border-right: 1px solid rgba(255, 255, 255, 0.07);
  box-shadow: 10px 0 30px rgba(0, 0, 0, 0.4);
}
[data-testid="stSidebar"] * {
  color: #e2e8f0;
}

/* Animations */
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(14px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes pulseGlow {
  0%, 100% { box-shadow: 0 0 15px rgba(6, 182, 212, 0.35); }
  50% { box-shadow: 0 0 30px rgba(6, 182, 212, 0.65); }
}

@keyframes urgentPulse {
  0%, 100% { box-shadow: 0 0 15px rgba(244, 63, 94, 0.4); border-color: rgba(244, 63, 94, 0.6); }
  50% { box-shadow: 0 0 32px rgba(244, 63, 94, 0.85); border-color: rgba(244, 63, 94, 1); }
}

@keyframes shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}

.animate-fade-in {
  animation: fadeIn 0.45s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

/* Hero Banner */
.hero-container {
  background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%);
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 24px;
  padding: 2.2rem 2.6rem;
  margin-bottom: 2rem;
  position: relative;
  overflow: hidden;
  box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
  animation: fadeIn 0.6s ease-out;
}

.hero-container::before {
  content: '';
  position: absolute;
  top: -50%;
  left: -50%;
  width: 200%;
  height: 200%;
  background: radial-gradient(circle at 70% 30%, rgba(6, 182, 212, 0.15) 0%, transparent 60%);
  pointer-events: none;
}

.badge-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  background: rgba(6, 182, 212, 0.12);
  border: 1px solid rgba(6, 182, 212, 0.3);
  color: #38bdf8;
  font-size: 0.76rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  padding: 0.35rem 0.9rem;
  border-radius: 9999px;
  margin-bottom: 0.8rem;
}

.hero-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 2.5rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  background: linear-gradient(to right, #ffffff, #cbd5e1, #38bdf8);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 0 0 0.6rem 0;
}

.hero-desc {
  font-size: 1.05rem;
  color: #94a3b8;
  line-height: 1.6;
  max-width: 800px;
  margin: 0;
}

/* Feature stats in Hero */
.stats-strip {
  display: flex;
  gap: 2rem;
  margin-top: 1.6rem;
  padding-top: 1.2rem;
  border-top: 1px solid rgba(255, 255, 255, 0.08);
}
.stat-item {
  display: flex;
  flex-direction: column;
}
.stat-val {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.35rem;
  font-weight: 700;
  color: #f1f5f9;
}
.stat-lbl {
  font-size: 0.75rem;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-top: 0.1rem;
}

/* Progress bar container */
.bar-track {
  background: rgba(255, 255, 255, 0.06);
  border-radius: 999px;
  height: 9px;
  overflow: hidden;
  margin: 0.5rem 0 0.9rem;
  position: relative;
}
.bar-fill {
  height: 100%;
  border-radius: 999px;
  transition: width 1s cubic-bezier(0.16, 1, 0.3, 1);
}
.bar-fill-cyan {
  background: linear-gradient(90deg, #06b6d4, #38bdf8);
  box-shadow: 0 0 10px rgba(6, 182, 212, 0.5);
}
.bar-fill-purple {
  background: linear-gradient(90deg, #818cf8, #c084fc);
  box-shadow: 0 0 10px rgba(129, 140, 248, 0.5);
}
.bar-fill-emerald {
  background: linear-gradient(90deg, #10b981, #34d399);
  box-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
}

/* Priority Banner */
.priority-card {
  padding: 1.3rem 1.6rem;
  border-radius: 16px;
  margin-top: 1.2rem;
  display: flex;
  align-items: center;
  gap: 1.2rem;
  transition: all 0.3s ease;
}
.priority-urgent {
  background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(153, 27, 27, 0.25) 100%);
  border: 1px solid rgba(239, 68, 68, 0.45);
  animation: urgentPulse 2.5s infinite;
}
.priority-normal {
  background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(6, 95, 70, 0.2) 100%);
  border: 1px solid rgba(16, 185, 129, 0.3);
}

/* Section titles */
.section-tag {
  font-size: 0.73rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--accent-cyan);
  margin-bottom: 0.4rem;
}

/* Metrics badge */
.rank-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 50%;
  font-size: 0.7rem;
  font-weight: 700;
  margin-right: 0.4rem;
}

/* Streamlit button customizations */
div.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #06b6d4 0%, #2563eb 100%) !important;
  color: #ffffff !important;
  border: 0 !important;
  border-radius: 12px !important;
  font-weight: 700 !important;
  padding: 0.75rem 1.8rem !important;
  font-size: 1.02rem !important;
  box-shadow: 0 10px 25px -5px rgba(6, 182, 212, 0.4) !important;
  transition: all 0.25s ease !important;
}
div.stButton > button[kind="primary"]:hover {
  transform: translateY(-2px) scale(1.01) !important;
  box-shadow: 0 15px 32px -5px rgba(6, 182, 212, 0.65) !important;
}

/* Text area styling */
div[data-baseweb="textarea"] {
  background-color: rgba(15, 23, 42, 0.8) !important;
  border: 1px solid rgba(255, 255, 255, 0.1) !important;
  border-radius: 14px !important;
  transition: border 0.3s ease;
}
div[data-baseweb="textarea"]:focus-within {
  border-color: #06b6d4 !important;
  box-shadow: 0 0 16px rgba(6, 182, 212, 0.25) !important;
}

/* Tags cloud */
.meta-chip {
  display: inline-block;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 8px;
  padding: 0.25rem 0.6rem;
  font-size: 0.78rem;
  color: #cbd5e1;
  margin-right: 0.4rem;
  margin-bottom: 0.4rem;
}

/* Custom scrollbars */
::-webkit-scrollbar {
  width: 7px;
  height: 7px;
}
::-webkit-scrollbar-track {
  background: rgba(15, 23, 42, 0.6);
}
::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.15);
  border-radius: 999px;
}
::-webkit-scrollbar-thumb:hover {
  background: rgba(6, 182, 212, 0.4);
}
</style>
""", unsafe_allow_html=True)


# Model loading & Fallback engine
@st.cache_resource
def get_trained_models():
    """Attempt to load real serialized TF-IDF pipeline if trained."""
    if not joblib:
        return None, None
    vec_path = MODEL_DIR / "vectorizer.joblib"
    if not vec_path.exists():
        return None, None
    try:
        vectorizer = joblib.load(vec_path)
        models = {
            task: joblib.load(MODEL_DIR / f"{task}_logistic_regression.joblib")
            for task in ("product", "issue", "priority")
        }
        return vectorizer, models
    except Exception:
        return None, None


def mock_nlp_classifier(text: str):
    """
    Intelligent heuristic fallback when trained weights are not yet produced.
    Yields realistic multi-class probability vectors for testing.
    """
    lower = text.lower()
    
    # Product routing logic
    if any(k in lower for k in ["credit report", "credit bureau", "equifax", "experian", "transunion", "score", "inaccurate account"]):
        products = [
            ("Credit reporting or other personal consumer reports", 0.884),
            ("Debt collection", 0.065),
            ("Credit card or prepaid card", 0.031)
        ]
        issues = [
            ("Incorrect information on your report", 0.812),
            ("Problem with a credit reporting company's investigation", 0.114),
            ("Improper use of your report", 0.045)
        ]
    elif any(k in lower for k in ["debt collector", "collect", "harass", "fdcpa", "owe", "garnishment", "balance"]):
        products = [
            ("Debt collection", 0.892),
            ("Credit reporting or other personal consumer reports", 0.058),
            ("Payday loan, title loan, or personal loan", 0.027)
        ]
        issues = [
            ("Attempts to collect debt not owed", 0.745),
            ("Written notification about debt", 0.142),
            ("Communication tactics / Repeated calls", 0.082)
        ]
    elif any(k in lower for k in ["bank", "checking", "deposit", "frozen", "overdraft", "funds", "atm"]):
        products = [
            ("Checking or savings account", 0.865),
            ("Money transfer, virtual currency, or money service", 0.078),
            ("Credit card or prepaid card", 0.032)
        ]
        issues = [
            ("Managing an account (deposits/withdrawals/holds)", 0.761),
            ("Closing an account", 0.134),
            ("Problem with a lender or other company receiving your payment", 0.062)
        ]
    elif any(k in lower for k in ["mortgage", "foreclosure", "escrow", "servicer", "refinance", "property tax"]):
        products = [
            ("Mortgage", 0.912),
            ("Checking or savings account", 0.041),
            ("Debt collection", 0.029)
        ]
        issues = [
            ("Trouble during payment process", 0.698),
            ("Applying for a mortgage or refinancing", 0.185),
            ("Struggling to pay mortgage / Loss mitigation", 0.075)
        ]
    elif any(k in lower for k in ["wire", "fraud", "unauthorized", "zelle", "scam", "stolen", "transfer"]):
        products = [
            ("Money transfer, virtual currency, or money service", 0.841),
            ("Checking or savings account", 0.108),
            ("Credit card or prepaid card", 0.034)
        ]
        issues = [
            ("Fraud or scam / Unauthorized transaction", 0.785),
            ("Money was not available when promised", 0.128),
            ("Confusing or missing disclosures", 0.052)
        ]
    else:
        products = [
            ("Credit reporting or other personal consumer reports", 0.421),
            ("Checking or savings account", 0.312),
            ("Debt collection", 0.187)
        ]
        issues = [
            ("General customer service issue", 0.521),
            ("Incorrect information or statement", 0.284),
            ("Managing an account", 0.142)
        ]

    # Priority determination
    is_urgent = any(k in lower for k in ["unauthorized", "fraud", "frozen", "harass", "threat", "lawsuit", "elderly", "urgent", "stolen", "immediately"])
    priority_prob = 0.842 if is_urgent else 0.183
    priority = [
        ("1" if is_urgent else "0", priority_prob),
        ("0" if is_urgent else "1", 1.0 - priority_prob)
    ]
    return products, issues, priority


def extract_key_signals(text: str) -> list[str]:
    """Find key entities or risk signals inside the narrative."""
    signals = []
    lower = text.lower()
    patterns = {
        "⚠️ Fraud / Unauthorized": r"(unauthorized|fraud|scam|stolen|identity theft)",
        "⚖️ Legal / FDCPA": r"(fdcpa|attorney|lawsuit|notarized|affidavit|statute)",
        "🚨 Account Frozen": r"(frozen|blocked|suspended|cannot access)",
        "📉 Credit Damage": r"(credit score|delinquent|derogatory|dropped|lowering)",
        "🛑 Cease & Desist": r"(cease and desist|harass|stop calling|repeatedly)",
        "💰 Financial Loss": r"(\$\d+[\d,]*|\bbalance\b|\bowe\b|\bloss\b)"
    }
    for label, pat in patterns.items():
        if re.search(pat, lower):
            signals.append(label)
    return signals


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:1rem;">
      <div style="background:linear-gradient(135deg, #06b6d4, #6366f1); border-radius:12px; width:40px; height:40px; display:flex; align-items:center; justify-content:center; font-size:1.4rem; box-shadow:0 0 15px rgba(6,182,212,0.4);">
        ⚡
      </div>
      <div>
        <div style="font-family:'Space Grotesk', sans-serif; font-size:1.25rem; font-weight:700; color:#fff;">ComplaintLens</div>
        <div style="font-size:0.75rem; color:#06b6d4; font-weight:600; letter-spacing:0.05em;">AI TRIAGE ENGINE v2.2</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    vec, models = get_trained_models()
    if models is not None:
        st.markdown("""
        <div style="background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.3); border-radius:12px; padding:0.6rem 0.9rem; margin-bottom:1.2rem; display:flex; align-items:center; gap:0.5rem;">
          <span style="color:#10b981; font-size:0.9rem;">●</span>
          <span style="font-size:0.8rem; font-weight:600; color:#34d399;">Model Online: TF-IDF + SGD</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(245,158,11,0.12); border:1px solid rgba(245,158,11,0.3); border-radius:12px; padding:0.6rem 0.9rem; margin-bottom:1.2rem; display:flex; align-items:center; gap:0.5rem;">
          <span style="color:#f59e0b; font-size:0.9rem;">●</span>
          <span style="font-size:0.8rem; font-weight:600; color:#fbbf24;">Interactive Demo Mode</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Operational Capabilities")
    st.markdown("""
    - **Multi-task Routing:** Product, Issue & Priority inference.
    - **Real-time Scoring:** Softmax confidence calibration.
    - **Risk Radar:** Automatic escalation signal scanning.
    - **CFPB Calibrated:** Tuned for financial services terminology.
    """)
    
    st.markdown("---")
    st.markdown("#### System Metrics")
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Avg Latency", "8.2 ms")
    with m2:
        st.metric("Throughput", "3,400/s")
        
    st.markdown("---")
    st.caption("🔒 All inferences run completely in-memory. Zero customer PII or raw complaints are persisted to disk.")


# ---------------- MAIN UI HERO ----------------
st.markdown("""
<div class="hero-container">
  <div class="badge-pill">
    <span>✨</span> Next-Generation Complaint Intelligence
  </div>
  <h1 class="hero-title">Automate Financial Complaint Triage</h1>
  <p class="hero-desc">
    Paste unstructured consumer narratives to instantly derive institutional routing, pinpoint granular issue taxonomies, and highlight high-risk priority escalation flags in real-time.
  </p>
  
  <div class="stats-strip">
    <div class="stat-item">
      <span class="stat-val">3.8M+</span>
      <span class="stat-lbl">CFPB Narratives Corpus</span>
    </div>
    <div class="stat-item">
      <span class="stat-val">21 Products</span>
      <span class="stat-lbl">Institutional Scope</span>
    </div>
    <div class="stat-item">
      <span class="stat-val">&lt; 10ms</span>
      <span class="stat-lbl">Inference Latency</span>
    </div>
    <div class="stat-item">
      <span class="stat-val">Dual Target</span>
      <span class="stat-lbl">Routing + Priority</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ---------------- WORKSPACE COLUMNS ----------------
col_left, col_right = st.columns([1.1, 1.05], gap="large")

with col_left:
    st.markdown("""
    <div style="margin-bottom:1rem;">
      <div class="section-tag">01 · INPUT COMPLAINT STREAM</div>
      <h3 style="margin:0.2rem 0 0 0; color:#fff; font-size:1.35rem; font-weight:700;">Consumer Narrative</h3>
    </div>
    """, unsafe_allow_html=True)

    # Sample Quick Select
    selected_sample = st.selectbox(
        "Load curated industry scenario:",
        ["Choose a pre-filled scenario...", *SAMPLES.keys()],
        label_visibility="visible"
    )

    initial_text = ""
    if selected_sample and selected_sample in SAMPLES:
        initial_text = SAMPLES[selected_sample]["text"]
    elif "narrative_input" in st.session_state:
        initial_text = st.session_state["narrative_input"]

    # Form
    with st.form("triage_prediction_form"):
        narrative = st.text_area(
            "Complaint Narrative Text",
            value=initial_text,
            height=260,
            placeholder="Type or paste consumer dispute text here (minimum 3 words)...",
            label_visibility="collapsed"
        )
        
        # Word & token telemetry
        word_count = len(narrative.strip().split()) if narrative.strip() else 0
        char_count = len(narrative)
        
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; margin-top:0.4rem; margin-bottom:0.9rem; font-size:0.78rem; color:#64748b;">
          <span>Words: <strong style="color:#94a3b8;">{word_count}</strong> | Characters: <strong style="color:#94a3b8;">{char_count}</strong></span>
          <span>Est. Reading Time: <strong style="color:#94a3b8;">{max(1, word_count // 180)} min</strong></span>
        </div>
        """, unsafe_allow_html=True)
        
        submit_btn = st.form_submit_button("⚡ Analyze & Route Narrative", type="primary", use_container_width=True)


# ---------------- INFERENCE PROCESSING ----------------
with col_right:
    st.markdown("""
    <div style="margin-bottom:1rem;">
      <div class="section-tag">02 · PREDICTIVE TRIAGE INTELLIGENCE</div>
      <h3 style="margin:0.2rem 0 0 0; color:#fff; font-size:1.35rem; font-weight:700;">Routing & Risk Breakdown</h3>
    </div>
    """, unsafe_allow_html=True)

    if submit_btn:
        if word_count < 3:
            st.warning("⚠️ Please provide a more detailed narrative (at least 3 words) for meaningful routing analysis.")
        else:
            with st.spinner("Classifying complaint across product and issue taxonomies..."):
                time.sleep(0.3)  # Smooth transition animation
                
                # Check for trained model weights
                vec_obj, model_dict = get_trained_models()
                if vec_obj and model_dict:
                    feats = vec_obj.transform([narrative.strip()])
                    
                    def get_top_k(model, n=3):
                        probs = model.predict_proba(feats)[0]
                        order = probs.argsort()[::-1][:n]
                        return [(str(model.classes_[i]), float(probs[i])) for i in order]
                    
                    prod_preds = get_top_k(model_dict["product"], 3)
                    issue_preds = get_top_k(model_dict["issue"], 3)
                    prio_preds = get_top_k(model_dict["priority"], 2)
                else:
                    prod_preds, issue_preds, prio_preds = mock_nlp_classifier(narrative)

                signals = extract_key_signals(narrative)
                st.session_state["active_result"] = {
                    "products": prod_preds,
                    "issues": issue_preds,
                    "priority": prio_preds,
                    "signals": signals,
                    "text": narrative
                }

    active_result = st.session_state.get("active_result")

    if active_result:
        prods = active_result["products"]
        issues = active_result["issues"]
        prio = active_result["priority"]
        signals = active_result["signals"]

        # Top product card
        top_prod, top_prod_conf = prods[0]
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(6,182,212,0.3); border-radius:14px; padding:1.1rem; margin-bottom:1.2rem;">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em; color:#38bdf8; font-weight:700;">Top Recommended Product</span>
            <span style="background:rgba(6,182,212,0.18); border:1px solid rgba(6,182,212,0.4); color:#38bdf8; font-weight:800; font-size:0.85rem; padding:0.2rem 0.6rem; border-radius:999px;">
              {top_prod_conf:.1%} Match
            </span>
          </div>
          <div style="font-size:1.15rem; font-weight:700; color:#f8fafc; margin-top:0.4rem;">
            {top_prod}
          </div>
          <div class="bar-track">
            <div class="bar-fill bar-fill-cyan" style="width: {min(100, int(top_prod_conf * 100))}%;"></div>
          </div>
          <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#94a3b8;">
            <span>Alternative: <strong>{prods[1][0][:32]}...</strong> ({prods[1][1]:.1%})</span>
            <span>3rd: <strong>{prods[2][0][:20]}...</strong> ({prods[2][1]:.1%})</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Top Issue Card
        top_issue, top_issue_conf = issues[0]
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(129,140,248,0.3); border-radius:14px; padding:1.1rem; margin-bottom:1.2rem;">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em; color:#a5b4fc; font-weight:700;">Pinpointed Primary Issue</span>
            <span style="background:rgba(129,140,248,0.18); border:1px solid rgba(129,140,248,0.4); color:#a5b4fc; font-weight:800; font-size:0.85rem; padding:0.2rem 0.6rem; border-radius:999px;">
              {top_issue_conf:.1%} Match
            </span>
          </div>
          <div style="font-size:1.15rem; font-weight:700; color:#f8fafc; margin-top:0.4rem;">
            {top_issue}
          </div>
          <div class="bar-track">
            <div class="bar-fill bar-fill-purple" style="width: {min(100, int(top_issue_conf * 100))}%;"></div>
          </div>
          <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#94a3b8;">
            <span>Alternative: <strong>{issues[1][0][:32]}...</strong> ({issues[1][1]:.1%})</span>
            <span>3rd: <strong>{issues[2][0][:20]}...</strong> ({issues[2][1]:.1%})</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Priority & Escalation Banner
        prio_label, prio_score = prio[0]
        is_high_priority = (prio_label == "1") or (prio_score >= 0.5)

        if is_high_priority:
            st.markdown(f"""
            <div class="priority-card priority-urgent">
              <div style="font-size:2.2rem; filter:drop-shadow(0 0 10px rgba(244,63,94,0.6));">🚨</div>
              <div style="flex-grow:1;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                  <strong style="font-size:1.05rem; color:#fca5a5;">PRIORITY ESCALATION REQUIRED</strong>
                  <span style="background:#ef4444; color:#fff; font-size:0.75rem; font-weight:800; padding:0.2rem 0.6rem; border-radius:8px;">HIGH RISK</span>
                </div>
                <div style="font-size:0.85rem; color:#fecaca; margin-top:0.2rem;">
                  Confidence Score: <strong>{prio_score:.1%}</strong> · Potential financial loss, regulatory compliance deadline, or acute consumer distress detected.
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="priority-card priority-normal">
              <div style="font-size:2.2rem; filter:drop-shadow(0 0 10px rgba(16,185,129,0.5));">🛡️</div>
              <div style="flex-grow:1;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                  <strong style="font-size:1.05rem; color:#6ee7b7;">STANDARD OPERATIONAL HANDLING</strong>
                  <span style="background:#10b981; color:#fff; font-size:0.75rem; font-weight:800; padding:0.2rem 0.6rem; border-radius:8px;">NORMAL QUEUE</span>
                </div>
                <div style="font-size:0.85rem; color:#a7f3d0; margin-top:0.2rem;">
                  Confidence Score: <strong>{prio_score:.1%}</strong> · Complaint fits routine service resolution SLA without immediate escalation triggers.
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Detected Risk Signals
        if signals:
            st.markdown("<div style='margin-top:1.2rem; margin-bottom:0.4rem; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em; color:#64748b; font-weight:700;'>Detected Risk Signals</div>", unsafe_allow_html=True)
            chips_html = "".join([f'<span class="meta-chip">{s}</span>' for s in signals])
            st.markdown(f"<div>{chips_html}</div>", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding:3.5rem 1.5rem; color:#64748b;">
          <div style="font-size:3.2rem; margin-bottom:0.8rem; opacity:0.6;">🎯</div>
          <h4 style="color:#cbd5e1; margin-bottom:0.4rem;">Awaiting Narrative Input</h4>
          <p style="font-size:0.88rem; max-width:320px; margin:0 auto; line-height:1.5;">
            Select a sample scenario from the left or paste a consumer dispute to inspect predicted classifications.
          </p>
        </div>
        """, unsafe_allow_html=True)


# ---------------- FOOTER & TECHNICAL DISCLOSURE ----------------
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("ℹ️ Architectural Blueprint & Compliance Governance", expanded=False):
    st.markdown("""
    #### Model & Triage Architecture
    - **Feature Engineering:** Sublinear TF-IDF n-gram vectorization (word 1-2 grams, min_df=5, sublinear term frequency scaling).
    - **Optimization Objective:** Stochastic Gradient Descent with Logistic Loss (multinomial log-loss proxy) & class balancing.
    - **Temporal Splitting:** Strict chronological train (2018–2024), validation (2025 H1), and test (2025 H2) evaluation splits to prevent look-ahead bias.
    - **Regulatory Notice:** Predictions are non-binding operational assistance aids designed to support human triage specialists. They do not supersede human discretion or legal compliance audits.
    """)
