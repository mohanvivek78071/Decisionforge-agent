"""Interactive demo:  streamlit run app/streamlit_app.py   (pip install -e '.[app]')"""
import os

import pandas as pd
import streamlit as st

from decisionforge.data.store import DataStore
from decisionforge.data.synth import generate
from decisionforge.llm import get_llm
from decisionforge.orchestrator import Orchestrator

st.set_page_config(page_title="DecisionForge", page_icon="🧭", layout="wide")
st.title("🧭 DecisionForge")
st.caption("Ask a business question. Agents plan, investigate, audit their own claims, then recommend.")


@st.cache_resource
def get_store():
    return DataStore(generate())


provider = st.sidebar.selectbox("LLM provider", ["offline", "anthropic"], help="'offline' needs no API key")
if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
    st.sidebar.warning("Set ANTHROPIC_API_KEY to use Claude.")
examples = [
    "Why did revenue decline recently and where exactly is it coming from?",
    "Do promotions actually lift units, and are they worth the discount by category?",
    "Forecast Snacks units for the next 8 weeks",
    "Are there any data quality problems in the table?",
    "What is the CEO's salary?",
]
q = st.selectbox("Example questions", examples)
q = st.text_input("Your question", value=q)

if st.button("Investigate", type="primary") and q:
    with st.spinner("Agents at work..."):
        res = Orchestrator(get_store(), get_llm(provider)).run(q)
    m = res.memo
    c1, c2, c3 = st.columns(3)
    c1.metric("Confidence", m.confidence)
    c2.metric("Findings verified", f"{sum(v.status == 'verified' for v in res.verdicts)}/{len(m.findings)}")
    c3.metric("Tool calls", len(res.ledger))
    st.subheader("Answer")
    st.write(m.answer)
    if not m.abstained:
        st.subheader("Evidence-backed findings")
        status = {v.finding_index: v.status for v in res.verdicts}
        for i, f in enumerate(m.findings):
            st.markdown(f"{'✅' if status.get(i) == 'verified' else '❌'} {f.text}  `[{f.evidence_id}]`")
        st.subheader("Recommendation")
        st.info(m.recommendation)
        with st.expander("Risks & assumptions"):
            for r_ in m.risks:
                st.write("•", r_)
        with st.expander("Evidence ledger (every tool call)"):
            for e in res.ledger:
                st.markdown(f"**{e.id} · {e.tool}** — {e.summary}")
                st.json({"args": e.args, "ok": e.ok}, expanded=False)
        with st.expander("Agent trace"):
            st.dataframe(pd.DataFrame(res.trace).astype(str))
