import re
from email.utils import parseaddr, parsedate_to_datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

st.set_page_config(page_title="Phishing Campaign Behaviour Detection", page_icon="🛡️", layout="wide")

ROOT = Path(__file__).parent
DEFAULT_DATA = ROOT / "data" / "Nazario.csv"

STAGES = [
    "Normal / low evidence",
    "Initial phishing contact",
    "Credential or information harvesting",
    "Account compromise indicators",
    "Possible data exfiltration",
]
STAGE_RISK = {
    STAGES[0]: "Low",
    STAGES[1]: "Medium",
    STAGES[2]: "High",
    STAGES[3]: "Critical",
    STAGES[4]: "Critical",
}

# These transparent keyword rules create event observations from email metadata/content.
# They are feature engineering, not a trained email classifier.
PATTERNS = {
    "credential": r"\b(password|passwd|login|log in|sign.?in|verify your account|validate|credential|account access|reset your)\b",
    "urgent": r"\b(urgent|immediately|suspend(ed)?|blocked|expire(s|d)?|final reminder|action required|within 24 hours)\b",
    "financial": r"\b(bank|banking|payment|invoice|billing|refund|wallet|account update|credit card|debit card)\b",
    "delivery": r"\b(delivery|shipment|tracking|parcel|package|courier|fedex|ups|dhl)\b",
    "brand": r"\b(microsoft|outlook|paypal|amazon|apple|netflix|metamask|usaa|cpanel|google|facebook|meta)\b",
    "data": r"\b(download|document|shared file|attachment|transfer|upload|statement|tax form)\b",
}

def sender_domain(value):
    address = parseaddr(str(value))[1].strip().lower()
    if "@" in address:
        return address.rsplit("@", 1)[1]
    # Some corpus rows have malformed sender strings; use a stable fallback.
    match = re.search(r"@([A-Za-z0-9.-]+\.[A-Za-z]{2,})", str(value))
    return match.group(1).lower() if match else "unknown-domain"

def parse_email_date(value):
    if pd.isna(value) or not str(value).strip():
        return pd.NaT
    try:
        dt = parsedate_to_datetime(str(value).strip())
        if dt is None:
            return pd.NaT
        return pd.Timestamp(dt).tz_convert("UTC") if dt.tzinfo else pd.Timestamp(dt, tz="UTC")
    except Exception:
        return pd.NaT

def derive_features(df):
    work = df.copy()
    for col in ["sender", "receiver", "subject", "body"]:
        if col not in work:
            work[col] = ""
        work[col] = work[col].fillna("").astype(str)
    if "urls" not in work:
        work["urls"] = 0
    work["urls"] = pd.to_numeric(work["urls"], errors="coerce").fillna(0)
    work["parsed_date"] = work["date"].apply(parse_email_date) if "date" in work else pd.NaT
    work["sender_domain"] = work["sender"].map(sender_domain)
    work["text"] = (work["subject"] + " " + work["body"]).str.lower()
    for name, pattern in PATTERNS.items():
        work[f"has_{name}"] = work["text"].str.contains(pattern, regex=True, na=False)
    work["has_url"] = (work["urls"] > 0) | work["text"].str.contains(r"https?://|www\.", regex=True, na=False)
    work["feature_count"] = work[[f"has_{x}" for x in PATTERNS] + ["has_url"]].sum(axis=1)
    return work

def observation_for(row):
    # One discrete observation symbol per email; HMM then models symbol order per sender domain.
    flags = []
    for key in ["credential", "urgent", "financial", "delivery", "brand", "data"]:
        if bool(row.get(f"has_{key}", False)):
            flags.append(key)
    if bool(row.get("has_url", False)):
        flags.append("url")
    if not flags:
        return "no_keyword_or_url_signal"
    # Keep the vocabulary manageable and reproducible.
    return "+".join(flags)

def risk_score(row):
    score = 0
    score += 2 if row.get("has_credential", False) else 0
    score += 1 if row.get("has_urgent", False) else 0
    score += 1 if row.get("has_financial", False) else 0
    score += 1 if row.get("has_brand", False) else 0
    score += 1 if row.get("has_url", False) else 0
    score += 1 if row.get("has_data", False) else 0
    return int(score)

def stage_from_score(score):
    if score == 0:
        return STAGES[0]
    if score <= 2:
        return STAGES[1]
    if score <= 4:
        return STAGES[2]
    if score <= 5:
        return STAGES[3]
    return STAGES[4]

class DiscreteHMM:
    """Categorical HMM with scaled Forward/Backward, Baum-Welch and Viterbi."""
    def __init__(self, n_states, n_symbols, seed=42):
        rng = np.random.default_rng(seed)
        self.n, self.m = n_states, n_symbols
        self.pi = np.full(n_states, 1.0 / n_states)
        self.A = rng.random((n_states, n_states)) + 0.1
        self.A /= self.A.sum(axis=1, keepdims=True)
        self.B = rng.random((n_states, n_symbols)) + 0.1
        self.B /= self.B.sum(axis=1, keepdims=True)

    def forward(self, obs):
        T = len(obs)
        alpha = np.zeros((T, self.n))
        scales = np.zeros(T)
        alpha[0] = self.pi * self.B[:, obs[0]]
        scales[0] = max(alpha[0].sum(), 1e-300)
        alpha[0] /= scales[0]
        for t in range(1, T):
            alpha[t] = (alpha[t-1] @ self.A) * self.B[:, obs[t]]
            scales[t] = max(alpha[t].sum(), 1e-300)
            alpha[t] /= scales[t]
        return alpha, scales, float(np.log(scales).sum())

    def backward(self, obs, scales):
        T = len(obs)
        beta = np.ones((T, self.n))
        for t in range(T-2, -1, -1):
            beta[t] = self.A @ (self.B[:, obs[t+1]] * beta[t+1])
            beta[t] /= max(scales[t+1], 1e-300)
        return beta

    def fit(self, sequences, iterations=35, smoothing=1e-3):
        history = []
        for _ in range(iterations):
            pi_sum = np.zeros(self.n)
            a_num = np.zeros((self.n, self.n))
            a_den = np.zeros(self.n)
            b_num = np.zeros((self.n, self.m))
            b_den = np.zeros(self.n)
            total_ll = 0.0
            for obs in sequences:
                if len(obs) == 0:
                    continue
                alpha, scales, ll = self.forward(obs)
                beta = self.backward(obs, scales)
                total_ll += ll
                gamma = alpha * beta
                gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)
                pi_sum += gamma[0]
                for t, symbol in enumerate(obs):
                    b_num[:, symbol] += gamma[t]
                    b_den += gamma[t]
                for t in range(len(obs)-1):
                    xi = self.A * np.outer(alpha[t], self.B[:, obs[t+1]] * beta[t+1])
                    xi /= max(xi.sum(), 1e-300)
                    a_num += xi
                    a_den += gamma[t]
            self.pi = (pi_sum + smoothing) / (pi_sum.sum() + self.n*smoothing)
            self.A = a_num + smoothing
            self.A /= self.A.sum(axis=1, keepdims=True)
            self.B = b_num + smoothing
            self.B /= self.B.sum(axis=1, keepdims=True)
            history.append(total_ll)
        return history

    def viterbi(self, obs):
        T = len(obs)
        log_a, log_b = np.log(self.A + 1e-300), np.log(self.B + 1e-300)
        dp = np.full((T, self.n), -np.inf)
        back = np.zeros((T, self.n), dtype=int)
        dp[0] = np.log(self.pi + 1e-300) + log_b[:, obs[0]]
        for t in range(1, T):
            candidates = dp[t-1][:, None] + log_a
            back[t] = np.argmax(candidates, axis=0)
            dp[t] = np.max(candidates, axis=0) + log_b[:, obs[t]]
        path = np.zeros(T, dtype=int)
        path[-1] = int(np.argmax(dp[-1]))
        for t in range(T-2, -1, -1):
            path[t] = back[t+1, path[t+1]]
        return path

    def posterior(self, obs):
        alpha, scales, ll = self.forward(obs)
        beta = self.backward(obs, scales)
        post = alpha * beta
        post /= np.maximum(post.sum(axis=1, keepdims=True), 1e-300)
        return post, ll

def align_states(model, symbols):
    # Heuristic semantic alignment based on expected feature severity.
    symbol_severity = []
    for sym in symbols:
        bits = set(sym.split("+"))
        value = 0
        value += 2 if "credential" in bits else 0
        value += 1 if "urgent" in bits else 0
        value += 1 if "financial" in bits else 0
        value += 1 if "brand" in bits else 0
        value += 1 if "url" in bits else 0
        value += 1 if "data" in bits else 0
        symbol_severity.append(value)
    expected = model.B @ np.asarray(symbol_severity)
    order = np.argsort(expected)
    mapping = {int(state): STAGES[min(rank, len(STAGES)-1)] for rank, state in enumerate(order)}
    return mapping

st.title("🛡️ Intelligent Phishing Campaign Behaviour Detection")
st.caption("Nazario email dataset • Discrete HMM • Baum–Welch • Forward–Backward • Sum-Product • Viterbi")

with st.sidebar:
    st.header("Dataset and model")
    uploaded = st.file_uploader("Upload another CSV (optional)", type=["csv"])
    iterations = st.slider("Baum–Welch iterations", 5, 100, 35, 5)
    group_mode = st.selectbox("Sequence grouping", ["Sender domain", "Recipient", "Single chronological mailbox"])
    st.caption("The bundled Nazario dataset is loaded automatically when no CSV is uploaded.")

if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
        source_name = uploaded.name
    except Exception as e:
        st.error(f"Unable to read CSV: {e}")
        st.stop()
elif DEFAULT_DATA.exists():
    raw = pd.read_csv(DEFAULT_DATA)
    source_name = "data/Nazario.csv"
else:
    st.error("Dataset not found. Place Nazario.csv inside the data folder or upload a CSV.")
    st.stop()

required = ["sender", "receiver", "date", "subject", "body", "urls", "label"]
missing = [c for c in required if c not in raw.columns]
if missing:
    st.error("This application expects the Nazario dataset columns: " + ", ".join(required) + ". Missing: " + ", ".join(missing))
    st.stop()

data = derive_features(raw)
data["observation"] = data.apply(observation_for, axis=1)
data["heuristic_score"] = data.apply(risk_score, axis=1)
data["heuristic_stage"] = data["heuristic_score"].map(stage_from_score)
data["label"] = pd.to_numeric(data["label"], errors="coerce")

# Keep rows with dates; the RFC-style parser handles timezone offsets and comments in this corpus.
usable = data.dropna(subset=["parsed_date"]).copy()
if usable.empty:
    st.error("No valid dates could be parsed. Check the date column.")
    st.stop()

if group_mode == "Sender domain":
    usable["sequence_id"] = usable["sender_domain"]
elif group_mode == "Recipient":
    usable["sequence_id"] = usable["receiver"].replace("", "(missing recipient)")
else:
    usable["sequence_id"] = "All emails in mailbox"

usable = usable.sort_values(["sequence_id", "parsed_date"]).reset_index(drop=True)
sequence_frames = [(str(cid), group.copy()) for cid, group in usable.groupby("sequence_id", sort=False)]
# The HMM expects chronological observations. Single-email sequences are retained for emissions,
# but naturally contribute no transition counts.
symbols = sorted(usable["observation"].unique().tolist())
symbol_to_id = {s:i for i,s in enumerate(symbols)}
encoded_sequences = [[symbol_to_id[x] for x in frame["observation"].tolist()] for _, frame in sequence_frames]

label_counts = data["label"].value_counts(dropna=False).sort_index()
total = len(data)
parsed_count = len(usable)
unique_campaigns = len(sequence_frames)
url_count = int(data["has_url"].sum())
domain_count = int(data["sender_domain"].nunique())

st.subheader("Dataset overview")
m1,m2,m3,m4,m5 = st.columns(5)
m1.metric("Email records", f"{total:,}")
m2.metric("Usable timestamps", f"{parsed_count:,}")
m3.metric("Sender domains", f"{domain_count:,}")
m4.metric("Emails with URLs", f"{url_count:,}")
m5.metric("Sequences", f"{unique_campaigns:,}")

with st.expander("Dataset preview and label distribution", expanded=False):
    st.write(f"**Loaded from:** `{source_name}`")
    st.dataframe(data[["sender","receiver","date","subject","urls","label","sender_domain","observation","heuristic_score"]].head(20), use_container_width=True)
    st.write("**Label counts**")
    st.dataframe(label_counts.rename_axis("label").reset_index(name="records"), use_container_width=True)
    st.write("**Data quality**")
    quality = pd.DataFrame({
        "Check": ["Missing receiver", "Missing subject", "Missing date", "Date parsed successfully", "Duplicate rows"],
        "Count": [int(raw["receiver"].isna().sum()), int(raw["subject"].isna().sum()),
                  int(raw["date"].isna().sum()), int(usable.shape[0]), int(raw.duplicated().sum())]
    })
    st.dataframe(quality, hide_index=True, use_container_width=True)

st.subheader("Dataset finding")
if data["label"].nunique(dropna=True) == 1:
    only_label = data["label"].dropna().iloc[0]
    st.warning(f"All {len(data):,} records have label = {only_label}. This dataset has only one label class, so it cannot produce a valid phishing-vs-legitimate accuracy, precision, recall, F1, or ROC-AUC evaluation.")
else:
    st.info("More than one label class was found. Check the label meaning and class definitions before evaluation.")

st.markdown("### Feature engineering")
st.write("The application derives interpretable event signals from the email subject and body: credential-related terms, urgency, financial language, delivery language, brand names, data-related terms, and URL presence. These are used to create categorical HMM observations.")
feature_summary = pd.DataFrame({
    "Signal": ["Credential", "Urgent", "Financial", "Delivery", "Brand mention", "Data-related", "URL present"],
    "Emails": [int(data[f"has_{k}"].sum()) for k in ["credential","urgent","financial","delivery","brand","data"]] + [url_count]
})
st.plotly_chart(px.bar(feature_summary, x="Signal", y="Emails", title="Derived email signals"), use_container_width=True)

st.markdown("### Train the sequential model")
st.write("Campaign sequences are formed by grouping emails using the selected key and sorting each group by parsed timestamp. The categorical HMM is trained with Baum–Welch. Forward–Backward calculates posterior state probabilities; Viterbi decodes the most likely hidden-state path.")
if st.button("Train HMM and analyse campaigns", type="primary", use_container_width=True):
    st.session_state["nazario_train"] = True
if not st.session_state.get("nazario_train", False):
    st.info("Click the button to train the HMM on the selected sequence grouping.")
    st.stop()

with st.spinner("Training HMM..."):
    hmm = DiscreteHMM(n_states=5, n_symbols=len(symbols))
    history = hmm.fit(encoded_sequences, iterations=iterations)
    paths = [hmm.viterbi(seq) for seq in encoded_sequences]
    posteriors = [hmm.posterior(seq)[0] for seq in encoded_sequences]
    state_to_stage = align_states(hmm, symbols)

rows = []
for (seq_id, frame), path, posterior in zip(sequence_frames, paths, posteriors):
    decoded_stages = [state_to_stage[int(s)] for s in path]
    last_stage = decoded_stages[-1]
    rows.append({
        "Sequence": seq_id,
        "Emails": len(frame),
        "Predicted final stage": last_stage,
        "Risk": STAGE_RISK[last_stage],
        "Final posterior confidence": float(posterior[-1, path[-1]]),
    })
campaign_results = pd.DataFrame(rows).sort_values(["Risk","Emails"], ascending=[False,False])

st.success("HMM training and sequence decoding completed.")
tab1, tab2, tab3, tab4 = st.tabs(["Campaign dashboard", "HMM probabilities", "Evaluation & limitations", "Project methodology"])

with tab1:
    a,b = st.columns([1.4,1])
    with a:
        st.dataframe(campaign_results, hide_index=True, use_container_width=True,
                     column_config={"Final posterior confidence": st.column_config.ProgressColumn("Final posterior confidence", min_value=0, max_value=1, format="%.1%%")})
    with b:
        risk_counts = campaign_results["Risk"].value_counts().rename_axis("Risk").reset_index(name="Sequences")
        st.plotly_chart(px.bar(risk_counts, x="Risk", y="Sequences", title="Sequences by inferred risk"), use_container_width=True)
    chosen = st.selectbox("Select a sequence to investigate", campaign_results["Sequence"].tolist())
    idx = [x[0] for x in sequence_frames].index(chosen)
    frame = sequence_frames[idx][1].copy()
    path, posterior = paths[idx], posteriors[idx]
    frame["event_number"] = np.arange(1, len(frame)+1)
    frame["decoded_state"] = [state_to_stage[int(s)] for s in path]
    frame["posterior_confidence"] = [float(posterior[t, path[t]]) for t in range(len(path))]
    st.markdown(f"**Sequence:** `{chosen}` · **Emails:** {len(frame)} · **Final inferred stage:** {frame['decoded_state'].iloc[-1]}")
    st.dataframe(frame[["event_number","parsed_date","sender","subject","urls","observation","decoded_state","posterior_confidence"]],
                 hide_index=True, use_container_width=True)
    st.plotly_chart(px.line(frame, x="event_number", y="posterior_confidence", markers=True,
                            title="Viterbi state posterior confidence"), use_container_width=True)
    st.caption("A sequence grouping key is an approximation of a campaign. Shared sender domains do not prove that messages belong to one coordinated campaign.")

with tab2:
    st.markdown("**Transition matrix A** — probability of moving between latent states.")
    st.dataframe(pd.DataFrame(hmm.A, index=[f"State {i}" for i in range(5)], columns=[f"State {i}" for i in range(5)]).style.format("{:.3f}"), use_container_width=True)
    st.plotly_chart(px.imshow(hmm.A, text_auto=".2f", aspect="auto", title="HMM transition probabilities",
                              labels={"x":"Next state","y":"Current state","color":"Probability"}), use_container_width=True)
    st.markdown("**Emission matrix B** — probability of each engineered observation under each latent state.")
    emission = pd.DataFrame(hmm.B, columns=symbols, index=[f"State {i}" for i in range(5)])
    st.dataframe(emission.style.format("{:.3f}"), use_container_width=True)
    st.plotly_chart(px.imshow(hmm.B, x=symbols, y=[f"State {i}" for i in range(5)], aspect="auto",
                              title="HMM emission probabilities", labels={"x":"Observation symbol","y":"Hidden state","color":"Probability"}), use_container_width=True)
    st.line_chart(pd.DataFrame({"Log likelihood":history}, index=np.arange(1,len(history)+1)))
    st.caption("Latent states are assigned human-readable stage names using a heuristic severity score over their learned emission distributions. This is not a semantic guarantee.")

with tab3:
    st.markdown("### What can and cannot be evaluated with this file")
    st.write(f"Observed label values: `{sorted(data['label'].dropna().unique().tolist())}`")
    st.write("Because the uploaded file contains only one label class, it is not possible to calculate meaningful binary classification metrics. The app therefore does not fabricate accuracy, precision, recall or F1.")
    st.write("The current model is an unsupervised sequence model. To evaluate stage decoding accuracy, a separate manually annotated set of campaign-stage labels is needed.")
    st.markdown("### Recommended evaluation for the final report")
    st.markdown("""
1. Obtain a legitimate-email dataset with compatible fields and documented labels, or a labelled multi-class campaign-stage dataset.
2. Define a reproducible event-feature mapping before model training.
3. Split chronologically into training and test periods to avoid future-to-past leakage.
4. Fit HMM parameters on training sequences only.
5. Align hidden states using training data only, then evaluate on held-out sequences.
6. Compare with a simple baseline and report class distribution, precision, recall, F1 and confusion matrix.
""")
    st.warning("The stage names shown in the dashboard are heuristic interpretations of latent states. Do not describe them as verified ground truth.")

with tab4:
    st.markdown("""
### Model formulation
- **Observation sequence:** each email becomes a categorical symbol derived from its metadata and text signals.
- **Hidden state:** an unobserved behavioural stage in the email sequence.
- **Initial probabilities (π):** distribution over the first hidden state.
- **Transition matrix (A):** probability of moving from one hidden state to another.
- **Emission matrix (B):** probability of observing an engineered email-event symbol in a hidden state.
- **MLE / Baum–Welch:** iteratively estimates HMM parameters using the Expectation-Maximization procedure.
- **Forward algorithm:** calculates the likelihood of an observed sequence.
- **Backward algorithm:** calculates the likelihood of future observations from each state.
- **Sum-Product:** combines forward and backward messages to obtain posterior state probabilities.
- **Scaling factors:** normalize forward messages to prevent numerical underflow.
- **Viterbi algorithm:** finds the most likely hidden-state sequence.

### Dataset-specific preprocessing
The original columns are `sender`, `receiver`, `date`, `subject`, `body`, `urls`, and `label`. Email dates are parsed with an RFC-style email date parser. Messages are sorted chronologically within the selected grouping. Missing or unparseable dates are excluded from sequence inference and counted in the data-quality section.

### Limitations
This is an academic prototype, not a production phishing detector. Keyword rules can miss context and produce false positives. Sender-domain grouping is only a proxy for campaigns. The supplied file has a single label class. Use only data you are authorized to process and avoid publishing personal email addresses or private message bodies.
""")

st.divider()
st.caption("Academic ML prototype • HMM-based sequential analysis • Human review required • Do not treat model output as proof of malicious activity")
