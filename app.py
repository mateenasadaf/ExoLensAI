import json
import os
from pathlib import Path

import requests

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ARTIFACTS = Path("artifacts")
MODEL_PATH = ARTIFACTS / "rf_model.joblib"
FEATURES_PATH = ARTIFACTS / "features.json"
CANDIDATES_PATH = ARTIFACTS / "candidate_observations.csv"
KNOWLEDGE_PATH = ARTIFACTS / "knowledge_base.json"

FEATURE_INFO = {
    "koi_period": ("Orbital period", "days", "How long the object takes to orbit its star."),
    "koi_duration": ("Transit duration", "hours", "How long the object takes to pass in front of its star from our viewpoint."),
    "koi_depth": ("Transit depth", "source value", "How much the star's observed brightness drops during a transit."),
    "koi_prad": ("Planet radius", "Earth radii", "The estimated size compared with Earth."),
    "koi_teq": ("Estimated equilibrium temperature", "K", "A model-based temperature estimate, not necessarily a measured surface temperature."),
    "koi_insol": ("Insolation", "Earth-relative", "The amount of stellar energy received compared with Earth."),
    "koi_model_snr": ("Signal-to-noise ratio", "SNR value", "How clearly the transit signal stands out from measurement noise."),
    "koi_steff": ("Star temperature", "K", "The star's estimated effective temperature."),
    "koi_srad": ("Star radius", "solar radii", "The star's estimated size compared with the Sun."),
}

st.set_page_config(page_title="ExoLens AI", page_icon="🪐", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
.stApp { background: radial-gradient(ellipse at 15% 8%,rgba(66,54,130,.30),transparent 36%), radial-gradient(ellipse at 88% 28%,rgba(21,101,148,.20),transparent 35%), linear-gradient(145deg,#050816 0%,#0a1024 52%,#080b19 100%); color:#edf3ff; font-family:'DM Sans',sans-serif; }
.stApp:before,.stApp:after { content:""; position:fixed; inset:0; pointer-events:none; z-index:0; }
.stApp:before { opacity:.42; background-image:radial-gradient(1px 1px at 7% 16%,#fff 98%,transparent),radial-gradient(1.5px 1.5px at 19% 72%,#b8dfff 98%,transparent),radial-gradient(1px 1px at 35% 28%,#fff 98%,transparent),radial-gradient(1.5px 1.5px at 48% 84%,#d9d2ff 98%,transparent),radial-gradient(1px 1px at 63% 12%,#fff 98%,transparent),radial-gradient(1px 1px at 78% 62%,#c7eaff 98%,transparent),radial-gradient(1.5px 1.5px at 91% 31%,#fff 98%,transparent); background-size:420px 360px; animation:exoStars 90s linear infinite; }
.stApp:after { opacity:.16; background-image:radial-gradient(2px 2px at 25% 30%,#8ce8ff 98%,transparent),radial-gradient(2px 2px at 75% 70%,#b6a0ff 98%,transparent); background-size:560px 520px; animation:exoStars 130s linear infinite reverse; }
@keyframes exoStars { from {transform:translateY(0)} to {transform:translateY(-360px)} }
[data-testid="stHeader"] {background:rgba(5,8,22,.62)}
[data-testid="stAppViewContainer"] > .main {position:relative;z-index:1}
.block-container {padding-top:2.3rem;padding-bottom:3rem;max-width:1180px}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.025em}
h1 {font-size:3rem!important;background:linear-gradient(90deg,#fff,#9cecff 55%,#b9a5ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
h2 {color:#e7edff!important} p,label,.stMarkdown {color:#d8e1f5}
[data-testid="stCaptionContainer"] {color:#a8b4d1}
[data-testid="stExpander"] {background:rgba(14,22,46,.70);border:1px solid rgba(139,164,220,.18);border-radius:16px}
div[data-testid="stMetric"] {background:linear-gradient(135deg,rgba(25,37,73,.88),rgba(17,26,52,.82));border:1px solid rgba(114,228,255,.20);padding:16px 18px;border-radius:16px}
div[data-testid="stMetricLabel"] {color:#aebbd9} div[data-testid="stMetricValue"] {color:#f2f7ff}
.stButton>button {border-radius:12px;border:1px solid rgba(114,228,255,.38);background:linear-gradient(100deg,rgba(38,75,118,.92),rgba(76,58,132,.92));color:white;transition:transform .2s ease,box-shadow .2s ease}
.stButton>button:hover {transform:translateY(-1px);box-shadow:0 0 22px rgba(114,228,255,.18);border-color:#72e4ff;color:white}
.stSelectbox>div>div,.stTextInput input,[data-testid="stChatInput"] textarea {background:rgba(8,14,32,.88);border-color:rgba(139,164,220,.28);color:#edf3ff;border-radius:12px}
[data-testid="stChatMessage"] {background:rgba(17,26,52,.76);border:1px solid rgba(139,164,220,.14);border-radius:14px}
[data-testid="stDataFrame"] {border:1px solid rgba(139,164,220,.18);border-radius:12px;overflow:hidden} hr {border-color:rgba(139,164,220,.20)}
.exo-kicker {color:#8deaff;text-transform:uppercase;letter-spacing:.18em;font-size:.76rem;font-weight:700}
.exo-subtitle {color:#bdc9e4;font-size:1.05rem;line-height:1.7;max-width:900px}
.exo-glow {height:1px;background:linear-gradient(90deg,transparent,rgba(114,228,255,.6),rgba(167,139,250,.55),transparent);margin:1.3rem 0 1.8rem}
@media (prefers-reduced-motion:reduce) {.stApp:before,.stApp:after{animation:none}}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_features():
    return json.loads(FEATURES_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_candidates():
    return pd.read_csv(CANDIDATES_PATH)


@st.cache_data
def load_knowledge_base():
    return json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8"))


def predict_observation(model, row):
    probs = model.predict_proba(row)[0]
    classes = list(model.named_steps["classifier"].classes_)
    confirmed_index = classes.index(1)
    return int(model.predict(row)[0]), float(probs[confirmed_index])


def explain_with_shap(model_pipeline, row, feature_names):
    import shap

    imputer = model_pipeline.named_steps["imputer"]
    classifier = model_pipeline.named_steps["classifier"]
    transformed = imputer.transform(row)
    transformed_df = pd.DataFrame(transformed, columns=feature_names)
    explainer = shap.TreeExplainer(classifier)
    values = explainer.shap_values(transformed_df)

    if isinstance(values, list):
        contributions = np.asarray(values[1])[0]
    else:
        values = np.asarray(values)
        if values.ndim == 3:
            contributions = values[0, :, 1]
        elif values.ndim == 2:
            contributions = values[0]
        else:
            raise ValueError(f"Unexpected SHAP output shape: {values.shape}")

    result = pd.DataFrame({
        "Feature": feature_names,
        "Measurement": transformed_df.iloc[0].to_numpy(),
        "SHAP value": contributions,
    })
    return result.sort_values("SHAP value", key=lambda s: s.abs(), ascending=False)



def get_openrouter_key():
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    try:
        key = str(st.secrets.get("OPENROUTER_API_KEY", "")).strip()
        if key:
            return key
    except Exception:
        pass
    for env_path in (Path(".env"), Path("backend/.env")):
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("\"'")
    return ""


def answer_question(question, chat_history, candidate_context, observation_context, shap_context):
    """Use an LLM for open-ended, conversational answers."""
    api_key = get_openrouter_key()
    if not api_key:
        return (
            "The conversational AI isn't connected yet because `OPENROUTER_API_KEY` is missing. "
            "Add your key to a `.env` file in the project folder, then restart Streamlit."
        )

    system_prompt = f"""You are ExoLens AI's friendly astronomy assistant. Respond naturally like a helpful general-purpose conversational AI, not a rigid FAQ bot.
Users can ask open-ended questions about astronomy, the selected observation, machine learning, SHAP, or follow-up topics. Answer the actual question rather than restricting the user to predefined subjects.
Use plain English first; explain technical terms when needed. Keep answers concise but adapt to the question. Do not overuse the words CONFIRMED and FALSE POSITIVE when simpler phrasing such as "likely to be a real planet" or "possible false alarm" is clearer.
For this model, describe predictions as model estimates, not verified real-world chances or official scientific decisions. Never invent missing values or certainty.
When explaining SHAP, say it shows how a measurement influenced this particular model output. A SHAP value is not a percentage and is not a direct probability change or physical cause. If a source column has no clear physical unit, say its number is the value recorded in that dataset column; do not call it "dataset units" as if that were a physical unit.
Use the context only when relevant. If asked something unrelated, answer normally.
Selected observation context: {candidate_context}
Observation measurements: {observation_context}
SHAP contributions: {shap_context}
"""
    history = []
    for message in chat_history[-12:]:
        if message.get("role") in ("user", "assistant"):
            history.append({"role": message["role"], "content": message["content"]})
    history.append({"role": "user", "content": question})

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://localhost",
                "X-Title": "ExoLens AI",
            },
            json={
                "model": os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
                "messages": [{"role": "system", "content": system_prompt}] + history,
                "temperature": 0.5,
                "max_tokens": 800,
            },
            timeout=45,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.Timeout:
        return "The assistant took too long to respond. Please try again."
    except requests.exceptions.RequestException as exc:
        detail = ""
        try:
            detail = response.text[:400]
        except Exception:
            pass
        return f"I couldn't reach the AI service. Check your OpenRouter key and internet connection. {detail or str(exc)}"
    except (KeyError, IndexError, ValueError):
        return "The AI service returned an unexpected response. Please try again."


def set_current_observation(row, model):
    prediction, probability = predict_observation(model, row)
    st.session_state["current_row"] = row.copy()
    st.session_state["current_prediction"] = prediction
    st.session_state["current_probability"] = probability


def show_prediction(prediction, probability):
    label = "Higher likelihood of a real exoplanet" if prediction == 1 else "Possible false alarm"
    left, right = st.columns(2)
    left.metric("Model prediction", label)
    right.metric("Probability assigned to CONFIRMED class", f"{probability:.1%}")
    st.progress(probability)
    st.caption(
        "This is the model's output, not a guarantee or official scientific confirmation. "
        "Its real-world probability calibration has not been established here."
    )


def feature_name(feature):
    return FEATURE_INFO.get(feature, (feature, "", ""))[0]


required = [MODEL_PATH, FEATURES_PATH, CANDIDATES_PATH, KNOWLEDGE_PATH]
missing = [str(path) for path in required if not path.exists()]

def plot_theme_chart(fig):
    """Apply ExoLens AI's dark-space visual theme to Matplotlib charts."""
    fig.patch.set_facecolor("#0b1022")
    for ax in fig.axes:
        ax.set_facecolor("#0b1022")
        ax.tick_params(colors="#d8e5ff", labelsize=9)
        ax.xaxis.label.set_color("#b8c7e8")
        ax.yaxis.label.set_color("#b8c7e8")
        ax.title.set_color("#eff6ff")
        for spine in ax.spines.values():
            spine.set_color("#33456d")
        ax.grid(True, color="#344263", alpha=0.38, linestyle="--", linewidth=0.6)
    fig.tight_layout()
    return fig

st.markdown('<div class="exo-kicker">Exoplanet intelligence · Model-assisted exploration</div>', unsafe_allow_html=True)
st.title("ExoLens AI")
st.markdown(
    '<div class="exo-subtitle">Explore Kepler observations with a machine-learning model trained to distinguish records labelled '
    '<b>CONFIRMED</b> and <b>FALSE POSITIVE</b>. Inspect the measurements behind an observation, see how each feature influences '
    'the model through SHAP, and ask astronomy questions in one continuous workspace. ExoLens AI supports exploration; it does not '
    'replace scientific validation or official planet confirmation.</div><div class="exo-glow"></div>',
    unsafe_allow_html=True,
)

if missing:
    st.error(
        "Some project artifacts are missing. Check that these files exist:\n\n"
        + "\n".join(f"- `{path}`" for path in missing)
    )
    st.stop()

model = load_model()
features = load_features()
candidates = load_candidates()
knowledge_base = load_knowledge_base()

missing_features = [f for f in features if f not in candidates.columns]
if missing_features:
    st.error(
        "The candidate CSV is missing required model features: "
        + ", ".join(missing_features)
        + ". Re-export candidate_observations.csv from your Colab notebook."
    )
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "active_record_index" not in st.session_state:
    st.session_state["active_record_index"] = None

st.markdown("## 1. Choose an observation")
st.write("Search by Kepler star ID if you have one, or browse the available records.")

search_id = st.text_input(
    "Optional: search by Kepler star ID (kepid)",
    placeholder="Enter a star ID, or leave blank to browse",
    key="candidate_search",
)
candidate_view = candidates.copy()
if search_id.strip() and "kepid" in candidate_view.columns:
    candidate_view = candidate_view[
        candidate_view["kepid"].astype(str).str.contains(search_id.strip(), regex=False, na=False)
    ]

if candidate_view.empty:
    st.warning("No matching records. Try another star ID.")
    st.stop()

candidate_view = candidate_view.copy()
candidate_view["_record_index"] = candidate_view.index
labels = []
label_to_index = {}
for _, record in candidate_view.head(5000).iterrows():
    row_index = int(record["_record_index"])
    star_id = record.get("kepid", "Unknown")
    label = f"Dataset record {row_index + 1} · Kepler star ID {star_id}"
    labels.append(label)
    label_to_index[label] = row_index

selected_label = st.selectbox("Select an observation", labels, key="selected_observation")
selected_index = label_to_index[selected_label]
selected_row = candidates.loc[[selected_index], features]
record_changed = st.session_state["active_record_index"] != selected_index
if record_changed:
    st.session_state["active_record_index"] = selected_index
    st.session_state["chat_history"] = []
    st.session_state["selected_prediction"], st.session_state["selected_probability"] = predict_observation(
        model, selected_row
    )

# Show the chosen record immediately underneath the selector, on the same page.
st.markdown("---")
st.markdown("## 2. Your selected observation")
st.caption(f"Dataset record {selected_index + 1} · Kepler star ID {candidates.loc[selected_index].get('kepid', 'Unknown')}")

record_rows = []
for feature in features:
    name, unit, description = FEATURE_INFO.get(feature, (feature, "", ""))
    value = selected_row.iloc[0][feature]
    record_rows.append({
        "Measurement": name,
        "Value": "Missing in source data" if pd.isna(value) else f"{value:g}",
        "Unit": unit,
        "What it means": description,
    })
st.dataframe(pd.DataFrame(record_rows), width="stretch", hide_index=True)

st.markdown("---")
st.markdown("## 3. Prediction and explanation")
prediction = st.session_state["selected_prediction"]
probability = st.session_state["selected_probability"]
show_prediction(prediction, probability)

if prediction == 1:
    st.markdown("**In simple terms:** The measurements look more like those of a real exoplanet to the AI model.\\n\\nThis is a promising sign, but scientists would need more evidence to confirm it.")
else:
    st.markdown("**In simple terms:** The measurements look more like a possible false alarm than a real planet.\\n\\nThis does not prove there is no planet; it is only the model's assessment.")


# Compact prediction score chart, styled to match the space theme.
st.markdown("### Prediction at a glance")
probability_fig, probability_ax = plt.subplots(figsize=(8, 2.0))
scores = [probability, max(0.0, min(1.0, 1.0 - probability))]
labels = ["Real-planet pattern", "Alternative pattern"]
probability_ax.barh(labels, scores, color=["#62e6ff", "#b69cff"], height=0.5)
probability_ax.set_xlim(0, 1)
probability_ax.set_xlabel("Model score — not a verified real-world probability")
probability_ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
for i, value in enumerate(scores):
    probability_ax.text(min(value + 0.015, 0.88), i, f"{value:.1%}", va="center", color="#f3f6ff", fontsize=10, fontweight="bold")
plot_theme_chart(probability_fig)
st.pyplot(probability_fig, use_container_width=True)
plt.close(probability_fig)

contributions = None
try:
    contributions = explain_with_shap(model, selected_row, features)
    toward_confirmed = contributions[contributions["SHAP value"] > 0].sort_values(
        "SHAP value", ascending=False
    )
    toward_false_positive = contributions[contributions["SHAP value"] < 0].sort_values(
        "SHAP value", ascending=True
    )

    st.markdown("### Why did the model make this prediction?")
    main_group = toward_confirmed.head(3) if prediction == 1 else toward_false_positive.head(3)
    opposing_group = toward_false_positive.head(2) if prediction == 1 else toward_confirmed.head(2)
    main_direction = "toward CONFIRMED" if prediction == 1 else "toward FALSE POSITIVE"
    opposing_direction = "toward FALSE POSITIVE" if prediction == 1 else "toward CONFIRMED"

    if not main_group.empty:
        st.markdown(f"**Measurements pushing {main_direction}:**")
        for _, item in main_group.iterrows():
            feature = item["Feature"]
            name, unit, description = FEATURE_INFO.get(feature, (feature, "", "This measurement is used by the model."))
            unit_text = f" {unit}" if unit and unit not in ("source value", "SNR value") else ""
            value = item["Measurement"]
            st.markdown(
                f"- **{name}** ({value:g}{unit_text}) — {description} "
                f"SHAP contribution: `{item['SHAP value']:+.4f}`."
            )
    else:
        st.write("No measurements with contributions in that direction were returned.")

    if not opposing_group.empty:
        st.markdown(f"**Measurements pushing {opposing_direction}:**")
        for _, item in opposing_group.iterrows():
            feature = item["Feature"]
            name, unit, description = FEATURE_INFO.get(feature, (feature, "", "This measurement is used by the model."))
            unit_text = f" {unit}" if unit and unit not in ("source value", "SNR value") else ""
            value = item["Measurement"]
            st.markdown(
                f"- **{name}** ({value:g}{unit_text}) — {description} "
                f"SHAP contribution: `{item['SHAP value']:+.4f}`."
            )

    with st.expander("See all SHAP values"):
        display = contributions.copy()
        display["Feature"] = display["Feature"].map(feature_name)
        display["Measurement"] = display["Measurement"].round(4)
        display["SHAP value"] = display["SHAP value"].round(5)
        st.dataframe(display, width="stretch", hide_index=True)

except Exception as exc:
    st.warning(f"SHAP explanation could not be calculated: {exc}")

st.markdown("---")
st.markdown("## 4. Ask questions about this observation")
st.caption("Ask anything about this observation or astronomy. Follow-up questions use the conversation above.")

other_probability = max(0.0, min(1.0, 1.0 - probability))
candidate_context = (
    f"Selected observation: CSV record {selected_index + 1}; Kepler target ID "
    f"{candidates.loc[selected_index].get('kepid', 'Unknown')}. "
    f"Model score for the likely-real-exoplanet class: {probability:.1%}; "
    f"score for the alternative class: {other_probability:.1%}. "
    "These are model scores, not verified real-world chances."
)
measurement_parts = []
for feature in features:
    value = selected_row.iloc[0][feature]
    name, unit, _ = FEATURE_INFO.get(feature, (feature, "", ""))
    value_text = "missing" if pd.isna(value) else f"{value:g}"
    unit_text = f" {unit}" if unit and unit not in ("source value", "SNR value") else ""
    measurement_parts.append(f"{name}: {value_text}{unit_text}")
observation_context = "; ".join(measurement_parts)

shap_context = "Not available for this observation."
if contributions is not None:
    shap_parts = []
    for _, item in contributions.iterrows():
        name = FEATURE_INFO.get(item["Feature"], (item["Feature"], "", ""))[0]
        shap_parts.append(f"{name}: recorded value {item['Measurement']:g}, SHAP contribution {item['SHAP value']:+.4f}")
    shap_context = "; ".join(shap_parts)

for message in st.session_state["chat_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask anything…")
if question:
    st.session_state["chat_history"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            response = answer_question(
                question,
                st.session_state["chat_history"][:-1],
                candidate_context,
                observation_context,
                shap_context,
            )
        st.markdown(response)
    st.session_state["chat_history"].append({"role": "assistant", "content": response})


st.caption(
    "Note: `kepid` identifies a Kepler target star and can repeat; the selector distinguishes "
    "records by their row in the exported CSV."
)

st.markdown("---")
if prediction == 1:
    st.write("Overall, the AI sees signs that are consistent with a real exoplanet.")
    st.write("It is a useful lead for further study, not a scientific confirmation.")
else:
    st.write("Overall, the AI thinks this signal may be a false alarm.")
    st.write("More observations and scientific checks would be needed to know for sure.")
