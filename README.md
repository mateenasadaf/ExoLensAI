
ExoLens AI 🪐
Explainable Exoplanet Classification and Astronomy Assistant
ExoLens AI is an interactive machine-learning application built around
NASA's Kepler Objects of Interest (KOI) data. It predicts whether an
observation is more consistent with a confirmed exoplanet or a
false positive, explains individual predictions with SHAP, and
provides an astronomy assistant powered by an LLM through OpenRouter.

Important: This is an educational and exploratory project. A model
prediction is not an official exoplanet confirmation or a verified
real-world probability.

🚀 Live Demo and Source Code
Live application: https://exolensai-mateena-sadaf.streamlit.app/
GitHub repository: https://github.com/mateenasadaf/ExoLensAI
Local address: http://localhost:8501
Streamlit Community Cloud may put apps to sleep after inactivity. A
wake-up screen may appear before the app starts.

📌 Project Overview
A transit-like signal detected by a telescope is not automatically a
planet. It can be caused by a real planet, an eclipsing binary,
instrumental effects, or other phenomena. The KOI dataset contains
observations and catalog dispositions that can be used to train a
supervised classifier.

ExoLens AI combines machine learning, explainable AI, interactive
visualization, and an LLM-based astronomy assistant.

Objectives
Classify Kepler observations into CONFIRMED and FALSE POSITIVE.
Browse and search candidate observations by Kepler target ID
(kepid).
Display the prediction and score assigned to the CONFIRMED class.
Explain individual predictions using SHAP feature contributions.
Answer questions about astronomy and the selected observation
through OpenRouter.
✨ Features
1. Exoplanet classification
A trained Random Forest classifier predicts the category for a selected
observation. The interface describes the output as a higher likelihood
of a real exoplanet or a possible false alarm.

2. Observation explorer
Users can browse candidate records or search by kepid. A target ID can
occur in multiple rows, so the app distinguishes records by their row in
the exported CSV.

3. Explainable AI with SHAP
The app uses SHAP's TreeExplainer to estimate how individual input
features contribute to a selected model output. Users can inspect
features pushing the output toward either class and view the complete
SHAP table.

A SHAP value is not a percentage, a direct probability change, or proof
of physical causation. It explains the model's behavior for one
prediction.

4. Astronomy chatbot
The app sends the user's question and relevant context about the
selected observation, measurements, and available SHAP contributions to
the OpenRouter chat-completions API. The configured model can be set
with OPENROUTER_MODEL; the current code defaults to
openai/gpt-4o-mini.

The chatbot is separate from the Random Forest: the classifier produces
predictions, while the LLM produces natural-language responses. LLM
responses can be inaccurate and should be checked against reliable
scientific sources.

5. Interactive visualizations
The Streamlit dashboard displays candidate measurements, the prediction
score, and SHAP explanations in a dark, astronomy-themed interface.

🧠 Algorithms and Technical Approach
Random Forest (primary model): An ensemble of decision trees
used for supervised classification of structured astronomical data.
It can learn nonlinear relationships between measurements.
Logistic Regression (comparison model): A baseline model used in
earlier experiments.
Missing-value imputation: The deployed scikit-learn pipeline
includes an imputer before the classifier. Keeping preprocessing in
the saved pipeline helps maintain consistent inference.
SHAP TreeExplainer: Produces local feature-contribution
explanations for the Random Forest output.
OpenRouter API: Routes chatbot questions to the configured LLM.
📊 Dataset and Recorded Numbers
The project's previously recorded working-dataset summary is:

Statistic Recorded value

Total observations 7,587
Total columns 11
Training observations 6,041
Testing observations 1,546
Confirmed observations in training set 2,161
Confirmed observations in test set 587
False-positive observations overall 4,839
Confirmed observations overall 2,748
Observations with missing feature values 1,773
Unique star identifiers 6,641

These figures are from the earlier project record and are not
recalculated by the deployed app. The target classes are imbalanced,
with more false-positive records than confirmed records.

Dataset columns
Column Meaning

kepid Kepler target-star identifier; not
intended as a model feature

koi_disposition Catalog disposition/target label

koi_period Estimated orbital period, in days

koi_duration Estimated transit duration, in
hours

koi_depth Recorded transit depth

koi_prad Estimated planetary radius, in
Earth radii

koi_teq Estimated equilibrium temperature,
in kelvin

koi_insol Estimated insolation relative to
Earth

koi_model_snr Transit-model signal-to-noise ratio

koi_steff Estimated effective temperature of
the host star, in kelvin

koi_srad Estimated host-star radius, in
solar radii
The deployed app reads its authoritative model feature list from
artifacts/features.json. The target label and identifier should not be
used as model inputs.

Previously recorded model results
Model Recorded accuracy

Logistic Regression 79.30%
Random Forest 91.66%

These are earlier experiment results, not newly reproduced metrics in
this README. Consult the original notebook for the exact split and
evaluation procedure. Accuracy alone is insufficient for an imbalanced
dataset; precision, recall, F1-score, and a confusion matrix should also
be evaluated.

🗂️ Repository Structure
The repository's top-level items and the app's expected artifacts are:

ExoLensAI/
├── .github/
│   └── workflows/              # GitHub Actions workflow configuration
├── artifacts/
│   ├── rf_model.joblib         # Saved scikit-learn pipeline
│   ├── features.json           # Ordered model feature names
│   ├── candidate_observations.csv
│   │                            # Observation data and exported prediction fields
│   └── knowledge_base.json     # Reference information used by the app
├── app.py                      # Main Streamlit application
├── requirements.txt             # Python dependencies
├── SETUP.md                     # Artifact-export and local-run instructions
└── README.md                    # Project documentation
Preserve the artifacts/ folder and the exact filenames expected by
app.py.

File-by-file explanation
app.py --- main entry point. Loads artifacts, provides
observation search/selection, runs predictions, creates SHAP
explanations and charts, and handles the OpenRouter chat interface.
requirements.txt --- dependencies include Streamlit, Pandas,
NumPy, scikit-learn, Matplotlib, Joblib, SHAP, PyArrow, and
Bottleneck. The repository pins scikit-learn==1.6.1.
SETUP.md --- explains how to export the trained model and
supporting files from the notebook and run the dashboard locally.
artifacts/rf_model.joblib --- serialized scikit-learn
pipeline, including missing-value imputation and the trained
classifier.
artifacts/features.json --- feature names and order expected
by the model.
artifacts/candidate_observations.csv --- candidate
measurements and exported prediction information displayed by the
app. The setup guide creates it by joining candidates with
prediction outputs using kepid.
artifacts/knowledge_base.json --- reference content used to
explain features and provide context to the app.
.github/workflows/ --- GitHub Actions workflow configuration.
Review the actual workflow file for its schedule and behavior;
scheduled requests do not guarantee uninterrupted Streamlit uptime.
💻 Run Locally in VS Code (Windows)
Prerequisites
Python installed (Python 3.12 was used in recent deployment
troubleshooting).
VS Code.
Git, if cloning the repository.
All four expected files inside artifacts/.
An OpenRouter API key for the chatbot. The local classifier and SHAP
explanations use saved artifacts; the chatbot needs internet access
and API access.
Step 1 --- Clone the repository
Open Terminal → New Terminal in VS Code and run:

git clone https://github.com/mateenasadaf/ExoLensAI.git
cd ExoLensAI
If you already have the project locally, open its folder instead of
cloning it again.

Step 2 --- Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
If PowerShell blocks activation, run the environment's Python directly:

.\.venv\Scripts\python.exe -m pip --version
Step 3 --- Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Step 4 --- Configure OpenRouter
Create a .env file in the project root, alongside app.py:

OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=openai/gpt-4o-mini
Replace the placeholder with your actual API key. Do not commit .env
or expose the key publicly. The app checks environment variables,
Streamlit secrets, and .env files for the key. Model availability and
any free usage are controlled by OpenRouter and may change.

For Streamlit Community Cloud, configure the key under the app's
Settings → Secrets:

OPENROUTER_API_KEY = "your_openrouter_api_key"
OPENROUTER_MODEL = "openai/gpt-4o-mini"
Step 5 --- Verify artifacts
Confirm these paths exist:

artifacts/rf_model.joblib
artifacts/features.json
artifacts/candidate_observations.csv
artifacts/knowledge_base.json
If files are missing, the app cannot load correctly. Follow SETUP.md
to export them from the original notebook or obtain the matching
artifacts from the repository, preserving their folder structure.

Step 6 --- Start Streamlit
python -m streamlit run app.py
Open the URL shown in the terminal, normally:

http://localhost:8501
Stop the server with Ctrl+C in the terminal.

☁️ Deployment
The hosted app is available at:

https://exolensai-mateena-sadaf.streamlit.app/

The app is hosted on Streamlit Community Cloud and linked to the GitHub
repository. Pushing changes to the configured branch allows the hosted
app to rebuild.

Sleep and wake-up behavior
Free Streamlit apps can sleep after inactivity. A wake-up screen does
not necessarily mean the app is broken; wait for startup to finish. A
GitHub Actions workflow may periodically access the deployment if
configured to do so, but this does not guarantee uninterrupted uptime.

🔐 Security
Never commit .env or API keys.
Store keys in local environment configuration or Streamlit Cloud
Secrets.
If a key is accidentally published, revoke it and create a
replacement.
Do not hard-code secrets in app.py.
🧪 Basic Testing Checklist
Confirm the app loads without missing-artifact errors.
Browse records and search for a known kepid.
Select an observation and verify the prediction and confirmed-class
score appear.
Check that the SHAP explanation and complete contribution table
render.
Ask the chatbot a question about the selected observation.
Ask a general astronomy question.
Test the missing-key message if needed, then restore the key.
For formal evaluation, calculate accuracy, precision, recall, F1-score,
and a confusion matrix on a held-out dataset. Keep preprocessing
consistent between training and inference.

⚠️ Limitations
Model scores are not verified real-world probabilities.
Predictions are not official NASA decisions or scientific
confirmation.
SHAP explains model contributions; it does not prove physical
causation.
The chatbot can generate incorrect information.
Class imbalance can make accuracy misleading.
Results depend on dataset quality, labels, feature processing, and
evaluation design.
The chatbot requires an internet connection and a working OpenRouter
key.
🔭 Future Improvements
Add a model-evaluation page with precision, recall, F1-score, and a
confusion matrix.
Compare models using the same data split and preprocessing.
Evaluate probability calibration and document the validation method.
Add reproducible data-preparation scripts and dataset provenance.
Expand candidate filters and data visualization.
Add automated tests for artifact loading, predictions, SHAP, and
missing API-key handling.
📚 References
NASA Exoplanet Archive --- Kepler Objects of
Interest
Scikit-learn documentation
SHAP documentation
Streamlit documentation
OpenRouter API documentation
Project type: Machine Learning · Explainable AI · Astronomy · LLM
Integration
Primary interface: Streamlit
Primary classifier: Random Forest
Comparison model: Logistic Regression
Explainability: SHAP
Conversational AI: OpenRouter

⋮⋮⋮⋮
⋮⋮⋮⋮
