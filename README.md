# 🪐 ExoLens AI
### Explainable Exoplanet Classification and Astronomy Assistant

ExoLens AI is an interactive machine learning application built using NASA's Kepler Objects of Interest (KOI) dataset. It predicts whether an astronomical observation is more consistent with a confirmed exoplanet or a false positive, explains predictions using SHAP (SHapley Additive exPlanations), and provides an AI-powered astronomy assistant through OpenRouter.

> **Disclaimer:** ExoLens AI is an educational and exploratory project. Its predictions are not official NASA confirmations, and model scores should not be interpreted as verified real-world probabilities.

---

## 🚀 Live Demo and Source Code

| Resource | Link |
|---|---|
| 🌐 Live Application | [Launch ExoLens AI](https://exolensai-mateena-sadaf.streamlit.app/) |
| 💻 GitHub Repository | [View Source Code](https://github.com/mateenasadaf/ExoLensAI) |
| 🖥️ Local Application | `http://localhost:8501` |

**Note:** Streamlit Community Cloud may put inactive applications to sleep. If a wake-up screen appears, wait for the application to start.

---

## 📌 1. Project Overview

An observation that resembles an exoplanet transit is not necessarily evidence of a planet. Similar signals can arise from eclipsing binary stars, instrumental effects, or other astronomical phenomena.

ExoLens AI uses supervised machine learning to classify observations using selected astronomical measurements. It combines classification, explainable AI, interactive data exploration, and a large language model (LLM) assistant in one application.

### Project Objectives

- Classify observations into **CONFIRMED** and **FALSE POSITIVE** categories.
- Browse and search observations using the Kepler target identifier (`kepid`).
- Display the predicted class and its associated model score.
- Explain individual predictions using SHAP feature contributions.
- Visualize astronomical measurements and model explanations.
- Answer astronomy-related questions using an LLM integrated through OpenRouter.

---

## ✨ 2. Key Features

### 2.1 Exoplanet Classification

A trained Random Forest classifier predicts whether a selected observation belongs to the CONFIRMED or FALSE POSITIVE class.

The application presents the model output as an exploratory classification result, not as an official scientific determination.

### 2.2 Observation Explorer

Users can browse available candidate observations and search using a Kepler target ID (`kepid`).

Since one target star can have multiple observation records, the application distinguishes individual records using their row in the exported CSV.

### 2.3 Explainable AI Using SHAP

ExoLens AI uses SHAP's `TreeExplainer` to explain the Random Forest's predictions.

The explanation interface allows users to:

- Inspect features that push a prediction toward either class.
- View individual feature contributions.
- Explore the complete SHAP contribution table.
- Understand which input measurements influenced a particular prediction.

**Important:** SHAP values explain the model's behavior. They are not percentages, direct probability changes, or proof of physical causation.

### 2.4 AI-Powered Astronomy Assistant

The astronomy chatbot uses the OpenRouter Chat Completions API to generate natural-language responses.

It can answer general astronomy questions and use relevant context from the selected observation, its measurements, and available SHAP contributions.

The model is configurable through the `OPENROUTER_MODEL` environment variable. The current code defaults to:

`openai/gpt-4o-mini`

The classifier and chatbot have separate responsibilities:

- **Random Forest:** Produces the classification.
- **SHAP:** Explains the classifier's output.
- **OpenRouter LLM:** Generates conversational explanations and answers.

LLM-generated information may be inaccurate and should be checked against reliable scientific sources.

### 2.5 Interactive Dashboard

The Streamlit dashboard provides an astronomy-themed interface for:

- Browsing candidate observations.
- Viewing predictions and associated scores.
- Exploring feature contributions.
- Interacting with the astronomy chatbot.
- Visualizing astronomical data.

---

## 🧠 3. Algorithms and Technical Approach

| Component | Technology | Purpose |
|---|---|---|
| Primary classifier | Random Forest | Classifies astronomical observations |
| Comparison model | Logistic Regression | Provides a baseline for model comparison |
| Missing-value handling | Scikit-learn imputation pipeline | Handles missing feature values during inference |
| Explainable AI | SHAP TreeExplainer | Estimates individual feature contributions |
| Conversational AI | OpenRouter API | Generates astronomy-related responses |
| Web interface | Streamlit | Provides an interactive dashboard |
| Model serialization | Joblib | Loads the saved model pipeline |

### Random Forest

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees. It can learn nonlinear relationships between structured astronomical measurements.

### Logistic Regression

Logistic Regression was used as a baseline model in earlier experiments to compare its classification performance with Random Forest.

### Missing-Value Imputation

The saved scikit-learn pipeline includes an imputer before the classifier. This helps apply the same missing-value preprocessing during inference as in the deployed model pipeline.

### SHAP Explainability

SHAP estimates how individual input features contribute to a model output relative to a reference value. This makes individual predictions easier to inspect and interpret.

---

## 📊 4. Dataset Description

ExoLens AI uses NASA's **Kepler Objects of Interest (KOI)** dataset.

The dataset contains astronomical observations and catalog dispositions that serve as labels for supervised classification.

### 4.1 Dataset Statistics

The following values were recorded during earlier project experiments. They have not been independently recalculated from the deployed application.

| Statistic | Recorded Value |
|---|---:|
| Total observations | 7,587 |
| Total columns | 11 |
| Training observations | 6,041 |
| Testing observations | 1,546 |
| Confirmed observations in training set | 2,161 |
| Confirmed observations in test set | 587 |
| False-positive observations overall | 4,839 |
| Confirmed observations overall | 2,748 |
| Observations with missing feature values | 1,773 |
| Unique star identifiers | 6,641 |

The dataset contains more false-positive observations than confirmed observations. This class imbalance should be considered when evaluating model performance.

### 4.2 Dataset Features

| Column | Description |
|---|---|
| `kepid` | Kepler target-star identifier; excluded from model inputs |
| `koi_disposition` | Catalog disposition and target label |
| `koi_period` | Estimated orbital period, in days |
| `koi_duration` | Estimated transit duration, in hours |
| `koi_depth` | Recorded transit depth |
| `koi_prad` | Estimated planetary radius, in Earth radii |
| `koi_teq` | Estimated equilibrium temperature, in kelvin |
| `koi_insol` | Estimated insolation relative to Earth |
| `koi_model_snr` | Transit-model signal-to-noise ratio |
| `koi_steff` | Estimated host-star effective temperature, in kelvin |
| `koi_srad` | Estimated host-star radius, in solar radii |

The deployed application reads the authoritative model feature list from `artifacts/features.json`. The target label and star identifier should not be used as model inputs.

---

## 📈 5. Model Performance

The following accuracy values were recorded during earlier project experiments.

| Model | Recorded Accuracy |
|---|---:|
| Logistic Regression | 79.30% |
| Random Forest | 91.66% |

Random Forest achieved the higher recorded accuracy and was selected as the primary classifier for the application.

**Evaluation note:** These figures are historical experiment results, not newly reproduced metrics. Refer to the original notebook for the exact train-test split and evaluation procedure.

Because the dataset is imbalanced, accuracy alone is insufficient to establish model quality. A more complete evaluation should include:

- Precision
- Recall
- F1-score
- Confusion matrix
- Probability calibration, where applicable

---

## 🏗️ 6. System Architecture

The application combines a local machine learning pipeline with an external LLM API.

```text
                  User
                    |
                    v
           Streamlit Dashboard
                    |
          +---------+----------+
          |                    |
          v                    v
   Observation Data       Astronomy Chatbot
          |                    |
          v                    v
   Feature Selection       OpenRouter API
          |                    |
          v                    v
    Random Forest          Configured LLM
          |
          v
    Class Prediction
          |
          v
      SHAP Explainer
          |
          v
   Prediction Explanation
          |
          v
    Results and Charts
```

The Random Forest classifier produces predictions locally using saved model artifacts. The chatbot separately sends questions and relevant context to OpenRouter.

---

## 🗂️ 7. Repository Structure

```text
ExoLensAI/
│
├── .github/
│   └── workflows/
│       └── GitHub Actions workflow files
│
├── artifacts/
│   ├── rf_model.joblib
│   ├── features.json
│   ├── candidate_observations.csv
│   └── knowledge_base.json
│
├── app.py
├── requirements.txt
├── SETUP.md
├── README.md
└── .gitignore
```

Preserve the `artifacts/` directory and the exact filenames expected by `app.py`.

### 7.1 File-by-File Explanation

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit application. Loads artifacts, supports observation selection, performs predictions, displays SHAP explanations and charts, and handles chatbot interactions. |
| `requirements.txt` | Lists the Python dependencies required to run the project. |
| `SETUP.md` | Describes how to export the trained model and supporting artifacts and run the application locally. |
| `artifacts/rf_model.joblib` | Serialized scikit-learn pipeline containing missing-value preprocessing and the trained Random Forest classifier. |
| `artifacts/features.json` | Stores the ordered feature names expected by the model. |
| `artifacts/candidate_observations.csv` | Stores candidate measurements and exported prediction information displayed by the application. |
| `artifacts/knowledge_base.json` | Contains reference information used to explain features and provide context. |
| `.github/workflows/` | Contains GitHub Actions workflow configuration, if configured. |
| `.gitignore` | Specifies files and folders Git should exclude, such as local environments and secrets. |

The application dependencies include Streamlit, Pandas, NumPy, scikit-learn, Matplotlib, Joblib, SHAP, PyArrow, and Bottleneck. The repository pins `scikit-learn==1.6.1`.

**Artifact compatibility:** Use the matching model artifacts and dependencies. A model saved under a different scikit-learn version may produce compatibility warnings or errors.

---

## 💻 8. Run ExoLens AI Locally in VS Code

### Prerequisites

- Python installed. Python 3.12 was used in recent deployment troubleshooting.
- Visual Studio Code.
- Git, if cloning the repository.
- All four expected files inside `artifacts/`.
- An OpenRouter API key for the chatbot.

The classifier and SHAP explanations use the saved artifacts. The chatbot requires internet access and a working API key.

### Step 1: Clone the Repository

Open VS Code, select **Terminal → New Terminal**, and run:

```bash
git clone https://github.com/mateenasadaf/ExoLensAI.git
cd ExoLensAI
```

If the repository is already on your computer, open its existing folder instead.

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use the environment's Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip --version
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 4: Configure OpenRouter

Create a file named `.env` in the project root, alongside `app.py`.

Add the following configuration:

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=openai/gpt-4o-mini
```

Replace `your_openrouter_api_key` with your actual key.

Keep `.env` private and ensure it is excluded from Git. Model availability, pricing, and free usage depend on OpenRouter and may change.

### Step 5: Verify the Model Artifacts

Confirm that these files exist:

```text
artifacts/rf_model.joblib
artifacts/features.json
artifacts/candidate_observations.csv
artifacts/knowledge_base.json
```

If any file is missing, follow `SETUP.md` to export the required artifacts from the original notebook or obtain the matching files from the repository.

### Step 6: Start the Application

Run:

```bash
python -m streamlit run app.py
```

Open the local address shown in the terminal:

**http://localhost:8501**

To stop the application, press `Ctrl + C` in the terminal.

---

## 🔑 9. OpenRouter Configuration for Deployment

For Streamlit Community Cloud:

1. Open the application in Streamlit Community Cloud.
2. Go to the app's settings.
3. Open **Secrets**.
4. Add the following configuration:

```toml
OPENROUTER_API_KEY = "your_openrouter_api_key"
OPENROUTER_MODEL = "openai/gpt-4o-mini"
```

5. Save the configuration and restart or rerun the app if required.

Never publish the actual API key in your README, source code, screenshots, or GitHub repository.

---

## ☁️ 10. Deployment

**Live application:** [ExoLens AI on Streamlit Community Cloud](https://exolensai-mateena-sadaf.streamlit.app/)

**Source repository:** [GitHub — ExoLensAI](https://github.com/mateenasadaf/ExoLensAI)

The application is hosted on Streamlit Community Cloud and connected to its GitHub repository. Changes pushed to the configured deployment branch can trigger an application rebuild.

### Streamlit Sleep and Wake-Up Behavior

Streamlit Community Cloud can put free applications to sleep after inactivity.

- A wake-up screen does not necessarily indicate a broken application.
- Wait for the application to finish starting.
- A configured GitHub Actions workflow may periodically access the deployment.
- Scheduled requests do not guarantee uninterrupted uptime.

Review the actual workflow file in `.github/workflows/` to understand its schedule and behavior.

---

## 🧪 11. Testing Checklist

Use this checklist after setting up the application or deploying a new version.

- [ ] The application starts without missing-artifact errors.
- [ ] Candidate observations load successfully.
- [ ] Searching by `kepid` returns the expected records.
- [ ] Selecting a record displays the predicted class.
- [ ] The confirmed-class score is displayed correctly.
- [ ] The SHAP explanation renders successfully.
- [ ] The complete SHAP contribution table is available.
- [ ] The chatbot answers an astronomy-related question.
- [ ] The chatbot can answer a question about the selected observation.
- [ ] The missing-API-key case is handled appropriately.
- [ ] The application works after deployment or restart.

For formal model evaluation, reproduce the held-out evaluation and calculate precision, recall, F1-score, and a confusion matrix. Ensure preprocessing is consistent between training and inference.

---

## 🔐 12. Security Considerations

- Never commit `.env` files or API keys.
- Store secrets in local environment configuration or Streamlit Cloud Secrets.
- Do not hard-code API keys in `app.py`.
- If a key is accidentally exposed, revoke it and generate a replacement.
- Avoid including secrets in screenshots, logs, or public documentation.

---

## ⚠️ 13. Limitations

- Predictions are exploratory model outputs, not official NASA confirmations.
- Model scores are not verified real-world probabilities.
- SHAP explanations describe model behavior and do not establish physical causation.
- LLM-generated answers may contain incorrect scientific information.
- Class imbalance can make accuracy misleading.
- Results depend on dataset quality, catalog labels, preprocessing, and evaluation design.
- The chatbot requires internet connectivity and valid OpenRouter API access.
- Streamlit Community Cloud may sleep after inactivity.

---

## 🔭 14. Future Improvements

- Add a dedicated model-evaluation page with precision, recall, F1-score, and a confusion matrix.
- Compare classification algorithms using the same evaluation split.
- Evaluate probability calibration and document the validation methodology.
- Add reproducible data-preparation scripts and dataset provenance.
- Expand candidate filtering and interactive data visualization.
- Add automated tests for artifact loading, prediction generation, SHAP output, and missing API-key handling.
- Improve the astronomy chatbot's grounding with authoritative scientific references.
- Add richer visual explanations of astronomical measurements and model behavior.

---

## 📚 15. References

- [NASA Exoplanet Archive — Kepler Objects of Interest](https://exoplanetarchive.ipac.caltech.edu/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenRouter API Documentation](https://openrouter.ai/docs)

---

## 🖼️ 16. Project Screenshots

The following section is reserved for screenshots of the deployed ExoLens AI application. Add your screenshots to a folder named `screenshots/` in the repository and update the filenames below to match the images you upload.

### 16.1 Main Dashboard

<!-- Replace the filename with your actual screenshot -->
![ExoLens AI Main Dashboard](screenshots/main-dashboard.png)

*The main interface for exploring astronomical observations and interacting with ExoLens AI.*

### 16.2 Exoplanet Classification

![Exoplanet Classification Results](screenshots/classification-results.png)

*An example classification showing the predicted category and associated model score.*

### 16.3 SHAP Explainability

![SHAP Feature Explanation](screenshots/shap-explanation.png)

*Feature contributions that help explain an individual Random Forest prediction.*

### 16.4 Astronomy Chatbot

![ExoLens AI Astronomy Chatbot](screenshots/astronomy-chatbot.png)

*The LLM-powered astronomy assistant answering questions about astronomical observations.*

### 16.5 Interactive Visualizations

![ExoLens AI Visualizations](screenshots/visualizations.png)

*Interactive views of astronomical measurements, prediction results, and model explanations.*

---

## 🛠️ Project Information

| Category | Details |
|---|---|
| Project Name | ExoLens AI |
| Domain | Machine Learning and Astronomy |
| Dataset | NASA Kepler Objects of Interest (KOI) |
| Primary Model | Random Forest |
| Comparison Model | Logistic Regression |
| Explainability | SHAP TreeExplainer |
| Conversational AI | OpenRouter API |
| Frontend | Streamlit |
| Model Pipeline | Scikit-learn and Joblib |
| Language | Python |

**Built as an educational project exploring machine learning, explainable AI, and conversational AI for astronomical data analysis.**


