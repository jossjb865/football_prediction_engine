# ⚽ Football Prediction Engine (MLOps)

An advanced, automated machine learning pipeline and sports analytics engine designed to predict football match outcomes, calculate optimal betting value (expected value, Asian handicaps, parlays), and deliver real-time alerts via Telegram.

## 🚀 Overview

`football_prediction_engine` is a modular MLOps-driven framework built in Python. It ingests historical fixture data and real-time odds from sports APIs, processes features using robust machine learning models (XGBoost, CatBoost), evaluates predictive probabilities, and automates notifications for high-value betting opportunities.

---

## 🛠️ Tech Stack & Libraries

- **Language:** Python 3.10+
- **Machine Learning:** `xgboost`, `catboost`, `scikit-learn`
- **Data Manipulation & Analysis:** `pandas`, `numpy`
- **API Integration & Requests:** `requests`, `aiohttp`
- **Automation & Notifications:** Telegram Bot API (`python-telegram-bot` / custom webhooks)
- **MLOps & Pipeline:** Custom logging, model serialization (`joblib` / `pickle`), automated data ingestion scripts.

---

## 📁 Repository Structure

```tree
football_prediction_engine/
│
├── data/                  # Raw and processed datasets (CSV / Parquet)
├── models/                # Serialized machine learning models (.pkl / .cbm)
├── src/
│   ├── ingestion/         # Scripts for fetching data & odds from sports APIs
│   ├── features/          # Feature engineering (form guides, Elo ratings, xG metrics)
│   ├── models/            # Training, cross-validation, and prediction scripts (XGBoost/CatBoost)
│   ├── evaluation/        # Backtesting, ROI calculation, and bankroll management
│   └── notifications/     # Telegram alert dispatcher
│
├── notebooks/             # Exploratory Data Analysis (EDA) and prototyping
├── tests/                 # Unit and integration tests
├── .env.example           # Environment variables template
├── requirements.txt       # Project dependencies
└── main.py                # Main orchestration script
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/jossjb865/football_prediction_engine.git
cd football_prediction_engine
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Duplicate `.env.example` as `.env` and fill in your API keys and Telegram credentials:
```env
SPORTS_API_KEY=your_api_key_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

---

## 🚀 Usage

### Run Data Ingestion & Feature Engineering
```bash
python src/ingestion/fetch_fixtures.py
python src/features/build_features.py
```

### Train Models
To train the CatBoost and XGBoost classification models on historical match data:
```bash
python src/models/train.py
```

### Run Predictions & Send Alerts
To evaluate upcoming fixtures, compute value bets, and dispatch Telegram alerts:
```bash
python main.py
```

---

## 📈 Features & Capabilities

- **Automated Data Pipelines:** Daily ingestion of fixtures, team stats, and bookmaker odds.
- **Advanced Modeling:** Gradient boosting ensembles tuned for sports outcomes (Home/Draw/Away, Over/Under, Both Teams to Score).
- **Value Betting Calculator:** Computes implied probability vs. model probability to filter out negative EV bets.
- **Instant Telegram Dispatch:** Formatted markdown alerts sent directly to your phone when profitable market opportunities are identified.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/jossjb865/football_prediction_engine/issues).

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
