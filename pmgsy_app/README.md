# PMGSY Intelligent Classifier

Classifies rural road/bridge construction projects into the correct **PMGSY scheme**
(PMGSY-I, PMGSY-II, PMGSY-III, RCPLWEA, PM-JANMAN) using a **Random Forest ML model**
and provides a natural-language explanation via **IBM Granite AI**.

## Project Structure

```
pmgsy_app/
├── backend/
│   ├── train_model.py   # ML training – run once
│   └── app.py           # Flask API server
├── frontend/
│   └── index.html       # Single-page UI (open in browser)
├── models/              # Generated after training
│   ├── pmgsy_classifier.pkl
│   ├── label_encoder.pkl
│   └── model_meta.json
└── requirements.txt
PMGSY_DATASET.csv        # Dataset (in workspace root)
```

## Quick Start

### 1. Install dependencies
```bash
cd pmgsy_app
pip install -r requirements.txt
```

### 2. Train the ML model
```bash
python backend/train_model.py
```
This reads `PMGSY_DATASET.csv` (from workspace root), trains a Random Forest classifier,
and saves the model artifacts to `pmgsy_app/models/`.

### 3. Start the Flask server
```bash
python backend/app.py
```
Server runs at `http://localhost:5000`.

### 4. Open the frontend
Open `pmgsy_app/frontend/index.html` in your browser. No build step required.

## API Endpoints

| Method | Path          | Description                              |
|--------|---------------|------------------------------------------|
| POST   | `/predict`    | Predict scheme + AI explanation          |
| GET    | `/model-info` | Model accuracy, CV scores, feature importances |
| GET    | `/health`     | Liveness check                           |

### POST /predict — Request body
```json
{
  "NO_OF_ROAD_WORK_SANCTIONED": 50,
  "LENGTH_OF_ROAD_WORK_SANCTIONED": 120.5,
  "NO_OF_BRIDGES_SANCTIONED": 2,
  "COST_OF_WORKS_SANCTIONED": 85.5,
  "NO_OF_ROAD_WORKS_COMPLETED": 45,
  "LENGTH_OF_ROAD_WORK_COMPLETED": 108.2,
  "NO_OF_BRIDGES_COMPLETED": 2,
  "EXPENDITURE_OCCURED": 76.3,
  "NO_OF_ROAD_WORKS_BALANCE": 5,
  "LENGTH_OF_ROAD_WORK_BALANCE": 12.3,
  "NO_OF_BRIDGES_BALANCE": 0
}
```

### POST /predict — Response
```json
{
  "predicted_scheme": "PMGSY-I",
  "probabilities": {
    "PMGSY-I": 0.72,
    "PMGSY-II": 0.15,
    "PMGSY-III": 0.08,
    "RCPLWEA": 0.03,
    "PM-JANMAN": 0.02
  },
  "explanation": "This project is classified under PMGSY-I ...",
  "llm_error": null
}
```

## ML Features Used

**Raw features (11):** road works sanctioned/completed/balance, lengths, bridge counts, cost, expenditure

**Derived features (5):** completion rate, expenditure rate, cost per km, bridge ratio, balance ratio

## IBM Granite Integration

The `/predict` endpoint fetches an IAM token from IBM Cloud and calls the
`ibm/granite-4-h-small` chat model to generate a concise explanation of why
the project matches the predicted scheme.

Set environment variables to override credentials:
```bash
set WX_API_KEY=your-key
set WX_PROJECT_ID=your-project-id
```
