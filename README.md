# 💪 FitTrack Pro — Data Analytics Dashboard

> **Individual Submission | Data Analytics – MGB | Project-Based Learning**

## 🎯 Business Idea

**FitTrack Pro** is an AI-powered subscription fitness & wellness SaaS platform targeting urban Indians aged 18–55.  
This dashboard demonstrates end-to-end data analytics on **synthetically generated** user data to validate the business model.

## 📊 What This App Does

| Step | Description | Marks |
|------|-------------|-------|
| 1 | Synthetic data generation (1,500 users, realistic distributions) | 10 |
| 2 | Data cleaning & transformation (de-dup, imputation, feature engineering) | 10 |
| 3 | Descriptive analytics & EDA (9 charts with business insights) | 30 |
| + | Correlation analysis & statistical testing | bonus |

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy on Streamlit Cloud

1. Fork / push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select this repo → set `app.py` as main file
4. Click **Deploy** — get your shareable link!

## 📁 File Structure

```
fittrack_analytics/
├── app.py            # Main Streamlit application
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## 🔑 Key Findings

- **Engagement score** is the strongest predictor of churn (r ≈ −0.55, p < 0.001)
- **Elite/Premium** plans contribute 70%+ of revenue despite fewer users
- **Mumbai & Delhi** are top revenue-generating cities
- Sessions per week vs calories burned shows very strong correlation (r ≈ +0.85)
