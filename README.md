# 🚀 AI Brand Strategist — Social Media Engagement Forecasting

![Flask Web App](https://github.com/nabilamir1/Social_Media_Engagement_Forecasting/blob/main/Screenshot%202025-11-30%20193732.png?raw=true)

**AI Brand Strategist** is an end-to-end Deep Learning solution designed to empower marketing teams. It leverages **Long Short-Term Memory (LSTM)** networks to analyze historical engagement patterns across global brands (Nike, Apple, Google, Adidas) and predict future trends with high accuracy.

---

## 📂 Dataset

We utilized the **Social Media Engagement Dataset**, processing daily metrics to train our time-series models.

👉 **[Download Dataset Here]([LINK_TO_DATASET](https://www.kaggle.com/datasets/subashmaster0411/social-media-engagement-dataset))**

| Feature | Description |
| :--- | :--- |
| **Brands** | Nike, Google, Apple, Adidas |
| **Input** | Historical Engagement Rates (14-Day Sliding Window) |
| **Target** | Next Day's Engagement Rate |
| **NLP Features** | Text Sentiment Labels (Positive/Negative) |

---

## 🎯 Project Goals

- **Forecast Engagement:** Predict next-day engagement rates.
- **Sentiment Analysis:** Analyze audience emotions to detect potential crises.
- **Strategic Insights:** Recommend the "Golden Time" (Best Day of Week) to post.
- **Competitive Benchmarking:** Real-time comparison of competitors.
- **Deployment:** Accessible via a **Flask Web Dashboard** and a **Desktop App**.

---

## 🧠 Model Architecture

We implemented a robust **LSTM** architecture suitable for capturing long-term dependencies in volatile social media data.

- **Layer 1:** LSTM (50 units) + Dropout (0.2)
- **Layer 2:** LSTM (50 units) + Dropout (0.2)
- **Output:** Dense Layer (Regression)
- **Loss Function:** Mean Squared Error (MSE)

---

## 📊 Results Summary

The model was trained individually for each brand to ensure tailored accuracy.

| Brand | RMSE (Root Mean Squared Error) | Trend Accuracy |
| :--- | :---: | :---: |
| **Nike** | 0.39 | ✅ High |
| **Apple** | 0.42 | ✅ High |
| **Google** | 0.45 | ✅ Medium |
| **Adidas** | 0.38 | ✅ High |

---

## 👥 Team Members (Class of 2025)

| Name | Role |
| :--- | :--- |
| **Nabil Amir** 
| **Salah Eldin Mostafa** 
| **Fares Mohamed** 
| **Abdelrahman Mohamed Shokry** 

---

## 🚀 Quick Start

1. **Clone Repository**
   ```bash
   git clone [https://github.com/YourUsername/AI-Brand-Strategist.git](https://github.com/YourUsername/AI-Brand-Strategist.git)
