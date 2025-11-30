from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# --- Configuration ---
LOOKBACK_DAYS = 14
BRANDS_LIST = ["Nike", "Google", "Apple", "Adidas"]

try:
    MAIN_DF = pd.read_csv('Social Media Engagement Dataset.csv')
    MAIN_DF['timestamp'] = pd.to_datetime(MAIN_DF['timestamp'])
except:
    print("Warning: Main Dataset not found!")

# --- Helper Functions (نفس الدوال السابقة بالظبط) ---
def get_plot_url(brand_df, brand_name, pred_val=None, comp_df=None, comp_name=None):
    plt.figure(figsize=(10, 5))
    subset = brand_df.tail(45).reset_index(drop=True)
    plt.plot(subset.index, subset['engagement_rate'], label=f"{brand_name}", color='#0984e3', linewidth=3)
    
    if pred_val:
        last_idx = len(subset) - 1
        plt.plot(last_idx + 1, pred_val, marker='*', markersize=20, color='#00cec9', label='AI Forecast', zorder=5)

    if comp_df is not None:
        comp_subset = comp_df.tail(45).reset_index(drop=True)
        plt.plot(comp_subset.index, comp_subset['engagement_rate'], label=f"{comp_name} (Comp)", color='#fab1a0', linewidth=2, linestyle='--')

    plt.title(f"Engagement Trend: {brand_name}")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

def get_extras(brand):
    subset = MAIN_DF[MAIN_DF['brand_name'] == brand]
    avg_per_day = subset.groupby('day_of_week')['engagement_rate'].mean()
    best_day = avg_per_day.idxmax()

    recent = subset.tail(100)
    counts = recent['sentiment_label'].value_counts()
    total = len(recent)
    pos_pct = int((counts.get('Positive', 0) / total) * 100)
    neg_pct = int((counts.get('Negative', 0) / total) * 100)
    neu_pct = 100 - pos_pct - neg_pct

    return best_day, pos_pct, neg_pct, neu_pct

# --- Route (صفحة واحدة فقط) ---
@app.route('/', methods=['GET', 'POST'])
def home():
    context = {
        "brands": BRANDS_LIST,
        "selected": None,
        "prediction": None,
        "plot_url": None,
        "scroll_to_result": False # عشان نعرف المتصفح ينزل للنتيجة ولا لأ
    }

    if request.method == 'POST':
        selected_brand = request.form.get('brand')
        comp_brand = request.form.get('comp_brand')
        
        context["selected"] = selected_brand
        context["scroll_to_result"] = True # المستخدم داس، فلازم ننزله للنتيجة
        
        model_path = f"{selected_brand}_model.h5"
        data_path = f"{selected_brand}_data.csv"

        if os.path.exists(model_path) and os.path.exists(data_path):
            model = load_model(model_path)
            df = pd.read_csv(data_path)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df.values)

            last_seq = scaled_data[-LOOKBACK_DAYS:]
            X_input = last_seq.reshape(1, LOOKBACK_DAYS, 1)
            pred_scaled = model.predict(X_input)
            pred_val = scaler.inverse_transform(pred_scaled)[0][0]

            last_real = df['engagement_rate'].iloc[-1]
            trend_txt = "📈 Trending UP" if pred_val > last_real else "📉 Trending DOWN"
            trend_col = "#2ecc71" if pred_val > last_real else "#ff4757"

            context["prediction"] = f"{pred_val:.4f}"
            context["trend_txt"] = trend_txt
            context["trend_col"] = trend_col

            best_day, pos, neg, neu = get_extras(selected_brand)
            context["best_day"] = best_day
            context["pos_pct"] = pos
            context["neg_pct"] = neg
            context["neu_pct"] = neu

            comp_df = None
            if comp_brand and comp_brand != "None":
                comp_file = f"{comp_brand}_data.csv"
                if os.path.exists(comp_file):
                    comp_df = pd.read_csv(comp_file)
            
            context["plot_url"] = get_plot_url(df, selected_brand, pred_val, comp_df, comp_brand)

    return render_template('index.html', **context)

if __name__ == '__main__':
    app.run(debug=True)