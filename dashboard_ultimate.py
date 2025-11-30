import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys

# --- Configuration ---
BRANDS_LIST = ["Nike", "Google", "Apple", "Adidas"]
LOOKBACK_DAYS = 14

# --- Path Setup ---
if getattr(sys, 'frozen', False):
    app_path = os.path.dirname(sys.executable)
else:
    app_path = os.path.dirname(os.path.abspath(__file__))

class UltimateDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("🚀 AI Brand Strategist (Ultimate Edition)")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1e272e")

        self.current_brand = BRANDS_LIST[0]
        self.competitor_brand = None
        
        self.model = None
        self.scaler = None
        self.df_daily = None    # Processed Daily Data
        self.df_main = None     # Original Big Dataset (For Sentiment/Days)
        
        # Load Main Dataset Once
        try:
            self.df_main = pd.read_csv(os.path.join(app_path, 'Social Media Engagement Dataset.csv'))
            self.df_main['timestamp'] = pd.to_datetime(self.df_main['timestamp'])
        except Exception as e:
            messagebox.showerror("Error", "Could not load 'Social Media Engagement Dataset.csv'.\nPlease ensure it is in the folder.")
            sys.exit()

        self.setup_ui()
        self.load_brand_resources(self.current_brand)

    def setup_ui(self):
        # --- 1. Header ---
        header = tk.Frame(self.root, bg="#0fb9b1", height=70)
        header.pack(side="top", fill="x")
        tk.Label(header, text="📈 AI Brand Analytics & Forecasting Hub", 
                 font=("Segoe UI", 20, "bold"), bg="#0fb9b1", fg="white").pack(pady=15)

        # --- 2. Control Bar (Top) ---
        controls = tk.Frame(self.root, bg="#2d3436", pady=10)
        controls.pack(fill="x")

        # Select Brand
        tk.Label(controls, text="Target Brand:", font=("Arial", 11), bg="#2d3436", fg="#dfe6e9").pack(side="left", padx=10)
        self.brand_var = tk.StringVar(value=BRANDS_LIST[0])
        self.combo_brand = ttk.Combobox(controls, textvariable=self.brand_var, values=BRANDS_LIST, state="readonly", width=15)
        self.combo_brand.pack(side="left")
        self.combo_brand.bind("<<ComboboxSelected>>", self.on_brand_change)

        # Compare Checkbox
        self.chk_var = tk.IntVar()
        self.chk_compare = tk.Checkbutton(controls, text="Compare Mode", variable=self.chk_var, 
                                          bg="#2d3436", fg="#00d2d3", selectcolor="#2d3436", font=("Arial", 10, "bold"),
                                          command=self.toggle_compare)
        self.chk_compare.pack(side="left", padx=20)

        # Competitor Select (Hidden by default)
        self.comp_var = tk.StringVar(value=BRANDS_LIST[1])
        self.combo_comp = ttk.Combobox(controls, textvariable=self.comp_var, values=BRANDS_LIST, state="disabled", width=15)
        self.combo_comp.pack(side="left")
        self.combo_comp.bind("<<ComboboxSelected>>", self.on_comp_change)

        # Predict Button
        tk.Button(controls, text="🔮 ANALYZE & PREDICT", font=("Arial", 11, "bold"), 
                  bg="#ff5e57", fg="white", padx=20, command=self.predict_all).pack(side="right", padx=20)

        # --- 3. Main Dashboard Layout ---
        main_frame = tk.Frame(self.root, bg="#1e272e")
        main_frame.pack(fill="both", expand=True, padx=15, pady=15)

        # Left Column: Insights (Sentiment, Best Day, Forecast)
        left_col = tk.Frame(main_frame, bg="#1e272e", width=300)
        left_col.pack(side="left", fill="y", padx=(0, 15))

        # --- Insight Card 1: Forecast ---
        card1 = tk.Frame(left_col, bg="#485460", bd=2, relief="flat", padx=10, pady=10)
        card1.pack(fill="x", pady=(0, 10))
        tk.Label(card1, text="AI Forecast (Next Day)", font=("Arial", 12), bg="#485460", fg="#d2dae2").pack()
        self.lbl_forecast = tk.Label(card1, text="--", font=("Arial", 26, "bold"), bg="#485460", fg="#0be881")
        self.lbl_forecast.pack()
        self.lbl_trend = tk.Label(card1, text="Wait...", font=("Arial", 10), bg="#485460", fg="white")
        self.lbl_trend.pack()

        # --- Insight Card 2: Golden Time (Best Day) ---
        card2 = tk.Frame(left_col, bg="#485460", bd=2, relief="flat", padx=10, pady=10)
        card2.pack(fill="x", pady=(0, 10))
        tk.Label(card2, text="⏰ Best Day to Post", font=("Arial", 12), bg="#485460", fg="#d2dae2").pack()
        self.lbl_best_day = tk.Label(card2, text="--", font=("Arial", 20, "bold"), bg="#485460", fg="#ffd32a")
        self.lbl_best_day.pack()
        tk.Label(card2, text="Based on historical avg", font=("Arial", 8), bg="#485460", fg="#bdc3c7").pack()

        # --- Insight Card 3: Sentiment Analysis ---
        card3 = tk.Frame(left_col, bg="#485460", bd=2, relief="flat", padx=10, pady=10)
        card3.pack(fill="x")
        tk.Label(card3, text="🧠 Sentiment Analysis", font=("Arial", 12), bg="#485460", fg="#d2dae2").pack(pady=(0,5))
        
        self.lbl_sentiment_txt = tk.Label(card3, text="Pos: 0% | Neg: 0%", font=("Arial", 10), bg="#485460", fg="white")
        self.lbl_sentiment_txt.pack()
        
        # Simple Bar for Sentiment
        self.canvas_sent = tk.Canvas(card3, width=200, height=20, bg="#bdc3c7", highlightthickness=0)
        self.canvas_sent.pack(pady=5)
        self.rect_pos = self.canvas_sent.create_rectangle(0, 0, 0, 20, fill="#2ecc71", outline="")
        
        # Right Column: The Graph
        right_col = tk.Frame(main_frame, bg="white")
        right_col.pack(side="right", fill="both", expand=True)
        self.graph_frame = right_col

    # --- Logic ---

    def toggle_compare(self):
        if self.chk_var.get() == 1:
            self.combo_comp.config(state="readonly")
            self.competitor_brand = self.combo_comp.get()
        else:
            self.combo_comp.config(state="disabled")
            self.competitor_brand = None
        self.plot_graph()

    def load_brand_resources(self, brand):
        try:
            # 1. Load LSTM Files
            model_file = os.path.join(app_path, f"{brand}_model.h5")
            data_file = os.path.join(app_path, f"{brand}_data.csv")
            
            if not os.path.exists(model_file): return False

            self.model = load_model(model_file)
            self.df_daily = pd.read_csv(data_file)
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.scaled_data = self.scaler.fit_transform(self.df_daily.values)

            # 2. Run Analytics (Sentiment & Best Day) using Main DF
            self.analyze_extras(brand)
            
            # 3. Plot
            self.plot_graph()
            return True

        except Exception as e:
            print(e)
            return False

    def analyze_extras(self, brand):
        # Filter Main Data for this brand
        subset = self.df_main[self.df_main['brand_name'] == brand]
        
        if len(subset) == 0: return

        # A. Best Day Logic
        # Group by day name and find max engagement
        avg_per_day = subset.groupby('day_of_week')['engagement_rate'].mean()
        best_day = avg_per_day.idxmax()
        self.lbl_best_day.config(text=best_day)

        # B. Sentiment Logic (Last 100 posts)
        recent_posts = subset.tail(100)
        counts = recent_posts['sentiment_label'].value_counts()
        total = len(recent_posts)
        pos = counts.get('Positive', 0)
        neg = counts.get('Negative', 0)
        
        pos_pct = int((pos / total) * 100)
        neg_pct = int((neg / total) * 100)

        self.lbl_sentiment_txt.config(text=f"Pos: {pos_pct}%  |  Neg: {neg_pct}%")
        
        # Update Visual Bar (Green width based on Positive %)
        bar_width = (pos_pct / 100) * 200
        self.canvas_sent.coords(self.rect_pos, 0, 0, bar_width, 20)


    def on_brand_change(self, event):
        self.current_brand = self.brand_var.get()
        self.lbl_forecast.config(text="--")
        self.lbl_trend.config(text="Wait...")
        self.load_brand_resources(self.current_brand)

    def on_comp_change(self, event):
        self.competitor_brand = self.comp_var.get()
        if self.chk_var.get() == 1:
            self.plot_graph()

    def predict_all(self):
        # Forecast Logic for Current Brand
        if self.model is None: return

        try:
            last_seq = self.scaled_data[-LOOKBACK_DAYS:]
            X_input = last_seq.reshape(1, LOOKBACK_DAYS, 1)
            pred_scaled = self.model.predict(X_input)
            pred_val = self.scaler.inverse_transform(pred_scaled)[0][0]

            self.lbl_forecast.config(text=f"{pred_val:.3f}")
            
            last_real = self.df_daily['engagement_rate'].iloc[-1]
            if pred_val > last_real:
                self.lbl_trend.config(text="📈 TRENDING UP", fg="#0be881")
            else:
                self.lbl_trend.config(text="📉 TRENDING DOWN", fg="#ff5e57")

            # Update Graph with Prediction Star
            self.plot_graph(pred_val)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def plot_graph(self, pred_val=None):
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)

        # 1. Plot Primary Brand
        subset = self.df_daily.tail(45).reset_index(drop=True)
        ax.plot(subset.index, subset['engagement_rate'], label=f"{self.current_brand}", color='#0984e3', linewidth=3)

        # 2. Plot Prediction (if exists)
        if pred_val is not None:
            last_idx = len(subset) - 1
            ax.plot(last_idx + 1, pred_val, marker='*', markersize=20, color='#00cec9', label='AI Forecast', zorder=5)

        # 3. Plot Competitor (if Compare Mode is ON)
        if self.chk_var.get() == 1 and self.competitor_brand:
            # Load competitor data on the fly
            comp_file = os.path.join(app_path, f"{self.competitor_brand}_data.csv")
            if os.path.exists(comp_file):
                comp_df = pd.read_csv(comp_file)
                comp_subset = comp_df.tail(45).reset_index(drop=True)
                ax.plot(comp_subset.index, comp_subset['engagement_rate'], label=f"{self.competitor_brand} (Comp)", color='#fab1a0', linewidth=2, linestyle='--')

        ax.set_title("Engagement Trend Analysis", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = UltimateDashboard(root)
    root.mainloop()