import os
import time
import joblib
import streamlit as st
import pandas as pd
import numpy as np

# Page config
st.set_page_config(page_title="Car Price Predictor — NeoGlass", page_icon="🚗", layout="wide")

# ---------- Helpers ----------
def resource_path(name):
    """Return local path if exists else return remote placeholder."""
    if os.path.exists(name):
        return name
    return None

def load_model(path="car_price_model.pkl"):
    try:
        model_obj = joblib.load(path)
        return model_obj, None
    except FileNotFoundError:
        return None, f"Model file not found at `{path}`. Save your trained model as {path} and place it next to app.py."
    except Exception as e:
        return None, f"Failed to load model: {e}"

def prepare_input_for_model(model, input_dict):
    # Build dataframe and reindex to model expected features if possible
    df_input = pd.DataFrame([input_dict])
    if hasattr(model, "feature_names_in_"):
        try:
            cols = list(model.feature_names_in_)
            df_input = df_input.reindex(columns=cols, fill_value=0)
            return df_input
        except Exception:
            return df_input
    else:
        # best-effort: sort columns alphabetically (user should ensure names match training)
        return df_input

# ---------- CSS (NeoGlass) ----------
st.markdown(
    """
    <style>
    :root{
      --glass-bg: rgba(255,255,255,0.06);
      --glass-border: rgba(255,255,255,0.08);
      --accent: #1A73E8;
      --muted: rgba(255,255,255,0.6);
      --card-radius: 14px;
    }
    html,body,#root, .block-container{
      background: linear-gradient(180deg,#0f1724 0%, #071025 100%);
      color: #e6eef8;
    }
    .neo-title{
      font-size:44px;
      font-weight:800;
      color:var(--accent);
      text-align:center;
      margin-bottom:6px;
    }
    .neo-sub{
      color:var(--muted);
      text-align:center;
      margin-bottom:30px;
    }
    .neo-card{
      background: linear-gradient(135deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
      border: 1px solid var(--glass-border);
      backdrop-filter: blur(8px) saturate(120%);
      border-radius: var(--card-radius);
      padding: 22px;
      box-shadow: 0 6px 20px rgba(2,6,23,0.6);
    }
    .neo-small{
      color:var(--muted);
      font-size:13px;
    }
    .glow-btn > button {
      background: linear-gradient(90deg,var(--accent), #2BC0F0);
      border: none;
      padding: 12px 18px;
      font-weight:700;
      font-size:16px;
      border-radius:10px;
      color:#fff;
      box-shadow: 0 6px 30px rgba(26,115,232,0.18);
    }
    .sidebar .stButton>button {
      border-radius:10px;
    }
    /* small responsiveness fixes */
    @media (max-width: 760px){
      .neo-title{font-size:32px;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
st.sidebar.markdown("## 🚗 Navigation")
page = st.sidebar.radio("", ["Home", "Predict Price", "Model Info", "Model Comparison"])

st.sidebar.markdown("---")
st.sidebar.markdown("Created by **Aaditya** • CarDekho Price Predictor")
st.sidebar.markdown("Model: **Random Forest** (recommended)")

# ---------- Load model ----------
model, model_load_msg = load_model("car_price_model.pkl")
model_loaded = model is not None

# ---------- Home ----------
if page == "Home":
    st.markdown("<div class='neo-title'>Car Price Prediction System</div>", unsafe_allow_html=True)
    st.markdown("<div class='neo-sub'>NeoGlass UI • ML-powered price estimation using car features</div>", unsafe_allow_html=True)

    left, right = st.columns([1, 1])

    with left:
        image_local = resource_path("images/hero_car.jpg") or resource_path("hero_car.jpg")
        if image_local:
            st.image(image_local, use_column_width=True)
        else:
            st.image("https://cdn.pixabay.com/photo/2021/01/08/07/02/car-5898455_1280.jpg", use_column_width=True)

    with right:
        st.markdown("<div class='neo-card'>", unsafe_allow_html=True)
        st.markdown("### 📌 About the Project")
        st.markdown(
            "This Car Price Predictor estimates the selling price of used cars using an ML model trained on the CarDekho dataset. "
            "Enter car details and get an instant prediction. The UI uses a NeoGlass style for a polished look."
        )
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("#### 🔥 Key Features", unsafe_allow_html=True)
        st.markdown("- Live input-based prediction  \n- Random Forest model  \n- Clean NeoGlass UI  \n- Easy export & reuse", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("<div class='neo-card'>", unsafe_allow_html=True)
    st.markdown("### 📷 Snapshot")
    sample_img = resource_path("images/snapshot.png")
    if sample_img:
        st.image(sample_img, use_column_width=True)
    else:
        st.markdown("<div class='neo-small'>No local snapshot image found — using remote placeholder.</div>", unsafe_allow_html=True)
        st.image("https://cdn.pixabay.com/photo/2015/04/23/22/00/tree-736885_1280.jpg", width=700)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Predict Page ----------
elif page == "Predict Price":
    st.markdown("<div class='neo-title'>Predict Car Price</div>", unsafe_allow_html=True)
    st.markdown("<div class='neo-sub'>Fill the form and press Predict</div>", unsafe_allow_html=True)

    # input columns
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='neo-card'>", unsafe_allow_html=True)
        year = st.number_input("Year of Purchase", 1990, 2025, 2016)
        km_driven = st.number_input("Kilometers Driven", 0, 1000000, 35000)
        mileage = st.number_input("Mileage (km/l)", 5.0, 40.0, 18.5, format="%.2f")
        engine = st.number_input("Engine CC", 500, 6000, 1197)
        max_power = st.number_input("Max Power (bhp)", 30.0, 500.0, 82.0, format="%.2f")
        seats = st.number_input("Seats", 2, 10, 5)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='neo-card'>", unsafe_allow_html=True)
        fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG"])
        seller = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
        trans = st.selectbox("Transmission", ["Manual", "Automatic"])
        owner = st.selectbox("Owner Type", ["First Owner", "Second Owner", "Third Owner", "Test Drive Car", "Fourth & Above Owner"])
        brand = st.selectbox("Brand", ["Maruti", "Hyundai", "Tata", "Mahindra", "Honda", "Toyota", "Kia", "Renault", "Volkswagen", "Ford", "BMW", "Audi", "Mercedes-Benz"])
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Prediction area card
    st.markdown("<div class='neo-card'>", unsafe_allow_html=True)
    st.markdown("### Ready to predict")
    cols = st.columns([2, 1])
    with cols[0]:
        st.markdown("Enter values and click Predict. The app will reindex your input to the model's expected features where possible.", unsafe_allow_html=True)
    with cols[1]:
        if not model_loaded:
            st.error("Prediction unavailable — model not loaded.")
            st.info(model_load_msg or "Place car_price_model.pkl next to app.py and refresh.")
        else:
            st.markdown("<div class='glow-btn'>", unsafe_allow_html=True)
            do_predict = st.button("Predict Price")
            st.markdown("</div>", unsafe_allow_html=True)

            if do_predict:
                # build input dict
                input_dict = {
                    'year': year,
                    'km_driven': km_driven,
                    'mileage': mileage,
                    'engine': engine,
                    'max_power': max_power,
                    'seats': seats,
                }
                input_dict[f"fuel_{fuel}"] = 1
                input_dict[f"seller_type_{seller}"] = 1
                input_dict[f"transmission_{trans}"] = 1
                input_dict[f"owner_{owner}"] = 1
                input_dict[f"brand_{brand}"] = 1

                # prepare and predict
                df_input = prepare_input_for_model(model, input_dict)
                try:
                    pred_price = model.predict(df_input)[0]
                    # single styled display
                    st.markdown(f"<h2 style='color: #34d399;'>Predicted Price: ₹ {pred_price:,.0f}</h2>", unsafe_allow_html=True)

                    # Save prediction history in session_state
                    if "history" not in st.session_state:
                        st.session_state.history = []
                    st.session_state.history.append({"input": input_dict, "pred": float(pred_price), "time": time.strftime("%Y-%m-%d %H:%M:%S")})
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

    # Show prediction history
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='neo-card'>", unsafe_allow_html=True)
    st.markdown("### Prediction History")
    if "history" in st.session_state and st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df.sort_values(by="time", ascending=False).reset_index(drop=True))
        if st.button("Clear History"):
            st.session_state.history = []
            st.experimental_rerun()
    else:
        st.markdown("<div class='neo-small'>No predictions yet in this session.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------- Model Info ----------
elif page == "Model Info":
    st.markdown("<div class='neo-title'>Model Information</div>", unsafe_allow_html=True)
    st.markdown("<div class='neo-sub'>Random Forest • Model Details & Guidance</div>", unsafe_allow_html=True)

    st.markdown("<div class='neo-card'>", unsafe_allow_html=True)
    if model_loaded:
        st.markdown(f"**Model loaded successfully.**")
        try:
            feat_names = list(model.feature_names_in_)
            st.markdown(f"- Number of features: **{len(feat_names)}**")
            st.markdown("- Top 20 features (if available):")
            display = pd.DataFrame({"feature": feat_names})
            st.dataframe(display.head(20))
        except Exception:
            st.markdown("- Model doesn’t expose `feature_names_in_`. Ensure input features match training columns.")
    else:
        st.error("Model not loaded.")
        st.info(model_load_msg or "Place car_price_model.pkl next to app.py and refresh.")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Model Comparison ----------
elif page == "Model Comparison":
    st.markdown("<div class='neo-title'>Model Comparison</div>", unsafe_allow_html=True)
    st.markdown("<div class='neo-sub'>R² summary for your trained models</div>", unsafe_allow_html=True)

    st.markdown("<div class='neo-card'>", unsafe_allow_html=True)

    # Use actual R² values if stored in session_state; else default
    if "model_accuracy" in st.session_state:
        comp = st.session_state.model_accuracy
    else:
        # Default values based on training results
        comp = {
            "Random Forest": 0.97,
            "Linear Regression": 0.85,
            "KNN Regressor": 0.80
        }

    # display table
    comp_df = pd.DataFrame({
        "Model": list(comp.keys()),
        "R2 Score": [comp[k] for k in comp],
        "Accuracy (%)": [comp[k] * 100 for k in comp]
    })

    # Highlight best and worst R²
    def highlight_best_worst(row):
        if row["R2 Score"] == max(comp.values()):
            return ['color: #d4edda']*3  # green for best
        elif row["R2 Score"] == min(comp.values()):
            return ['color: #f8d7da']*3  # red for worst
        else:
            return ['']*3

    st.dataframe(comp_df.style.apply(highlight_best_worst, axis=1))

    # simple bar chart
    st.bar_chart(pd.Series({k: comp[k]*100 for k in comp}))

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:var(--muted);padding-bottom:18px;'>Made by Aaditya — CarDekho Price Predictor</div>", unsafe_allow_html=True)
