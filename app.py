import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------- Page setup ----------------
st.set_page_config(page_title="Solar Power Predictor", page_icon="☀️", layout="wide")

st.markdown("""
<style>
.hero {background: linear-gradient(135deg, #ff9a3c 0%, #ffcf5c 50%, #4facfe 100%);
       padding: 25px 30px; border-radius: 18px; color: white; display: flex;
       align-items: center; gap: 25px; margin-bottom: 20px;}
.hero h1 {margin: 0; font-size: 2.3rem; color: white;}
.hero p {margin: 5px 0 0 0; font-size: 1.1rem;}
.card {background: #fff8ec; border-left: 6px solid #ff9a3c; padding: 15px 18px;
       border-radius: 12px; margin-bottom: 10px; color: #333;}
.card h3 {margin: 0; color: #e8740c;}
@keyframes spin {from {transform: rotate(0deg);} to {transform: rotate(360deg);}}
.sun-rays {animation: spin 20s linear infinite; transform-origin: 60px 60px;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


@st.cache_data
def load_data():
    try:
        return pd.read_csv("solar_power_output.csv")
    except FileNotFoundError:
        return None


model = load_model()
df = load_data()


def predict(irr):
    return max(float(model.predict(pd.DataFrame({"solar_irradiance": [irr]}))[0]), 0.0)


def solar_scene(irr):
    """Draw a sun + solar panel image; the sun gets bigger and brighter with irradiance."""
    level = min(irr / 1000, 1)
    radius = 18 + 14 * level
    sky = f"rgb({int(120 + 100*level)}, {int(170 + 50*level)}, 255)"
    rays = "".join(
        f'<line x1="60" y1="60" x2="{60 + 50*np.cos(a):.1f}" y2="{60 + 50*np.sin(a):.1f}" '
        f'stroke="#FFD700" stroke-width="3" opacity="{0.3 + 0.7*level:.2f}"/>'
        for a in np.linspace(0, 2*np.pi, 12, endpoint=False))
    cells = "".join(
        f'<rect x="{150 + c*38}" y="{120 + r*28}" width="34" height="24" fill="#1e3a8a" stroke="#93c5fd"/>'
        for r in range(3) for c in range(5))
    return f"""
    <svg viewBox="0 0 380 240" width="100%" style="border-radius:14px;background:{sky}">
      <g class="sun-rays">{rays}</g>
      <circle cx="60" cy="60" r="{radius:.1f}" fill="#FFD700" opacity="{0.5 + 0.5*level:.2f}"/>
      <polygon points="140,110 345,110 355,210 130,210" fill="#0f172a"/>
      {cells}
      <rect x="235" y="205" width="10" height="30" fill="#475569"/>
      <rect x="0" y="232" width="380" height="8" fill="#65a30d"/>
      <text x="370" y="25" text-anchor="end" font-size="16" fill="white" font-weight="bold">
        {irr:.0f} W/m²</text>
    </svg>"""


# ---------------- Hero banner ----------------
st.markdown("""
<div class="hero">
  <svg width="80" height="80" viewBox="0 0 120 120">
    <g class="sun-rays">""" + "".join(
    f'<line x1="60" y1="60" x2="{60 + 55*np.cos(a):.1f}" y2="{60 + 55*np.sin(a):.1f}" stroke="white" stroke-width="4"/>'
    for a in np.linspace(0, 2*np.pi, 12, endpoint=False)) + """</g>
    <circle cx="60" cy="60" r="30" fill="#FFF3B0"/>
  </svg>
  <div><h1>Solar Power Output Predictor</h1>
  <p>Simple Linear Regression · Predict panel output from sunlight</p></div>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Data Insights", "📁 Batch Predict"])

# ================= TAB 1: Prediction =================
with tab1:
    st.markdown("#### ✍️ Enter values and click **Predict**")
    with st.form("predict_form"):
        f1, f2 = st.columns(2)
        irr = f1.number_input("☀️ Solar irradiance (W/m²)", min_value=0.0, max_value=1500.0, value=600.0, step=10.0)
        panels = f2.number_input("🔲 Number of panels", min_value=1, max_value=100, value=10)
        sun_hours = f1.number_input("🕒 Peak sun hours per day", min_value=1.0, max_value=12.0, value=5.5, step=0.5)
        price = f2.number_input("💰 Electricity price (₹ per kWh)", min_value=1.0, max_value=30.0, value=8.0, step=0.5)
        submitted = st.form_submit_button("⚡ Predict Solar Power Output", type="primary", width="stretch")

    if submitted:
        st.session_state["inputs"] = (irr, panels, sun_hours, price)

    if "inputs" not in st.session_state:
        st.info("👆 Enter the values above and press **Predict** to see the results.")
    else:
        irr, panels, sun_hours, price = st.session_state["inputs"]
        pred = predict(irr)
        total_kw = pred * panels / 1000
        daily_kwh = total_kw * sun_hours

        st.markdown(f"""<div class="card" style="text-align:center">
        <h3>Predicted Solar Power Output</h3>
        <span style="font-size:2.6rem;font-weight:bold;color:#e8740c">{pred:.2f} W</span><br>
        per panel at {irr:.0f} W/m²</div>""", unsafe_allow_html=True)
        if irr < 104 or irr > 999:
            st.warning("⚠️ This irradiance is outside the training range (104–999 W/m²), "
                       "so the prediction is an estimate beyond the data.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("⚡ Output per panel", f"{pred:.1f} W")
        c2.metric("🔋 Total system power", f"{total_kw:.2f} kW")
        c3.metric("📅 Daily energy", f"{daily_kwh:.2f} kWh")
        c4.metric("💰 Monthly savings", f"₹{daily_kwh * 30 * price:,.0f}")

        left, right = st.columns([1, 1])
        with left:
            st.markdown("#### 🌤️ Live Solar Scene")
            st.markdown(solar_scene(irr), unsafe_allow_html=True)
            if irr < 300:
                st.info("🌥️ Low sunlight: cloudy or early morning conditions.")
            elif irr < 700:
                st.warning("⛅ Moderate sunlight: decent generation.")
            else:
                st.success("☀️ Strong sunlight: peak generation!")

        with right:
            st.markdown("#### 🎯 Output Gauge")
            gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta", value=pred,
                number={"suffix": " W"},
                delta={"reference": 283, "suffix": " vs avg"},
                gauge={"axis": {"range": [0, 600]},
                       "bar": {"color": "#e8740c"},
                       "steps": [{"range": [0, 150], "color": "#dbeafe"},
                                 {"range": [150, 350], "color": "#fde68a"},
                                 {"range": [350, 600], "color": "#fdba74"}]}))
            gauge.update_layout(height=320, margin=dict(t=30, b=10))
            st.plotly_chart(gauge, width="stretch")

        st.markdown("#### 📈 Where your prediction sits on the regression line")
        xs = np.linspace(0, 1200, 100)
        line = model.predict(pd.DataFrame({"solar_irradiance": xs}))
        fig = go.Figure()
        if df is not None:
            fig.add_trace(go.Scatter(x=df["solar_irradiance"], y=df["solar_power_output"], mode="markers",
                                     name="Historical data", marker=dict(color="#60a5fa", size=6, opacity=0.5)))
        fig.add_trace(go.Scatter(x=xs, y=line, mode="lines", name="Regression line",
                                 line=dict(color="red", width=3)))
        fig.add_trace(go.Scatter(x=[irr], y=[pred], mode="markers", name="Your input",
                                 marker=dict(color="gold", size=20, symbol="star", line=dict(color="black", width=2))))
        fig.update_layout(xaxis_title="Solar irradiance (W/m²)", yaxis_title="Output (W)", height=420)
        st.plotly_chart(fig, width="stretch")

        st.markdown("#### 🕒 Estimated generation through the day")
        hours = np.arange(6, 19)
        hourly_irr = irr * np.sin(np.pi * (hours - 6) / 12)          # bell-shaped sunlight curve
        hourly_out = [predict(h) * panels / 1000 for h in hourly_irr]
        day = px.area(x=hours, y=hourly_out, labels={"x": "Hour of day", "y": "System output (kW)"},
                      color_discrete_sequence=["#ff9a3c"])
        day.update_layout(height=320)
        st.plotly_chart(day, width="stretch")

        st.markdown(f"""<div class="card"><h3>🧮 Model Equation</h3>
        Output = {model.intercept_:.3f} + {model.coef_[0]:.4f} × Irradiance<br>
        Every extra <b>100 W/m²</b> of sunlight adds about <b>{model.coef_[0]*100:.0f} W</b> per panel.</div>""",
                    unsafe_allow_html=True)

# ================= TAB 2: Data insights =================
with tab2:
    if df is None:
        st.error("Place `solar_power_output.csv` next to app.py to see data insights.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Records", len(df))
        k2.metric("Avg output", f"{df['solar_power_output'].mean():.0f} W")
        k3.metric("Max output", f"{df['solar_power_output'].max():.0f} W")
        k4.metric("Correlation (irr ↔ output)", f"{df['solar_irradiance'].corr(df['solar_power_output']):.3f}")

        st.markdown("#### 📊 Univariate: distribution of a column")
        col = st.selectbox("Choose a column", df.columns, index=4)
        st.plotly_chart(px.histogram(df, x=col, nbins=25, marginal="box",
                                     color_discrete_sequence=["#ff9a3c"]), width="stretch")

        st.markdown("#### 🔗 Bivariate: feature vs output")
        feat = st.selectbox("Choose a feature", df.columns[:-1], index=2)
        st.plotly_chart(px.scatter(df, x=feat, y="solar_power_output", color="temperature",
                                   color_continuous_scale="Plasma"), width="stretch")

        st.markdown("#### 🌡️ Multivariate: correlation heatmap")
        st.plotly_chart(px.imshow(df.corr().round(2), text_auto=True, color_continuous_scale="RdBu_r",
                                  zmin=-1, zmax=1), width="stretch")

        st.markdown("""<div class="card"><h3>💡 Key Insights</h3>
        • Solar irradiance is the main driver of output (r ≈ 0.999).<br>
        • Output is about half of irradiance (≈ 0.5 × W/m²).<br>
        • Temperature, humidity and wind speed have almost no effect.</div>""", unsafe_allow_html=True)

# ================= TAB 3: Batch prediction =================
with tab3:
    st.write("Upload a CSV with a **`solar_irradiance`** column.")
    file = st.file_uploader("CSV file", type="csv")
    if file is not None:
        data = pd.read_csv(file)
        if "solar_irradiance" not in data.columns:
            st.error("Column `solar_irradiance` not found.")
        else:
            data["predicted_output"] = model.predict(data[["solar_irradiance"]]).clip(min=0)
            st.success(f"✅ Predicted {len(data)} rows")
            st.dataframe(data, width="stretch")
            st.plotly_chart(px.scatter(data, x="solar_irradiance", y="predicted_output",
                                       color="predicted_output", color_continuous_scale="YlOrRd"),
                            width="stretch")
            st.download_button("⬇️ Download predictions", data.to_csv(index=False).encode(),
                               "predictions.csv", "text/csv")

st.caption("Built with Streamlit · Simple Linear Regression model")
