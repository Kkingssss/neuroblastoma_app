import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

# ─── Model definition (เหมือนเดิม) ───────────────────────────────────────────
Dense   = tf.keras.layers.Dense
Attn    = tf.keras.layers.MultiHeadAttention
Dropout = tf.keras.layers.Dropout
L2Norm  = tf.keras.layers.Lambda(
    lambda x: tf.math.l2_normalize(x, axis=1), name='L2norm')

class sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return mean + tf.random.normal(shape=tf.shape(mean)) * tf.exp(0.5 * log_var)

def build(N):
    inp = tf.keras.Input((N, 2))
    x   = Dense(32, activation='relu')(inp); x0 = x
    x, _ = Attn(1, 128, name='attention0')(x, x, return_attention_scores=True)
    x   = tf.keras.layers.Concatenate()([x0, x])
    x   = tf.keras.layers.Flatten()(x)
    x   = Dense(128, activation='relu')(x)
    x   = [Dense(64, activation='relu')(x) for _ in range(2)]
    x   = sampling()(x); x = L2Norm(x); h = x
    def head(h, out_dim, activation):
        x = Dense(128, activation='relu')(h); x = Dropout(0.5)(x)
        x = Dense(32,  activation='relu')(x); x = Dropout(0.5)(x)
        return Dense(out_dim, activation=activation)(x)
    outR = head(h, N,  None)
    out1 = head(h, 1,  'sigmoid')
    out2 = head(h, 3,  'softmax')
    m = tf.keras.Model(inp, (outR, out1, out2))
    m.compile(optimizer=tf.keras.optimizers.AdamW(1e-5),
              loss={'outR': tf.keras.losses.MeanSquaredError(),
                    'out1': tf.keras.losses.BinaryFocalCrossentropy(),
                    'out2': tf.keras.losses.CategoricalCrossentropy()})
    return m

# ─── Normalization ────────────────────────────────────────────────────────────
def nLogNorm_out(x, U=None, k=1.5):
    def nOut(col):
        q25, q75 = np.percentile(col[~np.isnan(col)], [25, 75])
        return q75 + k * (q75 - q25)
    A = []
    if U is None:
        U = []
        for i in range(x.shape[1]):
            u = nOut(x[:, i]); A.append(_lognorm(x[:, i], u)); U.append(u)
    else:
        for i in range(x.shape[1]):
            A.append(_lognorm(x[:, i], U[i]))
    return np.stack(A, axis=1), np.array(U)

def _lognorm(x, u):
    a = np.copy(x); aw = np.where(~np.isnan(a))
    a[aw] = np.log(np.clip(a[aw], 0, None) + 1) / np.log(u + 1)
    return a

def to_model_input(values: np.ndarray, U: np.ndarray) -> np.ndarray:
    """values: [n,5] raw  →  [n,5,2] normalized + mask"""
    x, _ = nLogNorm_out(values, U)
    mask  = (~np.isnan(values)).astype(float)
    x     = np.nan_to_num(x, nan=0.0)
    return np.stack([x, mask], axis=-1).astype(np.float32)

# ─── Load model & normalization params ────────────────────────────────────────
@st.cache_resource
def load_model():
    m = build(5)
    m.load_weights('kfull_best.weights.h5')
    U = np.load('Upperout_train_ss3.npy')
    return m, U

FEATURES     = ['NMN', 'MN', 'MTY', 'VMA', 'HVA']
TUMOR_LABELS = ['Non-NB', 'Adrenal', 'Extra-Adrenal']
THRESHOLDS   = {
    '🔍 Screening  (Sens=1.00, Spec=0.25)': 0.616,
    '⚖️ Balanced   (Sens=0.89, Spec=0.75)': 0.794,
    '✅ Confirmatory (Sens=0.44, Spec=1.00)': 0.825,
}

# ─── UI ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title='De-NB', page_icon='🧬', layout='wide')
st.markdown("""
<style>
  .main {background:#0f1117}
  .metric-card {
    background: linear-gradient(135deg,#1e2130,#2d3250);
    border-radius:12px; padding:20px; margin:8px 0;
    border-left: 4px solid #7c83fd;
  }
  .nb-positive {border-left-color:#ff6b6b !important}
  .nb-negative {border-left-color:#51cf66 !important}
</style>""", unsafe_allow_html=True)

st.title('🧬 De-NB: Neuroblastoma Diagnostic Assistant')
st.caption('VAE + Self-Attention model | Independent test AUPRC = 0.936 (95% CI: 0.771–1.000)')

model, U = load_model()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header('⚙️ Settings')
    mode_label = st.radio('Clinical mode', list(THRESHOLDS.keys()))
    threshold  = THRESHOLDS[mode_label]

    st.markdown('---')
    st.markdown('**Mode guide**')
    st.markdown('- 🔍 Screening: ไม่พลาด NB (NPV=1.0)')
    st.markdown('- ⚖️ Balanced: balance Sens/Spec')
    st.markdown('- ✅ Confirmatory: ยืนยัน NB (PPV=1.0)')
    st.markdown('---')
    st.caption('⚠️ For research use only. n=13 test set.')

# ─── Input tabs ───────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(['📝 Manual Input', '📊 Excel Upload', '📁 NPY Upload'])

def predict_and_show(raw_values: np.ndarray, patient_ids: list):
    """raw_values: [n,5] with NaN for missing"""
    X_inp = to_model_input(raw_values, U)
    preds = model.predict(X_inp, verbose=0)

    score_nb  = preds[1].flatten()
    prob_loc  = preds[2]
    label_nb  = ['🔴 NB' if s >= threshold else '🟢 Not NB' for s in score_nb]
    label_loc = [TUMOR_LABELS[np.argmax(p)] for p in prob_loc]

    st.markdown('---')
    st.subheader('📋 Results')

    # ── per-patient cards ────────────────────────────────────────────────────
    for i, pid in enumerate(patient_ids):
        is_nb    = score_nb[i] >= threshold
        card_cls = 'nb-positive' if is_nb else 'nb-negative'
        loc_probs = {TUMOR_LABELS[j]: f'{prob_loc[i,j]*100:.1f}%' for j in range(3)}

        col1, col2, col3 = st.columns([2, 2, 3])
        with col1:
            st.markdown(f"""<div class="metric-card {card_cls}">
                <b>{pid}</b><br>
                <span style="font-size:1.4em">{label_nb[i]}</span><br>
                <small>Score: {score_nb[i]:.4f} (thr={threshold})</small>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card">
                <b>Location</b><br>
                <span style="font-size:1.2em">{label_loc[i]}</span>
            </div>""", unsafe_allow_html=True)
        with col3:
            fig = px.bar(
                x=list(loc_probs.values()), y=list(loc_probs.keys()),
                orientation='h', text=list(loc_probs.values()),
                color_discrete_sequence=['#7c83fd','#ff6b6b','#51cf66'],
                template='plotly_dark', height=120
            )
            fig.update_layout(margin=dict(l=0,r=0,t=0,b=0),
                              showlegend=False, xaxis_title='',
                              yaxis_title='', xaxis=dict(range=[0,100]))
            st.plotly_chart(fig, use_container_width=True, key=f'loc_{i}')

    # ── summary table ────────────────────────────────────────────────────────
    df_out = pd.DataFrame({
        'Patient':        patient_ids,
        'NB Score':       score_nb.round(4),
        'Diagnosis':      label_nb,
        'Location':       label_loc,
        **{f'P({TUMOR_LABELS[j]})': prob_loc[:,j].round(3) for j in range(3)}
    })

    st.markdown('---')
    st.subheader('📊 Summary')

    # gauge chart — NB score distribution
    if len(patient_ids) > 1:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=patient_ids, y=score_nb,
            marker_color=['#ff6b6b' if s >= threshold else '#51cf66' for s in score_nb],
            text=[f'{s:.3f}' for s in score_nb], textposition='outside'
        ))
        fig2.add_hline(y=threshold, line_dash='dash', line_color='#ffd43b',
                       annotation_text=f'Threshold {threshold}')
        fig2.update_layout(template='plotly_dark', title='NB Score per Patient',
                           yaxis=dict(range=[0, 1.1]), height=350)
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(df_out, use_container_width=True)

    # download
    buf = BytesIO()
    df_out.to_excel(buf, index=False); buf.seek(0)
    st.download_button('⬇️ Download Excel', buf,
                       'denb_results.xlsx',
                       'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# ── Tab 1: Manual input ───────────────────────────────────────────────────────
with tab1:
    st.markdown('กรอกค่า biomarker (เว้นว่างได้ถ้าไม่มีข้อมูล)')
    n_patients = st.number_input('จำนวนผู้ป่วย', 1, 20, 1)
    rows = []
    for i in range(n_patients):
        st.markdown(f'**Patient {i+1}**')
        cols = st.columns(5)
        row  = []
        for j, feat in enumerate(FEATURES):
            val = cols[j].text_input(feat, key=f'p{i}_{feat}', placeholder='NaN')
            row.append(float(val) if val.strip() != '' else np.nan)
        rows.append(row)

    if st.button('🔬 Predict', type='primary'):
        raw = np.array(rows, dtype=float)
        predict_and_show(raw, [f'Patient {i+1}' for i in range(n_patients)])

# ── Tab 2: Excel upload ───────────────────────────────────────────────────────
with tab2:
    st.markdown('**Format:** คอลัมน์ต้องมีชื่อ `NMN, MN, MTY, VMA, HVA` (เว้นว่างได้)')
    st.download_button('⬇️ Download template',
                       pd.DataFrame(columns=FEATURES).to_csv(index=False),
                       'template.csv', 'text/csv')

    uploaded_xl = st.file_uploader('Upload Excel / CSV', type=['xlsx','xls','csv'])
    if uploaded_xl:
        df_in = (pd.read_csv(uploaded_xl) if uploaded_xl.name.endswith('.csv')
                 else pd.read_excel(uploaded_xl))
        missing_cols = [f for f in FEATURES if f not in df_in.columns]
        if missing_cols:
            st.error(f'ไม่พบคอลัมน์: {missing_cols}')
        else:
            st.dataframe(df_in[FEATURES].head(), use_container_width=True)
            pid = df_in['Patient'].tolist() if 'Patient' in df_in.columns \
                  else [f'Patient {i+1}' for i in range(len(df_in))]
            if st.button('🔬 Predict from file', type='primary'):
                predict_and_show(df_in[FEATURES].values.astype(float), pid)

# ── Tab 3: NPY upload (เดิม) ──────────────────────────────────────────────────
with tab3:
    st.markdown('Upload `.npy` file shape `[n, 5, 2]` (normalized + mask)')
    uploaded_npy = st.file_uploader('Upload NPY', type=['npy'])
    if uploaded_npy:
        data  = np.load(uploaded_npy)
        preds = model.predict(data, verbose=0)
        score_nb  = preds[1].flatten()
        label_nb  = ['🔴 NB' if s >= threshold else '🟢 Not NB' for s in score_nb]
        label_loc = [TUMOR_LABELS[np.argmax(p)] for p in preds[2]]
        df_npy = pd.DataFrame({
            'Patient':   [f'Patient {i+1}' for i in range(len(score_nb))],
            'NB Score':  score_nb.round(4),
            'Diagnosis': label_nb,
            'Location':  label_loc,
        })
        st.dataframe(df_npy, use_container_width=True)
