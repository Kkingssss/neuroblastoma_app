import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

# ─── Model ────────────────────────────────────────────────────────────────────
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
    def head(h, out_dim, act):
        x = Dense(128, activation='relu')(h); x = Dropout(0.5)(x)
        x = Dense(32,  activation='relu')(x);  x = Dropout(0.5)(x)
        return Dense(out_dim, activation=act)(x)
    m = tf.keras.Model(inp, (head(h,N,None), head(h,1,'sigmoid'), head(h,3,'softmax')))
    m.compile(optimizer=tf.keras.optimizers.AdamW(1e-5),
              loss={'output_0': 'mse', 'output_1': 'binary_crossentropy',
                    'output_2': 'categorical_crossentropy'})
    return m

# ─── Normalization ────────────────────────────────────────────────────────────
def _lognorm(x, u):
    a = np.copy(x).astype(float)
    ok = ~np.isnan(a)
    a[ok] = np.log(np.clip(a[ok], 0, None) + 1) / np.log(u + 1)
    return a

def nLogNorm_out(x, U):
    return np.stack([_lognorm(x[:, i], U[i]) for i in range(x.shape[1])], axis=1)

def to_model_input(values: np.ndarray, U: np.ndarray) -> np.ndarray:
    x    = nLogNorm_out(values, U)
    mask = (~np.isnan(values)).astype(float)
    x    = np.nan_to_num(x, nan=0.0)
    return np.stack([x, mask], axis=-1).astype(np.float32)

# ─── Constants ────────────────────────────────────────────────────────────────
FEATURES     = ['NMN', 'MN', 'MTY', 'VMA', 'HVA']
TUMOR_LABELS = ['Non-NB', 'Adrenal', 'Extra-Adrenal']
THRESHOLDS   = {
    '🔍 Screening  (Sens=1.00, NPV=1.00)':   0.616,
    '✅ Confirmatory (Spec=1.00, PPV=1.00)': 0.825,
}

@st.cache_resource
def load_model():
    m = build(5)
    m.load_weights('kfull_best.weights.h5')
    U = np.load('Upperout_train_ss3.npy')
    return m, U

# ─── Page ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title='De-NB', page_icon='🧬', layout='wide')
st.title('🧬 De-NB: Neuroblastoma Diagnostic Assistant')
st.caption('VAE + Self-Attention | Test AUPRC = 0.936 (95% CI: 0.771–1.000, n=13) | Research use only')

model, U = load_model()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header('⚙️ Clinical Mode')
    mode_label = st.radio('Select threshold', list(THRESHOLDS.keys()))
    threshold  = THRESHOLDS[mode_label]
    st.markdown('---')
    st.markdown('**🔍 Screening** — ใช้ตรวจแรก ถ้า Negative หยุดได้ (ไม่พลาด NB)')
    st.markdown('**✅ Confirmatory** — ถ้า Screening+ แล้วต้องการยืนยันก่อน imaging')
    st.markdown('---')
    st.caption('⚠️ Research use only. Validate before clinical deployment.')

# ─── Predict & display ────────────────────────────────────────────────────────
def show_results(raw: np.ndarray, pids: list):
    X_inp    = to_model_input(raw, U)
    preds    = model.predict(X_inp, verbose=0)
    score_nb = preds[1].flatten()
    prob_loc = preds[2]
    diag     = ['🔴 NB' if s >= threshold else '🟢 Not NB' for s in score_nb]
    loc      = [TUMOR_LABELS[np.argmax(p)] for p in prob_loc]

    st.markdown('---')
    st.subheader('Results')

    # per-patient
    for i, pid in enumerate(pids):
        c1, c2, c3 = st.columns([2, 2, 3])
        c1.metric(pid, diag[i], f'Score: {score_nb[i]:.4f}')
        c2.metric('Location', loc[i])
        with c3:
            fig = px.bar(
                x=[f'{prob_loc[i,j]*100:.1f}%' for j in range(3)],
                y=TUMOR_LABELS, orientation='h',
                text=[f'{prob_loc[i,j]*100:.1f}%' for j in range(3)],
                height=130
            )
            fig.update_layout(margin=dict(l=0,r=0,t=0,b=0),
                              showlegend=False, xaxis_title='', yaxis_title='')
            st.plotly_chart(fig, use_container_width=True, key=f'loc_{i}')

    # summary bar chart (ถ้ามากกว่า 1 patient)
    if len(pids) > 1:
        fig2 = go.Figure(go.Bar(
            x=pids, y=score_nb,
            marker_color=['red' if s >= threshold else 'green' for s in score_nb],
            text=[f'{s:.3f}' for s in score_nb], textposition='outside'
        ))
        fig2.add_hline(y=threshold, line_dash='dash',
                       annotation_text=f'Threshold {threshold}')
        fig2.update_layout(title='NB Score per Patient',
                           yaxis=dict(range=[0, 1.15]), height=350)
        st.plotly_chart(fig2, use_container_width=True)

    # table + download
    df_out = pd.DataFrame({
        'Patient':   pids,
        'Score':     score_nb.round(4),
        'Diagnosis': diag,
        'Location':  loc,
        **{f'P({TUMOR_LABELS[j]})': prob_loc[:,j].round(3) for j in range(3)}
    })
    st.dataframe(df_out, use_container_width=True)

    buf = BytesIO()
    df_out.to_excel(buf, index=False); buf.seek(0)
    st.download_button('⬇️ Download Excel', buf, 'denb_results.xlsx',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(['📝 Manual Input', '📊 Excel / CSV', '📁 NPY'])

with tab1:
    n = st.number_input('จำนวนผู้ป่วย', 1, 20, 1)
    rows = []
    for i in range(int(n)):
        st.markdown(f'**Patient {i+1}**')
        cols = st.columns(5)
        row  = []
        for j, feat in enumerate(FEATURES):
            val = cols[j].text_input(feat, key=f'inp_{i}_{j}',
                                     placeholder='blank = missing')
            row.append(float(val) if val.strip() else np.nan)
        rows.append(row)
    if st.button('🔬 Predict', type='primary', key='btn_manual'):
        show_results(np.array(rows, dtype=float),
                     [f'Patient {i+1}' for i in range(int(n))])
with tab2:
    st.markdown('คอลัมน์ต้องมี: `NMN, MN, MTY, VMA, HVA` (เว้นว่างได้)')
    st.download_button('⬇️ Template CSV',
        pd.DataFrame(columns=['Patient']+FEATURES).to_csv(index=False),
        'template.csv', 'text/csv')
    f = st.file_uploader('Upload', type=['xlsx','xls','csv'])
    if f:
        df_in = pd.read_csv(f) if f.name.endswith('.csv') else pd.read_excel(f)
        missing = [c for c in FEATURES if c not in df_in.columns]
        if missing:
            st.error(f'ไม่พบคอลัมน์: {missing}')
        else:
            st.dataframe(df_in[FEATURES].head())
            pids = df_in['Patient'].tolist() if 'Patient' in df_in.columns \
                   else [f'Patient {i+1}' for i in range(len(df_in))]
            if st.button('🔬 Predict', type='primary', key='btn_xl'):
                show_results(df_in[FEATURES].values.astype(float), pids)

with tab3:
    f2 = st.file_uploader('Upload .npy  shape [n,5,2]', type=['npy'])
    if f2:
        data  = np.load(f2)
        preds = model.predict(data, verbose=0)
        score = preds[1].flatten()
        df_npy = pd.DataFrame({
            'Patient':   [f'Patient {i+1}' for i in range(len(score))],
            'Score':     score.round(4),
            'Diagnosis': ['🔴 NB' if s >= threshold else '🟢 Not NB' for s in score],
            'Location':  [TUMOR_LABELS[np.argmax(p)] for p in preds[2]],
        })
        st.dataframe(df_npy, use_container_width=True)
