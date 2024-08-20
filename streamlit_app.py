import streamlit as st
import tensorflow as tf
import numpy as np

# Define your model structure
Dense = tf.keras.layers.Dense
Attn = tf.keras.layers.MultiHeadAttention
Dropout = tf.keras.layers.Dropout

L2Norm = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='L2norm')

class sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        z = tf.random.normal(shape=tf.shape(mean))
        return mean + z * tf.exp(0.5 * log_var)

def build(N):
    inp = tf.keras.Input((N, 2))
    x = inp
    x = Dense(32, activation='relu')(x)
    x0 = x
    x, _ = Attn(1, 128, name='attention0')(x, x, return_attention_scores=True)
    x = tf.keras.layers.Concatenate()([x0, x])
    x = tf.keras.layers.Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = [Dense(64, activation='relu')(x) for _ in range(2)]
    x = sampling()(x)
    x = L2Norm(x)
    h = x

    x = Dense(128, activation='relu')(h)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    outR = Dense(N, activation=None, name='outR')(x)

    x = Dense(128, activation='relu')(h)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    out1 = Dense(1, activation='sigmoid', name='out1')(x)

    x = Dense(128, activation='relu')(h)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    out2 = Dense(3, activation='softmax', name='out2')(x)

    x = tf.keras.Model(inp, (outR, out1, out2))
    x.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5),
              loss={'outR': tf.keras.losses.MeanSquaredError(),
                    'out1': tf.keras.losses.BinaryFocalCrossentropy(),
                    'out2': tf.keras.losses.CategoricalCrossentropy()},
              metrics={'outR': ['mae', 'mse'],
                       'out1': [tf.keras.metrics.AUC(name='auc1', curve='PR')],
                       'out2': [tf.keras.metrics.AUC(name='auc2', curve='PR')]})
    return x

nPer = lambda x, p: np.percentile(x[~np.isnan(x)], p, axis=0)

def nOut(x, k=1.5):
    q25,q75 = nPer(x, 25),nPer(x, 75)
    iqr = q75-q25
    return q25 - k*iqr, q75 + k*iqr

def lognorm(x, u):
    x = np.clip(x, 0, np.max(x))
    return np.log(x+1)/np.log(u+1)

def nLogNorm(x, u):
    a = np.copy(x)
    aw = np.where(~np.isnan(a))
    a[aw] = lognorm(a[aw], u)
    return a

def nLogNorm_out(x, U=None, k=1.5):
    A = []
    if np.any(U==None):
        U = []
        for i in range(x.shape[1]):
            upper = nOut(x[:,i], k)[1]
            a = nLogNorm(x[:,i], upper)
            A.append(a)
            U.append(upper)
    else:
        for i in range(x.shape[1]):
            upper = U[i]
            a = nLogNorm(x[:,i], upper)
            A.append(a)
    return np.stack(A, axis=1), np.array(U)

# Load the model
model = build(5)
fold = 'full'
model.load_weights(f'k{fold}_best.weights.h5')

# Streamlit app title and description
st.title("Neuroblastoma Prediction")

st.write("Upload Plasma and Urine Metabolites (NPY file)")

# File uploader for NPY files
uploaded_file = st.file_uploader("Upload a NPY file", type="npy")
if uploaded_file is not None:
    # Load the NPY file
    data = np.load(uploaded_file)

   
    # Predict using the model
    predictions = model.predict(data)

    # Display predictions
    st.write("Predictions:")
    st.write("Status of Disease:", predictions[1])
    st.write("Location of Tumor:", predictions[2])
        
        #st.write("Tumor Classification:", predictions[2])
