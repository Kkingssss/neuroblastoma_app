import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

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
st.markdown("### Upload Plasma and Urine Metabolites (NPY file)")

# File uploader for NPY files
uploaded_file = st.file_uploader("Upload a NPY file", type="npy")
if uploaded_file is not None:
    # Load the NPY file
    data = np.load(uploaded_file)

    # Predict using the model
    predictions = model.predict(data)

    # Extract the predictions
    status_of_disease = predictions[1].flatten()
    location_of_tumor_probabilities = predictions[2]  # Shape: (num_samples, num_classes)

    # Define tumor location labels
    tumor_labels = ['Non - NB', 'Adrenal', 'Extra - Adrenal']

    # Get the predicted tumor location
    location_of_tumor = np.argmax(location_of_tumor_probabilities, axis=1)  # Get index of max probability
    location_of_tumor_labels = [tumor_labels[i] for i in location_of_tumor]

    # Determine the disease status
    status_of_disease_labels = ['NB ' if value > 0.8 else 'Not NB' for value in status_of_disease]

    # Create DataFrames for displaying predictions
    status_df = pd.DataFrame({
        'Patient Number': [f'Patient {i + 1}' for i in range(len(status_of_disease))],
        'Status of Disease Value': status_of_disease,
        'Status of Disease': status_of_disease_labels
    })

    location_df = pd.DataFrame({
        'Patient Number': [f'Patient {i + 1}' for i in range(len(location_of_tumor_labels))],
        'Location of Tumor': location_of_tumor_labels
    })

    # Create a summary column based on the location of tumor
    location_summary = ['Non - NB' if loc == 'Non - NB' else loc for loc in location_of_tumor_labels]

    summary_df = pd.DataFrame({
        'Patient Number': [f'Patient {i + 1}' for i in range(len(location_summary))],
        'Location of Tumor': location_summary
    })

    # Display the DataFrames with custom styling
    st.subheader("Status of Disease")
    st.dataframe(status_df.style.format({'Status of Disease Value': '{:.4f}'}))

    st.subheader("Location of Tumor")
    st.dataframe(location_df)

    # Adding a downloadable CSV link
    csv_status = status_df.to_csv(index=False)
    csv_location = location_df.to_csv(index=False)

    st.download_button(
        label="Download Status of Disease",
        data=csv_status,
        file_name='status_of_disease.csv',
        mime='text/csv'
    )

    st.download_button(
        label="Download Location of Tumor",
        data=csv_location,
        file_name='location_of_tumor.csv',
        mime='text/csv'
    )


