Neuroblastoma Prediction Streamlit App
This repository contains a Streamlit application that predicts outcomes based on Plasma and Urine Metabolites data using a TensorFlow model. The app allows users to upload a document and receive predictions generated by the model.

Features
Upload a document (.txt) containing Plasma and Urine Metabolites data.
The TensorFlow model processes the data and outputs predictions for:
NB status
Location of Tumor

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
