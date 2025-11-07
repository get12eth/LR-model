import streamlit as st
import joblib
import pandas as pd
import io
import json
import time
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Load Assets ---
try:
    model_path = 'models/logistic_regression_model.joblib'
    scaler_path = 'models/standard_scaler_lr.joblib'
    encoders_path = 'models/label_encoders_lr.joblib'

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoders = joblib.load(encoders_path)
    
    st.sidebar.success("Model and Preprocessing assets loaded successfully.")
except FileNotFoundError:
    st.error("Error: Model or preprocessing files not found. Please ensure 'logistic_regression_model.joblib', 'standard_scaler_lr.joblib', and 'label_encoders_lr.joblib' are in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading assets: {e}")
    try:
        import altair as alt
    except Exception:
        alt = None
    st.stop()

# --- 2. Define Feature Lists (Based on Training Features) ---

expected_features = [
    'loan_limit', 'Gender', 'approv_in_adv', 'loan_type', 'loan_purpose', 
    'Credit_Worthiness', 'open_credit', 'business_or_commercial', 'rate_of_interest', 
    'Interest_rate_spread', 'Upfront_charges', 'term', 'Neg_ammortization', 
    'interest_only', 'lump_sum_payment', 'construction_type', 'occupancy_type', 
    'Secured_by', 'total_units', 'income', 'credit_type', 'Credit_Score', 
    'co-applicant_credit_type', 'age', 'submission_of_application', 'LTV', 
    'Region', 'Security_Type', 'dtir1'
]

#Separate features into the two columns for rendering
#Column 1 Features: Loan Terms & Rates (numerical & categorical)

col1_features = [
    'loan_limit', 'Gender', 'approv_in_adv', 'loan_type', 'loan_purpose', 
    'Credit_Worthiness', 'business_or_commercial', 
    'rate_of_interest', 'Interest_rate_spread', 'Upfront_charges', 'term'
]

#Column 2 Features: Applicant & Property Details (numerical & categorical)

col2_features = [
    'Neg_ammortization', 'interest_only', 'lump_sum_payment', 'construction_type', 
    'occupancy_type', 'Secured_by', 'total_units', 'credit_type', 
    'co-applicant_credit_type', 'age', 'submission_of_application', 'Region', 
    'Security_Type', 'open_credit', 
    'income', 'Credit_Score', 'LTV', 'dtir1'
]

#Numerical and Categorical feature lists for preprocessing

numerical_cols = [
    'rate_of_interest', 'Interest_rate_spread', 'Upfront_charges', 'term', 
    'income', 'Credit_Score', 'LTV', 'dtir1'
]
categorical_cols = [
    'loan_limit', 'Gender', 'approv_in_adv', 'loan_type', 'loan_purpose', 
    'Credit_Worthiness', 'open_credit', 'business_or_commercial', 'Neg_ammortization', 
    'interest_only', 'lump_sum_payment', 'construction_type', 'occupancy_type', 
    'Secured_by', 'total_units', 'credit_type', 'co-applicant_credit_type', 
    'age', 'submission_of_application', 'Region', 'Security_Type'
]


# --- 3. Preprocessing Function (Crucial for Deployment) ---

def preprocess_input(input_df, scaler, label_encoders, numerical_cols, categorical_cols):
    df = input_df.copy()

    df = df.fillna(0) 

    #Minimal imputation for app stability

    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace('Sex Not Available', 'Male') 

    # --- Encoding Categorical Features ---
    for col in categorical_cols:
        if col in df.columns and col in label_encoders:
            le = label_encoders[col]
            known_labels = {label: idx for idx, label in enumerate(le.classes_)}
            
            def get_encoded_value(value):
                return known_labels.get(value, 0) 
            
            df[col] = df[col].apply(get_encoded_value)

    # --- Scaling Numerical Features ---

    if numerical_cols:
        df[numerical_cols] = df[numerical_cols].astype(float) 
        df[numerical_cols] = scaler.transform(df[numerical_cols])

    return df[expected_features]

# --- 4. Streamlit UI and Input Form ---

st.set_page_config(page_title="Loan Default Risk Predictor", layout="wide")

# session state defaults
if 'page' not in st.session_state:
    st.session_state['page'] = 'Inputs'
if 'theme' not in st.session_state:
    st.session_state['theme'] = 'Dark'

def apply_theme():

    #simple CSS switch based on session_state['theme']

    if st.session_state['theme'] == 'Light':
        bg1, bg2, text = '#f7fafc', '#e6eef8', '#0b1220'
    else:
        bg1, bg2, text = '#0b1220', '#071028', '#e6eef8'
    st.markdown(f"""
    <style>
    .reportview-container .main {{background: linear-gradient(180deg, {bg1} 0%, {bg2} 100%); color: {text};}}
    .muted {{color: #6b7280}}
    .result-card {{border-radius:12px; padding:16px; color: {text};}}
    .result-card .big {{font-size:36px; font-weight:800}}
    </style>
    """, unsafe_allow_html=True)

apply_theme()

#Sidebar: left-aligned navigation & tools

st.sidebar.title("LoanGuard")
nav = st.sidebar.radio("Navigation", ['Overview', 'Inputs', 'Results', 'Settings'], index=['Overview','Inputs','Results','Settings'].index(st.session_state['page']))
st.session_state['page'] = nav

st.sidebar.markdown('---')
st.sidebar.header('Model & Tools')
st.sidebar.markdown(f"**Model:** `{model_path.split('/')[-1]}`  \n**Scaler:** `{scaler_path.split('/')[-1]}`")

#Example scenarios (small set)

EXAMPLES = {
    'Select an example': {},
    'High risk — low score, high LTV': {
        'income': 2500, 'Credit_Score': 320, 'LTV': 95.0, 'dtir1': 60.0, 'loan_amount':400000, 'property_value':420000
    },
    'Payment shock — interest-only style': {
        'income': 3000, 'Credit_Score': 550, 'LTV': 100.0, 'dtir1': 48.0, 'loan_amount':350000
    },
}

chosen_example = st.sidebar.selectbox("Load Example", list(EXAMPLES.keys()))
if st.sidebar.button("Apply Example") and chosen_example and chosen_example != 'Select an example':
    example = EXAMPLES[chosen_example]
    for feat, val in example.items():
        key = f"num_{feat}" if feat in numerical_cols else f"cat_{feat}"
        st.session_state[key] = val
    # trigger a rerun in a way that's compatible across Streamlit versions
    try:
        st.experimental_rerun()
    except Exception:
        # fallback: change query params to force a rerun (use non-deprecated API)
        st.query_params = {'_rerun': str(time.time())}

# Theme toggle
theme_choice = st.sidebar.selectbox('Theme', ['Dark','Light'], index=0 if st.session_state['theme']=='Dark' else 1)
if theme_choice != st.session_state['theme']:
    st.session_state['theme'] = theme_choice
    apply_theme()

#small footer in sidebar

st.sidebar.markdown('---')
st.sidebar.caption('Built with Streamlit — LoanGuard AI')

# layout columns for Inputs page
c1, c2 = st.columns(2)
input_data = {}

# --- Input Rendering Logic (Ensures No Duplication) ---




def render_inputs(col_container, features_list):
    for col in features_list:
        
        #Determine if the feature is numerical or categorical
        is_numerical = col in numerical_cols
        
        # --- SPECIAL HANDLERS ---
        if col == 'Credit_Score':
             input_data[col] = col_container.number_input(
                f"**{col.replace('_', ' ').title()}**",
                min_value=300.0,
                max_value=900.0,
                value=700.0,
                step=1.0,
                key=f'num_{col}',
                help="FICO-like score, typically 300 to 850."
            )
        elif col == 'term':
            input_data[col] = col_container.selectbox(
                f"**{col.replace('_', ' ').title()}** (Months)",
                options=[180.0, 360.0, 480.0],
                index=1,
                key=f'num_{col}',
                help="Duration of the loan in months (e.g., 360 for 30 years)."
            )
        
        # --- CATEGORICAL INPUTS ---

        elif not is_numerical:
            options = list(label_encoders[col].classes_)
            if col == 'Gender' and 'Sex Not Available' in options:
                options.remove('Sex Not Available')

            input_data[col] = col_container.selectbox(
                f"**{col.replace('_', ' ').title()}**", 
                options,
                key=f'cat_{col}'
            )

        # --- GENERAL NUMERICAL INPUTS ---
        else:
            input_data[col] = col_container.number_input(
                f"**{col.replace('_', ' ').title()}**",
                value=0.0,
                step=0.01,
                format="%.4f",
                key=f'num_{col}'
            )


def render_inputs_page():
    st.subheader("Loan Terms and Amounts")
    left, right = st.columns(2)
    with left:
        render_inputs(left, col1_features)
    with right:
        render_inputs(right, col2_features)

    # Prediction button
    if st.button("Predict Default Risk", type="primary"):
        # build input_df from input_data (widgets already populated input_data)
        input_df = pd.DataFrame([input_data])
        try:
            X_processed = preprocess_input(
                input_df,
                scaler,
                label_encoders,


                numerical_cols,
                categorical_cols
            )
        except Exception as e:
            st.error(f"Error during preprocessing: {e}")
            return

        #Predict
        probabilities = model.predict_proba(X_processed.astype(float))[0]
        default_prob = float(probabilities[1])
        prediction = int(model.predict(X_processed.astype(float))[0])

        # Save last results to session_state for Results page / export
        st.session_state['last_input_csv'] = input_df.to_csv(index=False)
        st.session_state['last_processed_csv'] = X_processed.to_csv(index=False)
        st.session_state['last_prediction'] = prediction
        st.session_state['last_prob'] = default_prob

        # Redirect to Results page then rerun
        st.session_state['page'] = 'Results'
        try:
            st.experimental_rerun()
        except Exception:
            # fallback: update query params using the supported API
            st.query_params = {'_rerun': str(time.time())}




def render_results_page():
    st.subheader("Prediction Results & Exports")
    if 'last_input_csv' not in st.session_state:
        st.info("No prediction yet — run a prediction from the Inputs page or load an example.")
        return

    prob = st.session_state.get('last_prob', 0.0)
    pred = st.session_state.get('last_prediction', 0)

    # prominent card
    colL, colR = st.columns([3,2])
    with colL:
        verdict = 'HIGH RISK' if pred == 1 else 'LOW RISK'
        color = '#ff6b6b' if pred == 1 else '#06b6d4'
        st.markdown(f"<div class='result-card' style='background:{color}'>"
                    f"<div class='big'>{prob:.1%}</div>"
                    f"<div style='font-weight:700'>{verdict}</div>"
                    f"</div>", unsafe_allow_html=True)
    with colR:
        st.markdown("**Exports**")
        st.download_button("Download Input CSV", st.session_state['last_input_csv'], file_name='input.csv', mime='text/csv')
        st.download_button("Download Processed CSV", st.session_state['last_processed_csv'], file_name='processed.csv', mime='text/csv')
    result_json = { 'prediction': int(pred), 'probability': prob }
    st.download_button("Download Result JSON", data=json.dumps(result_json), file_name='result.json', mime='application/json')

    # Show processed data head
    st.markdown('---')
    st.markdown('**Processed features (first rows)**')
    processed_df = pd.read_csv(io.StringIO(st.session_state['last_processed_csv']))
    # --- Feature importance visualization ---
    try:
        imp = None
        if hasattr(model, 'feature_importances_'):
            imp = np.array(model.feature_importances_)
        elif hasattr(model, 'coef_'):
            coefs = np.array(model.coef_)
            if coefs.ndim == 1:
                imp = coefs
            else:
                # prefer class 1 coefficients when available (probability of class 1 is used elsewhere)
                if coefs.shape[0] > 1:
                    imp = coefs[1]
                else:
                    imp = coefs[0]

        if imp is not None and len(imp) == len(processed_df.columns):
            abs_imp = pd.Series(np.abs(imp), index=processed_df.columns)
            top = abs_imp.sort_values(ascending=False).head(10)

            st.markdown('**Top feature importances**')
            fig, ax = plt.subplots(figsize=(8, max(3, len(top) * 0.5)))
            top.sort_values().plot(kind='barh', color='#06b6d4', ax=ax)
            ax.set_xlabel('Absolute importance')
            ax.set_ylabel('Feature')
            ax.grid(axis='x', linestyle='--', alpha=0.4)
            st.pyplot(fig)
        else:
            st.info('Feature importance not available or shape mismatch with model coefficients.')
    except Exception as e:
        st.warning(f'Could not compute feature importances: {e}')
    st.dataframe(processed_df.head())





def render_overview_page():
    st.subheader('Overview')
    st.markdown('Welcome to LoanGuard — use the Inputs tab to enter borrower data and hit Predict. Results and export options are under Results.')
    if 'last_prob' in st.session_state:
        st.metric('Last Default Probability', f"{st.session_state['last_prob']:.1%}")


def render_settings_page():
    st.subheader('Settings')
    theme = st.selectbox('Theme', ['Dark','Light'], index=0 if st.session_state['theme']=='Dark' else 1)
    if st.button('Apply Theme'):
        st.session_state['theme'] = theme
        apply_theme()


# Page dispatcher
if st.session_state['page'] == 'Overview':
    render_overview_page()
elif st.session_state['page'] == 'Inputs':
    render_inputs_page()
elif st.session_state['page'] == 'Results':
    render_results_page()
elif st.session_state['page'] == 'Settings':
    render_settings_page()