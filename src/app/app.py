import os
import sys
sys.path.append(os.path.abspath('.\\'))

import streamlit as st
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from src.inference.inference import InferenceAgent
from src.config.config import Config

# Create the inference agent
config = Config()
inference_agent = InferenceAgent(config=config)

# Read data to set un numeric parameters
df = pd.read_csv(config.file_path)
df['date'] = df['date'].apply(
    lambda x: datetime.strptime(x, '%Y-%m-%d')
)

# Read rules
BASE_PATH = os.path.abspath('.\\data')
fname = os.path.join(BASE_PATH, 'gold', 'rules.pkl')
print(fname)

rules = {} # rules is an empty dict already

if os.path.getsize(fname) > 0:      
    with open(fname, "rb") as f:
        unpickler = pickle.Unpickler(f)
        # if file is not empty scores will be equal
        # to the value unpickled
        rules = unpickler.load()



st.set_page_config(layout="wide")
st.header('Sales Prediction App')
# Create the top row
upper_left, upper_right = st.columns(2)

# Upper-left area
with upper_left:
    st.subheader("Input Parameters")

    cpi = st.number_input(
        "CPI",
        min_value=np.min(df['cpi']),
        max_value=np.max(df['cpi']),
        value=np.mean(df['cpi']))
    
    unemployment_rate = st.number_input(
        "Unemployment rate",
        min_value=np.min(df['unemployment_rate']),
        max_value=np.max(df['unemployment_rate']),
        value=np.mean(df['unemployment_rate'])
    )

    previous_sales = st.number_input(
        "Sales from previous month",
        min_value=np.min(df['previous_sales']),
        max_value=np.max(df['previous_sales']),
        value=np.mean(df['previous_sales'])
    )

    month = st.selectbox(
        "Month to make prediction",
        [str(m) for m in df['month'].unique()]
    )

    productcategory = st.selectbox(
        "Product Category",
        df['productcategory'].unique()
    )

    company_type = st.selectbox(
            "Company type",
            df['company_type'].unique()
        )

    company_employees = st.selectbox(
        "Company employees",
        df['company_employees'].unique()
    )

    calculate = st.button("Make Prediction")
    if calculate:
        pred = inference_agent.inference(
            month = month,
            productcategory = productcategory,
            company_type = company_type,
            company_employees = company_employees,
            previous_sales=previous_sales,
            cpi=cpi,
            unemployment_rate = unemployment_rate)
        st.markdown(f'***Predicions: {int(pred)} Sales!***')



with upper_right:
    st.subheader("Main Product Rules")
    caption_html = "<br>".join([str(r) for r in rules[:10]])

    st.markdown(f"""
    <style>
    .upper-right {{
        position: fixed;
        top: 60px;
        left: 20px;
        z-index: 999999;
        background-color: white;
        padding: 10px 15px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        font-size: 14px;
        color: #555;
    }}

    .upper-right-title {{
        display: block;
        font-size: 18px;
        font-weight: bold;
        color: black;
        margin-bottom: 8px;
    }}
    </style>

    <div class="upper-left">
        <div>{caption_html}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

st.subheader("Sales Evolution")

fig, ax = plt.subplots(1,1,figsize=(20, 5))
sns.lineplot(
    data=df,
    x='date',
    y='previous_sales',
    hue='company_type',
    ax=ax
)
plt.grid()
plt.tight_layout()
st.pyplot(fig, use_container_width=True, width='stretch')

