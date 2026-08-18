import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os


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


title = "Main Product Rules"
caption_html = "<br>".join([str(r) for r in rules[:10]])

st.markdown(f"""
<style>
.upper-left {{
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

.upper-left-title {{
    display: block;
    font-size: 18px;
    font-weight: bold;
    color: black;
    margin-bottom: 8px;
}}
</style>

<div class="upper-left">
    <div class="upper-left-title">{title}</div>
    <div>{caption_html}</div>
</div>
""", unsafe_allow_html=True)