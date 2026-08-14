import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os


BASE_PATH = os.path.abspath('.\data')



with open(os.path.join(BASE_PATH, 'gold', 'rules.pkl')) as fp:
    rules = pickle.load(fp)

st.write(rules)

