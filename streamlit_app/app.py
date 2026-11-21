"""Streamlit app: predict a single-family home's sale price from four inputs.

Run locally:  streamlit run app.py
Expects xgb_price_bundle.joblib (from section 4 of 07_streamlit_app.ipynb) in the same folder.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BUNDLE_PATH = Path(__file__).parent / 'xgb_price_bundle.joblib'


@st.cache_resource
def load_bundle(path):
    return joblib.load(path)


def add_derived(row, inputs, defaults):
    """Fill every model column that depends on the four inputs.

    Must match add_derived() in 07_streamlit_app.ipynb, which checks each formula against training data.
    """
    living, beds, baths, lot = (float(inputs[k]) for k in ['LivingArea', 'Beds', 'Baths', 'LotSize'])
    row['log_LivingArea'] = np.log1p(living)
    row['log_LotSizeSquareFeet'] = np.log1p(lot)
    row['BedroomsTotal'] = beds
    row['BathroomsTotalInteger'] = baths
    row['BedBathRatio'] = beds / max(baths, 1)
    row['SqftPerBedroom'] = living / max(beds, 1)
    row['LotCoverage'] = living / lot if lot > 0 else np.nan
    # Price estimates are log(ppsf) + log size, so they move with living area
    row['Comp_logprice_est'] = row['Comp_ppsf'] + row['log_LivingArea']
    row['Nbhd_logprice_est'] = row['Nbhd_ppsf_6m'] + row['log_LivingArea']
    # Size gap to comps: shift the typical gap by how far this home is from the median size
    row['Comp_size_diff'] = defaults['Comp_size_diff'] + row['log_LivingArea'] - defaults['log_LivingArea']
    return row


def predict_price(bundle, inputs):
    defaults = bundle['defaults']
    row = add_derived(dict(defaults), inputs, defaults)
    X = pd.DataFrame([row]).reindex(columns=bundle['features']).astype(float)
    log_pred = bundle['model'].predict(X)[0]
    return float(np.expm1(log_pred))     # model was trained on log1p(ClosePrice)


st.set_page_config(page_title='Home Price Estimator', page_icon='🏠')
st.title('Home Price Estimator')
st.caption('XGBoost trained on CRMLS single-family sales. Features not entered below '
           'are held at their training median.')

if not BUNDLE_PATH.exists():
    st.error(f'Model file not found: {BUNDLE_PATH.name}. Run the export cell in Colab and '
             'place the file next to app.py.')
    st.stop()

bundle = load_bundle(BUNDLE_PATH)

with st.form('inputs'):
    c1, c2 = st.columns(2)
    living = c1.number_input('Living area (sq ft)', min_value=200, max_value=20000, value=1800, step=50)
    lot = c2.number_input('Lot size (sq ft)', min_value=100, max_value=500000, value=6000, step=100)
    beds = c1.number_input('Bedrooms', min_value=1, max_value=15, value=3, step=1)
    baths = c2.number_input('Bathrooms', min_value=1, max_value=15, value=2, step=1)
    submitted = st.form_submit_button('Predict price')

if submitted:
    inputs = {'LivingArea': living, 'Beds': beds, 'Baths': baths, 'LotSize': lot}

    unusual = [k for k, v in inputs.items()
               if not bundle['ranges'][k][0] <= v <= bundle['ranges'][k][1]]
    if unusual:
        st.warning('Outside the range of 98% of training homes: ' + ', '.join(unusual) +
                   '. Treat this estimate with extra caution.')

    price = predict_price(bundle, inputs)
    st.metric('Predicted sale price', f'${price:,.0f}')
    # Test-set MdAPE was about 7%, so half of homes land within roughly this band
    st.caption(f'Typical range (±7%): ${price * 0.93:,.0f} to ${price * 1.07:,.0f}')
