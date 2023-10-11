import streamlit as st
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))
from demo.utils import load_model_ckpt, analyze
import pandas as pd
import altair as alt
import torch


st.set_page_config(
    page_title="Text Stance Scoring",
    page_icon="💯",
    layout="wide",
)

# "Maybe check back later! :D"
st.markdown('## 💯 Text Stance Scoring')

with st.expander("ℹ️ - About this page", expanded=True):
    st.write(
        """     
-   This page hosts the functionality of evalutating the stance of a specific text.
-   It uses generating scores as the metric for analysis.
	    """
    )
    st.markdown("")

st.markdown('### Let\'s Get Started!')

with st.form(key="my_form"):
    text = st.text_area(
        label='Paste your text below', 
        value='We should be more concerned with promoting equality, protecting freedom, and caring about our environment.',
        help='The text to be analyzed',
        label_visibility="visible",
        height=90
    )
    submit_button = st.form_submit_button(label="Analyze!")

if submit_button:
    variation = 'pro_anti_russia'
    ckpt_vocab = {
        'pro_anti_russia': "ckpt/230306_04_russia_6B.pt",
        'leftright': "ckpt/221122_02_1.3B_leftright_rank1000.pt"
    }
    stance_vocab = {
        'pro_anti_russia': ['Anti', 'Lean Anti', 'Center', 'Lean Pro', 'Pro'],
        'leftright': ['Far Left', 'Left', 'Lean Left', 'Center', 'Lean Right', 'Right', 'Far Right']
    }
    viz_vocab = {
        'pro_anti_russia': {
            'min_value': -1,
            'max_value': 1,
            'bins': 5
        },
        'leftright': {
            'min_value': -3,
            'max_value': 3,
            'bins': 7
        },
    }

    # load model
    ckpt_name = ckpt_vocab[variation]
    model = load_model_ckpt(ckpt_name, device=torch.device('cuda:0'))
    results = analyze(model=model, prompt=text, **viz_vocab[variation])
    note_inference = st.markdown('Model loading finished, inference started.')
    note_inference.empty()

    chart_data = pd.DataFrame.from_dict(
        {
            'stance': stance_vocab[variation],
            'probability': [_[1] for _ in results[1]]
        },
        orient='columns'
    )
    st.text("")
    st.text("")

    c1, ce, c2 = st.columns([1, 0.5, 1])
    with c1:
        st.write('**Stance Distribution**')
        st.altair_chart(alt.Chart(chart_data).mark_bar().encode(
            x=alt.X('stance', sort=None),
            y='probability',
        ), use_container_width=True)
    with c2:
        st.write('**Evidence Segments**')
        st.write([
            seg[0].strip(' ') for seg in results[2] if seg[1] is not None
        ])
        # st.write(results)
    # print(results)
