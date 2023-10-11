import streamlit as st
import torch

from typing import List, Tuple
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))
from switching_lm.models.get_model import get_model


@st.cache_resource(show_spinner="Model loading started, this one-time loading process might take a few minutes...")
def load_model_ckpt(ckpt_name: str, device=torch.device("cuda:0")):
    args, state_dict = torch.load(ckpt_name, map_location=device)
    args.low_resource_mode = True
    model, _ = get_model(
        args.model_name, args.adapted_component, 1,
        args.rank, args.epsilon, args.init_var,
        low_resource_mode=getattr(args, 'low_resource_mode', False))
    model.load_state_dict(state_dict)
    model.to_device(device)
    return model


@st.cache_data(max_entries=100, show_spinner=False)
def generate(_model, prompt, stance_value, topic_name,
             min_length=100, max_length=400, seed=None):
    return _model.generate(
        prompt,
        [stance_value],
        min_length, max_length, seed
    )


def annotation_postprocess(segments: List[Tuple]):
    res = []
    colors = {
        'blue': '#bde0fe',
        'red': '#f4acb7',
    }
    for seg in segments:
        text, color = seg
        if color is not None:
            res.append((text, '', colors[color]))
        else:
            res.append(text)
    return res


def analyze(model, prompt, min_value=-1, max_value=1, bins=5):
    return model.switch_analysis(
        prompt, 0, min_value, max_value, bins
    )