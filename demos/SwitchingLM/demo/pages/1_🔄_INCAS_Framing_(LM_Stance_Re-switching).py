import streamlit as st
import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))
from demo.utils import load_model_ckpt, generate, annotation_postprocess
from nltk.tokenize import sent_tokenize
from annotated_text import annotated_text
import warnings
import time
warnings.filterwarnings("ignore")  # dangerous, should be removed after the temporary use for the slider value issue.
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# define callbacks
def _update_example(_bias, _seed, _max, _min, _prompt):
    st.session_state.bias = _bias
    st.session_state.seed = _seed
    st.session_state.max = _max
    st.session_state.min = _min
    st.session_state.prompt = _prompt


if __name__ == '__main__':
    st.set_page_config(
        page_title="Generation with Stance Re-switching",
        page_icon="🔄",
        layout="wide",
    )


    # initialize the session state variable for bias
    if 'bias' not in st.session_state:
        st.session_state.bias = 0.0
    if 'seed' not in st.session_state:
        st.session_state.seed = '0'
    if 'max' not in st.session_state:
        st.session_state.max = '400'
    if 'min' not in st.session_state:
        st.session_state.min = '100'
    if 'prompt' not in st.session_state:
        st.session_state.prompt = 'Replace with your prompt'
    if 'initialization' not in st.session_state:
        st.session_state.initialization = True


    st.markdown('## 🔄 Generation with Stance Re-switching')
    with st.expander("ℹ️ - About this page", expanded=True):
        st.write(
            """     
    -   This page hosts the functionality of generating texts with given stances.
    -   It uses the a learnable "switch" in the embedding space to alter the trend of generation.
    -   Below are some examples for this demo
            """
        )
        st.markdown("")
    with st.expander("Examples", expanded=True):
        buttons = [
            {
                'label': 'Example 1',
                '_bias': 0.0,
                '_seed': '0',
                '_max': '400',
                '_min': '100',
                '_prompt': "Russia's annexation of Crimea"
            },
            {
                'label': 'Example 2',
                '_bias': 0.0,
                '_seed': '0',
                '_max': '400',
                '_min': '100',
                '_prompt': "NATO expansion"
            },
            {
                'label': 'Example 3',
                '_bias': 0.0,
                '_seed': '0',
                '_max': '400',
                '_min': '100',
                '_prompt': "Russia's regime"
            },
            {
                'label': 'Example 4',
                '_bias': 0.0,
                '_seed': '0',
                '_max': '400',
                '_min': '100',
                '_prompt': "The recent increase in sanctions against Russia"
            },
            {
                'label': 'Example 5',
                '_bias': 0.0,
                '_seed': '0',
                '_max': '400',
                '_min': '100',
                '_prompt': "Russian media's coverage of international events"
            },
            {
                'label': 'Example 6',
                '_bias': 0.0,
                '_seed': '0',
                '_max': '400',
                '_min': '100',
                '_prompt': "The handling of Minsk Accords"
            },
            {
                'label': 'Example 7',
                '_bias': 0.0,
                '_seed': '0',
                '_max': '400',
                '_min': '100',
                '_prompt': "The anti-Russian sentiment on the Internet"
            },
        ]
        texts = [
            "Russia's annexation of Crimea",
            "NATO expansion",
            "Russia's regime",
            "The recent increase in sanctions against Russia",
            "Russian media's coverage of international events",
            "The handling of Minsk Accords",
            "The anti-Russian sentiment on the Internet",
        ]
        for i in range(len(buttons)):
            c_button, c_text, c_anno = st.columns([1, 6.5, 1])
            with c_button:
                (st.button(
                    label=buttons[i]['label'],
                    on_click=_update_example,
                    kwargs={
                        '_bias': buttons[i]['_bias'],
                        '_seed': buttons[i]['_seed'],
                        '_max': buttons[i]['_max'],
                        '_min': buttons[i]['_min'],
                        '_prompt': buttons[i]['_prompt']
                    }
                ))
            with c_text:
                st.write(texts[i])
        with c_anno:
            use_annotation = st.checkbox('use annotation')
    st.markdown('### Let\'s Get Started!')


    with st.form(key="my_form"):
        ce1, c1, ce2, c2, ce3 = st.columns([0.07, 1, 0.07, 4, 0.07])
        with c1:
            st.slider(
                label="Stance",
                min_value=-1.0,
                max_value=1.0,
                key='bias',
                step=0.01,
                value=st.session_state.bias,
                help='You can specify the stance of the generation'
            )
            # st.text('left                     right')
            st.text('anti-                     pro-\nRussia                  Russia')
            st.text("")
            st.text("")
            random_seed = st.text_input(
                label='Seed',
                key='seed',
                help='You can specify the seed for generation',
                value='None',
            )
            if random_seed == 'None':
                random_seed = None
            else:
                random_seed = int(random_seed)
            
            max_length = st.text_input(
                label='Max Length',
                key='max',
                help='You can specify the longest length for generation',
                value=100,
            )
            max_length = int(max_length)

            min_length = st.text_input(
                label='Min Length',
                key='min',
                help='You can specify the shortest length for generation',
                value=20,
            )
            min_length = int(min_length)


        with c2:
            prompt = st.text_area(
                label='Paste your prompt below',
                key='prompt',
                value='Replace with your prompt',
                help='The beginning of the text to be generated',
                label_visibility="visible",
                height=350
            )
            submit_button = st.form_submit_button(label="Generate!", type='primary')


    if submit_button:
        variation = 'pro_anti_russia'
        ckpt_vocab = {
            'pro_anti_russia': "ckpt/230306_04_russia_6B.pt",
            'leftright': "ckpt/221122_02_1.3B_leftright_rank1000.pt"
        }

        # load model
        ckpt_name = ckpt_vocab[variation]
        model = load_model_ckpt(ckpt_name, device=device)
        model.eval()
        regularized_bias = st.session_state.bias if variation in ['pro_anti_russia'] else st.session_state.bias*3

        # initialization: load all examples
        if st.session_state.initialization:
            for prompt in texts:
                for bias in [-1.00, 0.00, 1.00]:
                    print(f'prompt: {prompt}, bias: {bias},', end=' ')
                    tic = time.time()
                    with torch.no_grad():
                        generated_text = generate(
                            _model=model,
                            prompt=prompt, 
                            stance_value=bias, 
                            topic_name=None,
                            seed=random_seed,
                            max_length=400,
                            min_length=100
                        )
                    toc = time.time()
                    print('time:', toc-tic)
            prompt = texts[0]
            for bias in [-0.50, -0.30, -0.20, 0.50]:
                print(f'prompt: {prompt}, bias: {bias},', end=' ')
                tic = time.time()
                with torch.no_grad():
                    generated_text = generate(
                        _model=model,
                        prompt=prompt, 
                        stance_value=bias, 
                        topic_name=None,
                        seed=random_seed,
                        max_length=400,
                        min_length=100
                    )
                toc = time.time()
                print('time:', toc-tic)
            print('initialization finished')
            st.session_state.initialization = False


        # inference
        with st.spinner('Model loading finished, inference in progress...'):
            with torch.no_grad():
                print(f'prompt: {prompt}, bias: {regularized_bias}, type: {type(regularized_bias)}', end=' ')
                generated_text = generate(
                    _model=model,
                    prompt=prompt, 
                    stance_value=regularized_bias, 
                    topic_name=None,
                    seed=random_seed,
                    max_length=max_length,
                    min_length=min_length
                )
        st.success('Inference Compeleted!')

        if variation == 'leftright':
            generated_text = generated_text[0]['generated_text'] 
        # post-process
        sents = sent_tokenize(generated_text)
        if len(sents) > 1 and not any (sents[-1].endswith(_) for _ in ['.', '?', '!', '"']):
            sents = sents[:-1]
        generated_text = ' '.join(sents) 


        # analyze
        if use_annotation and variation in ['pro_anti_russia']:
            annotated = model.evidence_words(
                prompt=generated_text,
                original_switch_values=[st.session_state.bias],
                max_segments=min(len(generated_text) // 150, 5),
                max_length=10
            )
            annotated = annotation_postprocess(annotated)
            annotated_text(*annotated)
            
        else:
            # display
            st.markdown(f'### The generated text:')
            st.text_area(
                label='Generated Text',
                value=generated_text,
                help='The generated text',
                label_visibility="collapsed",
                height=200
            )

