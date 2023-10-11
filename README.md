# InfoPattern

This demo is built based on `streamlit`. 

To run the demo, make sure the repo is on the `demo` branch.
```bash
git branch
# * demo
#   main
```

Follow steps below in `SwitchingLM/demo/`. 
## 1. Prepare the checkpoint and dependencies
The checkpoint can be downloaded here: ([Russia](https://drive.google.com/file/d/1AgeX0ipPb5oqugwOBWyGHDu0_l4ZtudJ/view?usp=share_link) | [LeftRight](https://drive.google.com/file/d/1K8_zFSuisYt5efO3lxHs9zvTcjPkP63D/view?usp=share_link)) and should be placed under `ckpt/`. By default, the [Russia](https://drive.google.com/file/d/1AgeX0ipPb5oqugwOBWyGHDu0_l4ZtudJ/view?usp=share_link) checkpoint should be used.
```bash
mkdir ckpt
mv <YOUR_DOWNLOADED_CKPT> ckpt
```

To install dependencies, install [pytorch](https://pytorch.org/get-started/locally/), [transformers](https://huggingface.co/docs/transformers/index), and [streamlit](https://streamlit.io/) at the official websites, or simply through package management tools such as pip.
```bash
pip install torch
pip install transformers
pip install streamlit
pip install st-annotated-text
pip install nltk
pip install accelerate
pip install graphviz
```

Prepare the punkt functionality
```python
python -c "import nltk; nltk.download('punkt')"
```
## 2. Run the application
Once the checkpoint and the dependencies are prepared, run
```bash
streamlit run Home.py
```

## 3. Navigate between the tabs and explore different functionalities
Check out [our slides](https://docs.google.com/presentation/d/1Rmqz8uDEI2rSwdLxo7BdwjKI8sprkBNz6zXS3fOwVPM/edit?usp=sharing) for the functionalities!
