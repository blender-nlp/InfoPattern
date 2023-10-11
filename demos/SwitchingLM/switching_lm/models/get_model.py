from switching_lm.models.model_gpt_neo import Switching_GPTNeoModel
from switching_lm.models.model_gpt_j import Switching_GPTJModel
from switching_lm.models.model_dialogpt import Switching_DialoGPTModel


def get_model(model_name, adapted_component, num_switches, rank, epsilon,
              init_var, embedding_dim=None, low_resource_mode=False):
    if model_name.startswith("EleutherAI/gpt-neo") or model_name == "gpt2":
        model = Switching_GPTNeoModel(
            model_name, adapted_component, num_switches, rank, epsilon,
            init_var, embedding_dim, low_resource_mode)
        return model, model.tokenizer
    elif model_name.startswith("EleutherAI/gpt-j"):
        model = Switching_GPTJModel(
            model_name, adapted_component, num_switches, rank, epsilon,
            init_var, embedding_dim, low_resource_mode)
        return model, model.tokenizer
    elif model_name.startswith("microsoft/DialoGPT"):
        model = Switching_DialoGPTModel(
            model_name, adapted_component, num_switches, rank, epsilon,
            init_var, embedding_dim)
        return model, model.tokenizer
    else:
        raise NotImplementedError()
