import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import pipeline

from .model_utils import Hack_no_grad
from .switches import Projected_Adaptor
from switching_lm.utils import set_seed


punctuations = [
    '!', '"', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.',
    # '/', '#',
    ':', ';', '<', '=', '>', '?', '@',
    '[', '\\', ']', '^', '_', '`',
    '{', '|', '}', '~',
    '¨', '©', 'ª', '«', '¬', '®', '¯', '°', '±', '²', '³', '´', 'µ', '¶', '·',
    '¸', '¹', 'º', '»', '¼', '½', '¾',
    '\n', ' ',
]


class Switching_GPTNeoModel(nn.Module):
    def __init__(self, model_name, adapted_component,
                 num_switches, rank, epsilon, init_var,
                 use_embedding, low_resource_mode):
        # TODO: use_embedding is ignored now
        super().__init__()
        self.adapted_component = adapted_component
        self.generator = pipeline('text-generation', model=model_name)
        self.tokenizer = self.generator.tokenizer
        self.model = self.generator.model
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.init_var = init_var
        self.num_switches = num_switches
        self.device = torch.device("cpu")
        embed_dim = self.model.lm_head.weight.shape[1]
        vocab_size = self.model.lm_head.weight.shape[0]

        for _param in self.model.parameters():
            _param.requires_grad_(False)

        if adapted_component == "final_layer":
            self.model.transformer = Hack_no_grad(self.model.transformer)
            self.switch = Projected_Adaptor(
                self.model.lm_head, num_switches, embed_dim, vocab_size, rank,
                epsilon, init_var, "output")
            self.model.set_output_embeddings(self.switch)
        elif adapted_component == "input_embedding":
            self.switch = Projected_Adaptor(
                self.model.transformer.wte, num_switches, embed_dim,
                vocab_size, rank, epsilon, init_var, "input")
            self.model.transformer.set_input_embeddings(self.switch)
        else:
            raise NotImplementedError()

    def forward(self, input_ids, attention_mask, switch_values):
        self.switch.set_value(switch_values)
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids)
        return output

    def parameters(self):
        return self.switch.parameters()

    def state_dict(self):
        return self.switch.state_dict()

    def load_state_dict(self, state_dict):
        self.switch.load_state_dict(state_dict)

    def to_device(self, device):
        self.generator.device = device
        self.model.to(device)
        self.device = device

    def regularization_term(self):
        return self.switch.regularization_term()

    def generate(self, prompt, switch_values, min_length=20, max_length=100,
                 seed=None):
        '''
        prompt: a string
        switch_values
        min_length: minimum generation length
        max_length: maximum generation length
        seed: seed for generation. None if not specified.
        '''
        if seed is not None:
            set_seed(seed)
        switch_values = torch.Tensor(switch_values).to(
            self.device)
        self.switch.set_value(switch_values[None])
        with torch.no_grad():
            text = self.generator(
                prompt,
                do_sample=True, min_length=min_length, max_length=max_length,
                pad_token_id=self.tokenizer.pad_token_id)
        return text

    def switch_analysis(self, prompt, switch_dim, min_value=-3, max_value=3,
                        bins=7):
        tokenized = self.tokenizer(prompt)
        input_ids = torch.LongTensor(tokenized["input_ids"]).to(self.device)
        input_ids = input_ids.expand(bins + 1, -1)
        attention_mask = torch.LongTensor(tokenized["attention_mask"]).to(
            self.device)
        attention_mask = attention_mask.expand(bins + 1, -1)
        switch_values = torch.zeros(bins+1, self.num_switches).to(self.device)
        for bin_i in range(bins):
            switch_values[bin_i, switch_dim] = (
                min_value + (max_value - min_value) / (bins - 1) * bin_i
            )
        self.switch.set_value(switch_values)
        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids)
        length = input_ids.shape[1]
        loss_token = F.cross_entropy(
            output.logits[:, :-1].reshape((bins+1)*(length-1), -1),
            input_ids[:, 1:].reshape(-1),
            reduction="none"
        )
        loss_token = loss_token.reshape(bins + 1, length - 1)
        loss = loss_token.mean(-1)[:-1]
        dist = ((- loss + loss.mean()) * 100).softmax(0)
        dist_list = list(zip(
            [
                min_value + (max_value - min_value) / (bins - 1) * bin_i
                for bin_i in range(bins)
            ],
            dist.tolist(),
        ))
        best_guess = loss.argmin(0)
        best_guess_value = min_value + \
            (max_value - min_value) / (bins - 1) * best_guess.item()

        token_evidence = (- loss_token[best_guess] + loss_token[-1]) * 10
        token_evidence = [0] + token_evidence.tolist()
        # tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        word_evidence_list = []
        start = 0
        n_tokens = len(input_ids[0])
        for token_i in range(1, n_tokens+1):
            span = self.tokenizer.decode(input_ids[0][start: token_i])
            for _punc in punctuations:
                if token_i == n_tokens or _punc in span:
                    new_span = self.tokenizer.decode(
                        input_ids[0][start: token_i-1]).strip()
                    if len(new_span) <= 1:
                        break
                    word_evidence_list.append((
                        new_span,
                        np.array(token_evidence[start: token_i-1]).mean()
                    ))
                    start = token_i - 1
                    break

        # token_evidence_list = list(zip(tokens, token_evidence))
        return best_guess_value, dist_list, word_evidence_list
