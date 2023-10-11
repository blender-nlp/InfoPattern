import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from itertools import chain

from .model_utils import Hack_no_grad
from .switches import Projected_Adaptor
from switching_lm.utils import set_seed


class Switching_DialoGPTModel(nn.Module):
    def __init__(self, model_name, adapted_component,
                 num_switches, rank, epsilon, init_var, embedding_dim):
        super().__init__()
        self.adapted_component = adapted_component
        # self.generator = pipeline('text-generation', model=model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.init_var = init_var
        self.num_switches = num_switches
        self.device = torch.device("cpu")
        embed_dim = self.model.lm_head.weight.shape[1]
        vocab_size = self.model.lm_head.weight.shape[0]
        self.with_layer = embedding_dim != 0
        self.history = None

        if self.with_layer:
            self.switch_layer = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.LeakyReLU(),
                nn.Linear(embedding_dim, num_switches)
            )
            for _param in self.switch_layer.parameters():
                nn.init.normal_(_param, 0, 1e-2)
        else:
            self.switch_layer = nn.Sequential()

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

    def forward(self, input_ids, attention_mask, embeddings):
        if self.with_layer:
            switch_values = self.switch_layer(embeddings)
            switch_values = torch.tanh(switch_values)
        else:
            switch_values = embeddings
        self.switch.set_value(switch_values)
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids)
        return output

    def parameters(self):
        return chain(self.switch.parameters(), self.switch_layer.parameters())

    def state_dict(self):
        return {
            "switch": self.switch.state_dict(),
            "switch_layer": self.switch_layer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.switch.load_state_dict(state_dict["switch"])
        self.switch_layer.load_state_dict(state_dict["switch_layer"])

    def to_device(self, device):
        self.model.to(device)
        self.switch_layer.to(device)
        self.device = device

    def regularization_term(self):
        return self.switch.regularization_term()

    def generate(self, prompt, embeddings, device,
                 min_length=20, max_length=1000, seed=None, resuming=True,
                 beam_search=False, beam_sample=False):
        if seed is not None:
            set_seed(seed)
        if self.with_layer:
            switch_values = self.switch_layer(embeddings)
            switch_values = torch.tanh(switch_values)
        else:
            switch_values = embeddings
        self.switch.set_value(switch_values[None])
        if isinstance(prompt, list):
            prompt = self.tokenizer.eos_token.join(prompt)
        if self.history is not None and resuming:
            prompt = self.history + prompt
        if not resuming:
            self.history = None
        with torch.no_grad():
            input_encoded = self.tokenizer.encode(
                prompt + self.tokenizer.eos_token, return_tensors="pt",
            ).to(device)
            truncate_length = max(min(1024, max_length) - 100, max_length // 2)
            if input_encoded.shape[1] >= truncate_length:
                input_encoded = input_encoded[:, -truncate_length:]
            if beam_search:
                generated = self.model.generate(
                    input_encoded,
                    do_sample=False, num_beams=5, min_length=min_length, max_length=max_length,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            elif beam_sample:
                generated = self.model.generate(
                    input_encoded,
                    do_sample=True, num_beams=5, min_length=min_length, max_length=max_length,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            else:
                generated = self.model.generate(
                    input_encoded,
                    do_sample=True, min_length=min_length, max_length=max_length,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            text = self.tokenizer.decode(
                generated[0, input_encoded.shape[-1]:],
                skip_special_tokens=True)
        self.history = prompt + self.tokenizer.eos_token + text + \
            self.tokenizer.eos_token
        return text

    def switch_analysis(self, prompt, switch_dim, min_value=-3, max_value=3,
                        bins=7):
        raise NotImplementedError()
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
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        token_evidence_list = list(zip(tokens, token_evidence))

        return best_guess_value, dist_list, token_evidence_list
