import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPTJForCausalLM, AutoTokenizer

from .model_utils import Hack_no_grad, find_max_subspans
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


class Switching_GPTJModel(nn.Module):
    def __init__(self, model_name, adapted_component,
                 num_switches, rank, epsilon, init_var,
                 use_embedding, low_resource_mode):
        super().__init__()
        self.adapted_component = adapted_component
        # self.generator = pipeline('text-generation', model=model_name)
        # self.tokenizer = self.generator.tokenizer
        # self.model = self.generator.model
        if low_resource_mode:
            print("using low_resource_mode and fp16")
            self.model = GPTJForCausalLM.from_pretrained(
                "EleutherAI/gpt-j-6B", revision="float16",
                torch_dtype=torch.float16, low_cpu_mem_usage=True
            )
        else:
            self.model = GPTJForCausalLM.from_pretrained(
                "EleutherAI/gpt-j-6B",
            )
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.init_var = init_var
        self.num_switches = num_switches
        self.device = torch.device("cpu")
        self.low_resource_mode = low_resource_mode
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
        # self.generator.device = device
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
        if self.low_resource_mode:
            fp16 = torch.float16
            switch_values = switch_values.to(fp16)
            self.switch.projector1.data = self.switch.projector1.to(fp16)
            self.switch.projector2.data = self.switch.projector2.to(fp16)
        self.switch.set_value(switch_values[None])
        with torch.no_grad():
            input_ids = self.tokenizer(
                prompt, return_tensors="pt").input_ids.to(self.device)
            gen_tokens = self.model.generate(
                input_ids,
                do_sample=True, min_length=min_length, max_length=max_length,
                pad_token_id=self.tokenizer.pad_token_id)
            text = self.tokenizer.batch_decode(gen_tokens)[0]

        # recovering
        if self.low_resource_mode:
            fp32 = torch.float32
            self.switch.projector1.data = self.switch.projector1.to(fp32)
            self.switch.projector2.data = self.switch.projector2.to(fp32)
        return text

    def evidence_words(self, prompt, original_switch_values, max_segments=4,
                       max_length=10):
        if isinstance(original_switch_values, list):
            original_switch_values = torch.Tensor(original_switch_values)
        if original_switch_values.abs().sum() <= 0.2:
            return [(prompt, None)]
        tokenized = self.tokenizer(prompt)
        input_ids = torch.LongTensor(tokenized["input_ids"]).to(self.device)
        input_ids = input_ids.expand(2, -1)
        attention_mask = torch.LongTensor(tokenized["attention_mask"]).to(
            self.device)
        attention_mask = attention_mask.expand(2, -1)
        switch_values = torch.zeros(2, self.num_switches).to(self.device)
        switch_values[0] = original_switch_values
        switch_values[1] = (-original_switch_values > 0) * 2 - 1
        if self.low_resource_mode:
            fp16 = torch.float16
            switch_values = switch_values.to(fp16)
            self.switch.projector1.data = self.switch.projector1.to(fp16)
            self.switch.projector2.data = self.switch.projector2.to(fp16)
        self.switch.set_value(switch_values)
        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids)
        length = input_ids.shape[1]
        loss_token = F.cross_entropy(
            output.logits[:, :-1].reshape((2)*(length-1), -1),
            input_ids[:, 1:].reshape(-1),
            reduction="none"
        )
        loss_token = loss_token.reshape(2, length - 1)

        token_evidence = (- loss_token[0] + loss_token[1])
        tokens = input_ids[0]
        evidence_segments = find_max_subspans(
            token_evidence.cpu().numpy().tolist(), max_segments, max_length)[0]
        evidence_segments = [
            (_seg[0]+1, _seg[1]+1) for _seg in evidence_segments]
        start = 0
        output = []
        color = (
            "gray" if original_switch_values.shape[0] > 1
            else "red" if original_switch_values[0] > 0
            else "blue"
        )
        if len(evidence_segments) > 0:
            for _segment in evidence_segments:
                if _segment[0] > start:
                    output.append((
                        self.tokenizer.decode(tokens[start: _segment[0]]),
                        None
                    ))
                output.append((
                    self.tokenizer.decode(tokens[_segment[0]: _segment[1]]),
                    color
                ))
                start = _segment[1]
            length = tokens.shape[-1]
            if _segment[1] < length:
                output.append((
                    self.tokenizer.decode(tokens[_segment[1]: length]),
                    None
                ))
        else:
            output = [(prompt, None)]

        if self.low_resource_mode:
            fp32 = torch.float32
            self.switch.projector1.data = self.switch.projector1.to(fp32)
            self.switch.projector2.data = self.switch.projector2.to(fp32)
        return output

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
        if self.low_resource_mode:
            fp16 = torch.float16
            switch_values = switch_values.to(fp16)
            self.switch.projector1.data = self.switch.projector1.to(fp16)
            self.switch.projector2.data = self.switch.projector2.to(fp16)
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

        token_evidence = self.evidence_words(
            prompt, switch_values[best_guess],
            max_segments=min(len(prompt) // 150, 5),
            max_length=10
        )

        if self.low_resource_mode:
            fp32 = torch.float32
            self.switch.projector1.data = self.switch.projector1.to(fp32)
        return best_guess_value, dist_list, token_evidence
