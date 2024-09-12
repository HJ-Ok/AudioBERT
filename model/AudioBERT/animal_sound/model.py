import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, ASTModel, ASTConfig, AutoModelForMaskedLM
from peft import LoraModel, LoraConfig, get_peft_model


def extract_mask_token_embeddings(outputs, mask_token_index):
    predicted_token_ids = []

    for i in range(mask_token_index.size(0)):
        mask_indices = mask_token_index[i].nonzero(as_tuple=True)[0]

        if len(mask_indices) > 0:
            predicted_ids = outputs.logits[i, mask_indices].argmax(dim=-1)
            predicted_token_ids.append(predicted_ids)
        else:
            predicted_token_ids.append(torch.tensor([], device=outputs.logits.device))

    return predicted_token_ids


class AK_BERT(nn.Module):
    def __init__(self, language_model_path, audio_model_path, tokenizer, lora_r=64, lora_alpha=128, lora_dropout=0.01):
        super(AK_BERT, self).__init__()
        self.tokenizer = tokenizer
        self.language_config = AutoConfig.from_pretrained(language_model_path)
        self.language_enc = AutoModelForMaskedLM.from_pretrained(language_model_path)
        self.language_enc.resize_token_embeddings(len(tokenizer))
        # self.language_enc.gradient_checkpointing_enable()

        lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, target_modules=["query", "key", "value"], lora_dropout=lora_dropout
        )
        self.language_enc = get_peft_model(self.language_enc, lora_config, "default")

        self.audio_config = ASTConfig.from_pretrained(audio_model_path)
        self.audio_enc = ASTModel.from_pretrained(audio_model_path)

        print(self.language_enc.print_trainable_parameters())

        # for p in self.language_enc.parameters():
        #     p.requires_grad = False

        # for p in self.audio_enc.parameters():
        #     p.requires_grad = False

    def forward(self, audio_features, span_token_pos, input_ids, targets):
        mask_token_index = input_ids == self.tokenizer.mask_token_id

        lm_embs = self.language_enc.bert.embeddings(input_ids=input_ids)

        audio_embs = self.audio_enc(input_values=audio_features).last_hidden_state
        mean_audio_embs = torch.mean(audio_embs, dim=1)
        expanded_mean_audio_embs = mean_audio_embs.unsqueeze(1).expand(-1, input_ids.size(1), -1)

        lm_embs[torch.arange(span_token_pos.size(0)), span_token_pos] = expanded_mean_audio_embs_1[
            torch.arange(span_token_pos.size(0)), span_token_pos
        ]

        outputs = self.language_enc(inputs_embeds=lm_embs, labels=targets)

        predicted_token_id = extract_mask_token_embeddings(outputs, mask_token_index)
        return outputs, predicted_token_id
