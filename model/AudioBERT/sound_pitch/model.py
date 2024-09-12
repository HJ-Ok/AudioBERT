import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, ASTModel, ASTConfig, AutoModelForMaskedLM
from peft import LoraModel, LoraConfig, get_peft_model


class AK_BERT(nn.Module):
    def __init__(self, language_model_path, audio_model_path, tokenizer, lora_r=64, lora_alpha=128, lora_dropout=0.01):
        super(AK_BERT, self).__init__()
        self.tokenizer = tokenizer
        self.language_model_path = language_model_path
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
        self.audio_enc.gradient_checkpointing_enable()
        self.linear = nn.Linear(768, 1024)

        print(self.language_enc.print_trainable_parameters())

        # for p in self.audio_enc.parameters():
        #     p.requires_grad = False

    def forward(self, audio_features_1, audio_features_2, first_token_pos, input_ids, targets):
        mask_token_index = input_ids == self.tokenizer.mask_token_id

        lm_embs = self.language_enc.bert.embeddings(input_ids=input_ids)

        audio_embs_1 = self.audio_enc(input_values=audio_features_1).last_hidden_state
        audio_embs_2 = self.audio_enc(input_values=audio_features_2).last_hidden_state

        mean_audio_embs_1 = torch.mean(audio_embs_1, dim=1)
        mean_audio_embs_2 = torch.mean(audio_embs_2, dim=1)

        expanded_mean_audio_embs_1 = mean_audio_embs_1.unsqueeze(1).expand(-1, input_ids.size(1), -1)
        expanded_mean_audio_embs_2 = mean_audio_embs_2.unsqueeze(1).expand(-1, input_ids.size(1), -1)

        if "large" in self.language_model_path:
            expanded_mean_audio_embs_1 = self.linear(expanded_mean_audio_embs_1)
            expanded_mean_audio_embs_2 = self.linear(expanded_mean_audio_embs_2)

        token_indices_1 = first_token_pos[:, 0]
        token_indices_2 = first_token_pos[:, 1]

        lm_embs[torch.arange(first_token_pos.size(0)), token_indices_1] = expanded_mean_audio_embs_1[
            torch.arange(first_token_pos.size(0)), token_indices_1
        ]
        lm_embs[torch.arange(first_token_pos.size(0)), token_indices_2] = expanded_mean_audio_embs_2[
            torch.arange(first_token_pos.size(0)), token_indices_2
        ]

        outputs = self.language_enc(inputs_embeds=lm_embs, labels=targets)

        tokens = ["lower", "higher"]
        lower_token_id, higher_token_id = self.tokenizer.convert_tokens_to_ids(tokens)
        logits = outputs.logits[mask_token_index]
        lower_higher_logits = logits[..., [lower_token_id, higher_token_id]]
        predicted_index = lower_higher_logits.argmax(axis=-1)
        predicted_token_id = torch.where(predicted_index == 0, lower_token_id, higher_token_id)

        return outputs, predicted_token_id
