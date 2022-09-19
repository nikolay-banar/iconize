import torch
import torch.nn as nn
import torch.nn.functional as F
from simpletransformers.language_representation import RepresentationModel
from simpletransformers.config.model_args import ModelArgs


class TextEncoder(nn.Module):
    def __init__(self, opt):
        super(TextEncoder, self).__init__()
        model_args = ModelArgs(max_seq_length=opt.max_words)
        self.opt = opt
        if opt.text_encoder != "bert":
            raise Exception('BERT must be used as a text encoder')

        representation_model = RepresentationModel("bert", "bert-base-multilingual-cased", args=model_args,
                                                   use_cuda=False)
        self.bert = representation_model.model
        bert_config = representation_model.config
        self.tokenizer = representation_model.tokenizer

        embedding_dim = bert_config.hidden_size

        self._freeze_layers(n=opt.frozen_bert_layers)

        Ks = [1, 2, 3]
        in_channel = 1
        out_channel = 512
        self.convs1 = nn.ModuleList([nn.Conv2d(in_channel, out_channel, (K, embedding_dim)) for K in Ks])
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        self.mapping = nn.Linear(len(Ks) * out_channel, opt.final_dims)

        self.max_seq_length = opt.max_words

        self.device = None

    def _freeze_layers(self, n=6):
        if n > 0:
            for child in self.bert.children():
                # print(child)
                for param in child.parameters():
                    param.requires_grad = False

            if self.opt.text_encoder == "bert":
                model_to_unfreeze = self.bert.bert
            else:
                model_to_unfreeze = self.bert

            if len(model_to_unfreeze.encoder.layer) > n:
                for child in model_to_unfreeze.encoder.layer[n:]:
                    for param in child.parameters():
                        param.requires_grad = True

    def tokenize(self, text_list):
        # Tokenize the text with the provided tokenizer
        return self.tokenizer.batch_encode_plus(
            text_list,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, lengths=None):

        if self.device is not None:
            input_ids = input_ids.cuda(self.device)

            if attention_mask is not None:
                attention_mask = attention_mask.cuda(self.device)

            if token_type_ids is not None:
                token_type_ids = token_type_ids.cuda(self.device)

        x = self.bert(input_ids, token_type_ids=token_type_ids,
                      attention_mask=attention_mask)

        x = x.unsqueeze(1)  # (batch_size, 1, token_num, embedding_dim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(batch_size, out_channel, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        output = torch.cat(x, 1)
        output = self.dropout(output)
        output = self.mapping(output)
        output = F.normalize(output, p=2, dim=1)
        return output
