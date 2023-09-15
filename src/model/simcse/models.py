# Code adapted from SimCSE (https://github.com/princeton-nlp/SimCSE) governed by MIT license.

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from typing import Optional, Tuple
import transformers
from transformers import RobertaTokenizer, PreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings, ModelOutput,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x


class ProjectorLayer(nn.Module):
    """
    Advanced dense layers for getting sentence representations over pooled representation.
    """
    def __init__(self, config, extra_config):
        super().__init__()
        assert extra_config.projector, 'projector is enabled but config.projector is not set'
        sizes = [config.hidden_size] + list(map(int, extra_config.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

    def forward(self, features, **kwargs):
        x = self.projector(features)

        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_avg", "avg",
                                    "cls_before_pooler", "avg_top2", "avg_first_last"], \
            "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "cls_avg":
            cls = last_hidden[:, 0]
            avg = ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
            return torch.cat([cls, avg], dim=-1)
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls" or cls.model_args.pooler_type == "projector":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()


def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Get raw embeddings, outputs.last_hidden_state=[bs*num_sent, hidden_dim]
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls" or cls.model_args.pooler_type == "projector":
        pooler_output = cls.mlp(pooler_output)

    # Separate representation, z1/z2.shape=[bs, hidden]
    z1, z2 = pooler_output[:, 0], pooler_output[:, 1]

    # Hard negative
    if num_sent == 3:
        z3 = pooler_output[:, 2]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Gather hard negative
        if num_sent >= 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3
            z3 = torch.cat(z3_list, 0)

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs x n_gpu, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)
    # cos_sim as logit, along dim=-1, resulting a matrix shape=[bs, bs]
    #   z1.unsqueeze(1).shape=[B,1,H], z2.unsqueeze(0).shape=[1,B,H],
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    # Hard negative
    if num_sent >= 3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)  # shape=[bs, bs*2]

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)  # shape=[bs]
    loss_fct = nn.CrossEntropyLoss()

    # Calculate loss with hard negatives
    if num_sent == 3:
        # Note that weights are actually logits of weights
        z3_weight = cls.model_args.hard_negative_weight
        weights = torch.tensor(
            [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) +
             [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1)
             for i in range(z1_z3_cos.size(-1))]
        ).to(cls.device)
        cos_sim = cos_sim + weights

    loss = loss_fct(cos_sim, labels)

    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if (cls.pooler_type == "cls" or cls.model_args.pooler_type == "projector") and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )


class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )


class CLSimilarity(nn.Module):
    """
    Dot product or cosine similarity
    """
    def __init__(self, sim_type, temp):
        super().__init__()
        self.sim_type = sim_type
        self.temp = temp
        self.cosine = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        '''
        x.shape=[B,H], y.shape=[B,H]
        cos_sim as logit, along dim=-1, resulting a matrix shape=[bs, bs]
        '''
        if self.sim_type == 'cosine':
            # cast to x.shape=[B,1,H], y.shape=[1,B,H] or [B,H]
            return self.cosine(x.unsqueeze(1), y.unsqueeze(0)) / self.temp
        elif self.sim_type == 'dot':
            return torch.mm(x, y.T) / self.temp
        else:
            raise NotImplementedError(f'Unsupported similarity function [{self.sim_type}]. '
                                      f'Only cosine and dot are supported.')


def gather_norm(input, input_mask=None):
    if input_mask is not None:
        _norm = torch.linalg.norm((input * input_mask.unsqueeze(-1)), dim=1)
        _norm = torch.masked_select(_norm, input_mask.bool().reshape(-1))
    else:
        _norm = torch.linalg.norm(input, dim=1)
    return _norm.mean()


class PretrainedModelForContrastiveLearning(PreTrainedModel):
    def __init__(self, hfconfig, **model_kargs):
        super().__init__(hfconfig)
        model_args = model_kargs["model_args"]
        self.config = hfconfig
        self.model_args = model_args
        self.seed = model_kargs["seed"]
        setattr(hfconfig, 'add_pooling_layer', False)
        self.doc_encoder = transformers.AutoModel.from_pretrained(
            self.model_args.model_name_or_path,
            config=hfconfig,
        )
        setattr(self.doc_encoder, 'encoder_type', 'doc_encoder')
        if model_args.shared_encoder:
            print('Sharing the parameters between doc/query encoder!')
            self.query_encoder = self.doc_encoder
            setattr(self.query_encoder, 'encoder_type', 'shared')
        else:
            self.query_encoder = transformers.AutoModel.from_pretrained(
                self.model_args.model_name_or_path,
                config=self.model_args)
            setattr(self.query_encoder, 'encoder_type', 'query_encoder')
        '''
        each loss group receives a tuple of (D+,Q+,D-,Q-)
        '''
        self.cl_loss_groups = ['psg2self', 'title2psg']
        self.cl_loss_weights = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if model_args.cl_loss_weights:
            cl_loss_weights = eval(model_args.cl_loss_weights)
            assert len(cl_loss_weights) == len(self.cl_loss_weights), \
                f'Ensure the cl_loss_weights is set correctly, currently num_loss={len(self.cl_loss_weights)}'
            self.cl_loss_weights = cl_loss_weights
        print('CL.loss_names=', self.cl_loss_groups)
        print('CL.loss_weights=', self.cl_loss_weights)

        print('pooler_type=', self.model_args.pooler_type)
        self.mlp_only_train = self.model_args.mlp_only_train
        print('mlp_only_train=', self.mlp_only_train)
        self.pooler_type = self.model_args.pooler_type
        self.pooler = Pooler(self.model_args.pooler_type)

        print('q_proj_type=', self.model_args.q_proj_type)
        print('d_proj_type=', self.model_args.d_proj_type)
        if self.model_args.q_proj_type and self.model_args.q_proj_type != "none":
            if self.model_args.q_proj_type == "mlp":
                self.q_mlp = MLPLayer(hfconfig)
            elif self.model_args.q_proj_type == "projector":
                self.q_mlp = ProjectorLayer(hfconfig, model_args)
            else:
                raise NotImplementedError('Unknown q_proj_type ' + self.model_args.q_proj_type)
            self._init_weights(self.q_mlp)
        else:
            self.q_mlp = None
        if self.model_args.d_proj_type and self.model_args.d_proj_type != "none":
            if self.model_args.d_proj_type == "shared":
                self.d_mlp = self.q_mlp
            elif self.model_args.d_proj_type == "mlp":
                self.d_mlp = MLPLayer(hfconfig)
            elif self.model_args.d_proj_type == "projector":
                self.d_mlp = ProjectorLayer(hfconfig, model_args)
            else:
                raise NotImplementedError('Unknown d_proj_type ' + self.model_args.d_proj_type)
            self._init_weights(self.d_mlp)
        else:
            self.d_mlp = None
        print('sim_type=', self.model_args.sim_type)
        print('temp=', self.model_args.temp)
        self.sim_type = self.model_args.sim_type
        self.temp = self.model_args.temp
        self.sim = CLSimilarity(self.model_args.sim_type, temp=self.model_args.temp)

        print('memory_type', self.model_args.memory_type)
        print('memory_size', self.model_args.memory_size)
        self.memory_type = self.model_args.memory_type
        self.memory_size = self.model_args.memory_size

        self._init_weights(self.pooler)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        is_query=False,
        mlm_input_ids=None,
        mlm_labels=None,
        length=None
    ):
        if sent_emb:
            return self.sentemb_forward(
                is_query,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return self.cl_forward(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )

    def cl_forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        hard_negative_sentids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        # print(input_ids.shape)
        batch_size = input_ids.size(0)
        total_loss = 0.0
        specific_losses = {}
        loss_fct = nn.CrossEntropyLoss()

        for lg_id, loss_group in enumerate(self.cl_loss_groups):
            # at least one weight should be larger than 0 to enable the loss groups
            weight_sum = sum([self.cl_loss_weights[i] for i in range(lg_id*3, lg_id*3+3)])
            if weight_sum == 0: continue
            # ignore the doc/query sents that won't be used (loss_weight==0.0)
            doc_idx = [lg_id*4, lg_id*4+1] if self.cl_loss_weights[lg_id*3+1] > 0 else [lg_id*4]
            query_idx = [lg_id*4+1, lg_id*4+3] if self.cl_loss_weights[lg_id*3+2] > 0 else [lg_id*4+1]
            num_doc, num_query = len(doc_idx), len(query_idx)
            doc_idx = torch.tensor(doc_idx).to(input_ids.device)
            query_idx = torch.tensor(query_idx).to(input_ids.device)

            # Flatten input for encoding
            doc_tokens = torch.index_select(input_ids, dim=1, index=doc_idx)
            doc_attention_mask = torch.index_select(attention_mask, dim=1, index=doc_idx)
            doc_tokens = doc_tokens.view((-1, doc_tokens.size(-1)))  # (bs * num_sent, len)
            doc_attention_mask = doc_attention_mask.view((-1, doc_attention_mask.size(-1)))  # (bs * num_sent len)
            # outputs.last_hidden_state=[bs, hidden_dim]
            doc_outputs = self.doc_encoder(
                doc_tokens, attention_mask=doc_attention_mask, return_dict=True,
                output_hidden_states=True if self.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            )

            query_tokens = torch.index_select(input_ids, dim=1, index=query_idx)
            query_attention_mask = torch.index_select(attention_mask, dim=1, index=query_idx)
            query_tokens = query_tokens.view((-1, query_tokens.size(-1)))  # (bs * num_sent, len)
            query_attention_mask = query_attention_mask.view((-1, query_attention_mask.size(-1)))  # (bs * num_sent len)
            # outputs.last_hidden_state=[bs, hidden_dim]
            query_outputs = self.query_encoder(
                query_tokens, attention_mask=query_attention_mask, return_dict=True,
                output_hidden_states=True if self.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            )

            doc_norm = gather_norm(
                doc_outputs.last_hidden_state.reshape(-1, doc_outputs.last_hidden_state.shape[-1]),
                doc_attention_mask.reshape(-1))
            query_norm = gather_norm(
                query_outputs.last_hidden_state.reshape(-1, query_outputs.last_hidden_state.shape[-1]),
                query_attention_mask.reshape(-1))
            specific_losses[f'norm_doc_{loss_group}_model'] = doc_norm
            specific_losses[f'norm_query_{loss_group}_model'] = query_norm

            # Pooling
            doc_pooler_output = self.pooler(doc_attention_mask, doc_outputs)
            query_pooler_output = self.pooler(query_attention_mask, query_outputs)
            doc_norm = gather_norm(doc_pooler_output)
            query_norm = gather_norm(query_pooler_output)
            specific_losses[f'norm_doc_{loss_group}_pool'] = doc_norm
            specific_losses[f'norm_query_{loss_group}_pool'] = query_norm

            # If using "cls", we add an extra MLP layer
            # (same as BERT's original implementation) over the representation.
            if self.d_mlp:
                doc_pooler_output = self.mlp(doc_pooler_output)
            if self.q_mlp:
                query_pooler_output = self.mlp(query_pooler_output)
            doc_norm = gather_norm(doc_pooler_output)
            query_norm = gather_norm(query_pooler_output)
            specific_losses[f'norm_doc_{loss_group}_mlp'] = doc_norm
            specific_losses[f'norm_query_{loss_group}_mlp'] = query_norm

            doc_pooler_output = doc_pooler_output.view((batch_size, num_doc, doc_pooler_output.size(-1)))  # (bs, num_sent, hidden)
            query_pooler_output = query_pooler_output.view((batch_size, num_query, query_pooler_output.size(-1)))  # (bs, num_sent, hidden)
            # Separate representation, z1.shape=[bs, hidden], z2.shape=[bs, num_q, hidden]
            z1, z2 = doc_pooler_output, query_pooler_output

            if self.memory_size > 0:
                # https://discuss.pytorch.org/t/random-number-on-gpu/9649
                torch.manual_seed(self.seed + dist.get_rank())
                z3 = torch.rand(self.memory_size, z1.shape[-1]).to(z1.device)

            # Gather all embeddings if using distributed training
            if dist.is_initialized() and self.training:
                # Gather hard negative
                if self.memory_size > 0:
                    z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
                    dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
                    z3_list[dist.get_rank()] = z3
                    z3 = torch.cat(z3_list, dim=0)

                # Dummy vectors for allgather
                z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
                z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
                # Allgather
                dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
                dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

                # Since allgather results do not have gradients, we replace the
                # current process's corresponding embeddings with original tensors
                z1_list[dist.get_rank()] = z1
                z2_list[dist.get_rank()] = z2
                # Get full batch embeddings: (bs x n_gpu, hidden)
                z1 = torch.cat(z1_list, 0)
                z2 = torch.cat(z2_list, 0)

            # CL with in-batch negatives
            specific_losses['batch_size'] = z1.shape[0]
            doc = z1[:, 0, :]  # [B x n_gpu, 1, H] -> [B x n_gpu,H]
            q = z2[:, 0, :]  # [B x n_gpu, 1, H] -> [B x n_gpu,H]
            pos_sim = self.sim(doc, q)  # sim.shape=[bs x n_gpu, bs x n_gpu]

            # Hard negative
            if self.memory_size > 0:
                z1_z3_cos = self.sim(doc, z3)
                z2_z3_cos = self.sim(q, z3)
                pos_sim = torch.cat([pos_sim, z1_z3_cos, z2_z3_cos], dim=1)  # shape=[bs, bs*2]

            labels = torch.arange(pos_sim.size(0)).long().to(self.device)  # shape=[bs]
            for loss_idx in range(3):
                loss_weight = self.cl_loss_weights[lg_id * 3 + loss_idx]
                if loss_weight <= 0: continue
                if loss_idx == 0:  # CL with in-batch negatives
                    loss_name = 'D+Q+'
                    loss = loss_weight * loss_fct(pos_sim, labels)
                elif loss_idx == 1:  # CL with hard negative docs (D- and Q+)
                    loss_name = 'D-Q+'
                    doc = z1[:, 1, :]  # [B,H]
                    q = z2[:, 0, :]  # [B,H]
                    neg_sim = self.sim(doc, q)  # sim.shape=[bs, bs]
                    pairwise_sim = torch.cat([pos_sim, neg_sim], 1)  # shape=[bs, bs*2]
                    loss = loss_weight * loss_fct(pairwise_sim, labels)
                elif loss_idx == 2:  # CL with hard negative queries (D+ and Q-)
                    loss_name = 'D+Q-'
                    doc = z1[:, 0, :]  # [B,H]
                    q = z2[:, 1, :]  # [B,H]
                    neg_sim = self.sim(doc, q)  # sim.shape=[bs, bs]
                    pairwise_sim = torch.cat([pos_sim, neg_sim], 1)  # shape=[bs, bs*2]
                    loss = loss_weight * loss_fct(pairwise_sim, labels)
                else:
                    raise NotImplementedError(f'loss_idx={loss_idx} should not exist.')

                total_loss += loss
                specific_losses[loss_group+'_'+loss_name] = loss

        return ContrastiveLearningOutput(
            loss=total_loss,
            specific_losses=specific_losses
        )

    def sentemb_forward(
        self,
        is_query,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if is_query:
            encoder = self.query_encoder
            mlp = self.mlp if hasattr(self, 'mlp') else self.q_mlp
        else:
            encoder = self.doc_encoder
            mlp = self.mlp if hasattr(self, 'mlp') else self.d_mlp

        outputs = encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if self.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

        pooler_output = self.pooler(attention_mask, outputs)
        if mlp and not self.mlp_only_train:
            pooler_output = mlp(pooler_output)

        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=pooler_output,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
        )


@dataclass
class ContrastiveLearningOutput(ModelOutput):
    """
    Base class for outputs of sentence contrative learning models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    specific_losses: Optional[dict] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
