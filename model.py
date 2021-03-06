import torch.nn as nn
from transformers import GPT2Model
from transformers.modeling_gpt2 import *
from cond_attn import Cond_Block

from data.util import *

class Decoder(GPT2Model):
    def __init__(self, config, add_input=False, add_attn=False, attn_proj_vary=False):
        super(GPT2Model, self).__init__(config)

        # added code here
        self.add_input = add_input
        self.add_attn = add_attn
        self.attn_proj_vary = attn_proj_vary

        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.output_past = True
        self.uide = nn.Embedding(len(uid_list), config.n_embd)
        self.pide = nn.Embedding(len(pid_list), config.n_embd)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

        if self.add_input:
            nz = 2 * config.n_embd
            nx = config.n_embd
            self.input_proj = nn.Linear(nz, nx, bias=False)

        if self.add_attn:
            nz = 2 * config.n_embd
            nx = config.n_embd
            n = config.n_layer

            if self.attn_proj_vary:
                self.attn_proj = nn.Linear(nz, nx * n, bias=False)
            else:
                self.attn_proj = nn.Linear(nz, nx, bias=False)

            self.h = nn.ModuleList([Cond_Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        else:
            self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.init_weights()

    def forward(
            self,
            uids=None,
            pids=None,
            input_ids=None,
            past=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, input_shape[-1])
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.n_layer

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        
        # uids = uids.view(input_shape[-1])
        u_embeds = self.uide(uids)
        # pids = pids.view(input_shape[-1])
        p_embeds = self.pide(pids)
        representations = torch.cat((u_embeds, p_embeds), dim=1)
        # add code here
        if self.add_input:
            assert (representations is not None)
            input_proj = self.input_proj(representations).unsqueeze(1)
            hidden_states = hidden_states + input_proj

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        # add code here
        if self.add_attn:
            assert (representations is not None)
            attn_proj = self.attn_proj(representations).unsqueeze(1)
            if self.attn_proj_vary:
                attn_proj = attn_proj.split(hidden_states.size(-1), dim=-1)
                assert len(attn_proj) == len(self.h)

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            if self.add_attn:
                if self.attn_proj_vary:
                    z = attn_proj[i]
                else:
                    z = attn_proj
                outputs = block(
                    hidden_states, z, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask[i]
                )
            else:
                outputs = block(
                    hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask[i]
                )
            # import pdb; pdb().set_trace()
            hidden_states, present = outputs[:2]
            if self.output_past:
                presents = presents + (present,)

            if self.output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_past:
            outputs = outputs + (presents,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)

        lm_logits = self.lm_head(hidden_states)
        outputs = (lm_logits,) + outputs

        return outputs  # last hidden state, (presents), (all hidden_states), (attentions)
