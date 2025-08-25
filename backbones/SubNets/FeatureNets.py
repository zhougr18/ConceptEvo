import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel
import copy
import math

__all__ = [ 'BERTEncoder']

class BERTEncoder(BertPreTrainedModel):

    def __init__(self, config):

        super(BERTEncoder, self).__init__(config)
        self.bert = BertModel(config)
        
    def forward(self, text_feats):
        
        input_ids, input_mask, segment_ids = text_feats[:, 0], text_feats[:, 1], text_feats[:, 2]
        outputs = self.bert(input_ids = input_ids, attention_mask = input_mask, token_type_ids = segment_ids)
        # last_hidden_states = outputs.last_hidden_state
        
        return outputs


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

# sdif
class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class BertIntermediate(nn.Module):
    def __init__(self, hidden_size=768):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size * 4)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = gelu(hidden_states)
        return hidden_states

class BertCoAttention(nn.Module):
    def __init__(self, num_attention_heads=8, hidden_size=768, dp_rate=0.1):
        super(BertCoAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(dp_rate)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        # s2_attention_mask  b*1*1*49
        mixed_query_layer = self.query(s1_hidden_states)  # b*75*768
        mixed_key_layer = self.key(s2_hidden_states)  # b*49*768
        mixed_value_layer = self.value(s2_hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # b*12*75*64
        key_layer = self.transpose_for_scores(mixed_key_layer)  # b*12*49*64
        value_layer = self.transpose_for_scores(mixed_value_layer)  # b*12*49*64

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # b*12*75*49
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # b*12*75*49
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + s2_attention_mask
        # atention_scores b*12*75*49
        # Normalize the attention scores to probabilities.
        # b*12*75*49
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        # context_layer b*12*75*64
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        # context_layer b*75*768
        return context_layer

class BertOutput(nn.Module):
    def __init__(self, hidden_size=768, dp_rate=0.1):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(hidden_size * 4, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dp_rate)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertSelfOutput(nn.Module):
    def __init__(self,hidden_size=768, dp_rate=0.1):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size,hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dp_rate)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
class BertCrossAttention(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, dp_rate):
        super(BertCrossAttention, self).__init__()
        self.bertCoAttn = BertCoAttention(num_attention_heads, hidden_size, dp_rate)
        self.output = BertSelfOutput(hidden_size, dp_rate)

    def forward(self, s1_input_tensor, s2_input_tensor, s2_attention_mask):
        s1_cross_output = self.bertCoAttn(s1_input_tensor, s2_input_tensor, s2_attention_mask)
        attention_output = self.output(s1_cross_output, s1_input_tensor)
        return attention_output

class BertCrossAttentionLayer(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, dp_rate):
        super(BertCrossAttentionLayer, self).__init__()
        self.bertCorssAttn = BertCrossAttention(num_attention_heads, hidden_size, dp_rate)
        self.intermediate = BertIntermediate(hidden_size)
        self.output = BertOutput(hidden_size, dp_rate)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        attention_output = self.bertCorssAttn(s1_hidden_states, s2_hidden_states, s2_attention_mask)
        # b*75*768
        intermediate_output = self.intermediate(attention_output)
        # b*75*3072
        layer_output = self.output(intermediate_output, attention_output)
        # b*75*3072
        return layer_output

class BertCrossEncoder(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, dp_rate, n_layers):
        super(BertCrossEncoder, self).__init__()
        layer = BertCrossAttentionLayer(num_attention_heads, hidden_size, dp_rate)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        for layer_module in self.layer:
            s1_hidden_states = layer_module(s1_hidden_states, s2_hidden_states, s2_attention_mask)
        return s1_hidden_states