import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

logger = logging.get_logger(__name__)

'''
@dataclass
class Llama3Config:
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    intermediate_size: int = 16384
    max_position_embeddings: int = 2048
    rope_theta: float = 50000.0
    norm_eps: float = 1e-5
    dropout: float = 0.1
'''
from .configuration_llama3 import Llama3Config


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight

'''
def precompute_freqs_cis(dim, max_position_embeddings, theta=10000):
    """
    Compute rotary embeddings as a complex tensor.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_position_embeddings, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    # Add dimensions for broadcasting
    return torch.stack((torch.cos(freqs), torch.sin(freqs)), dim=-1).unsqueeze(0).unsqueeze(2)
'''
def precompute_freqs_cis(dim, max_position_embeddings, theta=10000):
    """
    Precomputes rotary embeddings' frequency components.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_position_embeddings, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (max_position_embeddings, dim // 2)
    return torch.stack((freqs.cos(), freqs.sin()), dim=-1)  # (max_position_embeddings, dim // 2, 2)



# original code
'''
def apply_rotary_emb(x, freqs_cis):
    """
    Apply rotary positional embeddings to the input tensor.
    """
    freqs_cis_real, freqs_cis_imag = freqs_cis[..., 0], freqs_cis[..., 1]
    x_real, x_imag = x.unbind(dim=-1)

    # Element-wise multiplication with broadcasting
    x_rotated_real = x_real * freqs_cis_real - x_imag * freqs_cis_imag
    x_rotated_imag = x_real * freqs_cis_imag + x_imag * freqs_cis_real

    return torch.stack((x_rotated_real, x_rotated_imag), dim=-1)
'''

# run돌아갔지만, acc박살
'''
def apply_rotary_emb(x, freqs_cis):
    """
    Apply rotary embeddings to input tensor x using precomputed freqs_cis.
    """
    # x: (batch_size, seq_len, num_heads, head_dim)
    batch_size, seq_len, num_heads, head_dim = x.size()
    
    # Reshape input tensor for rotary application
    x = x.view(batch_size, seq_len, num_heads, head_dim // 2, 2)  # Split into real and imaginary parts
    x_real, x_imag = x.unbind(dim=-1)  # Separate real and imaginary parts

    # Adjust `freqs_cis` dimensions to match `x`
    freqs_cis = freqs_cis[:seq_len, :].unsqueeze(1).unsqueeze(0)  # (1, 1, seq_len, head_dim // 2)
    freqs_cis_real, freqs_cis_imag = freqs_cis.unbind(dim=-1)  # Split into real and imaginary

    # Apply rotary transformations
    x_rotated_real = x_real * freqs_cis_real - x_imag * freqs_cis_imag
    x_rotated_imag = x_real * freqs_cis_imag + x_imag * freqs_cis_real

    # Combine real and imaginary parts back
    x_rotated = torch.cat((x_rotated_real, x_rotated_imag), dim=-1)
    return x_rotated.view(batch_size, seq_len, num_heads, head_dim)
'''

def apply_rotary_emb(x, freqs_cis):
    """
    Apply rotary embeddings to input tensor x using precomputed freqs_cis.
    """
    # x: (batch_size, seq_len, num_heads, head_dim)
    batch_size, seq_len, num_heads, head_dim = x.size()

    # Reshape x for real and imaginary parts
    x = x.view(batch_size, seq_len, num_heads, head_dim // 2, 2)
    x_real, x_imag = x.unbind(dim=-1)  # Split into real and imaginary parts

    # Adjust `freqs_cis` dimensions to match `x`
    freqs_cis = freqs_cis[:seq_len, :].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim // 2)
    freqs_cis_real, freqs_cis_imag = freqs_cis.unbind(dim=-1)

    # Apply rotary transformations
    x_rotated_real = x_real * freqs_cis_real - x_imag * freqs_cis_imag
    x_rotated_imag = x_real * freqs_cis_imag + x_imag * freqs_cis_real

    # Combine real and imaginary parts back
    x_rotated = torch.cat((x_rotated_real, x_rotated_imag), dim=-1)
    return x_rotated.view(batch_size, seq_len, num_heads, head_dim)






'''
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, freqs_cis, mask=None):
        batch_size, seq_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)

        q, k = apply_rotary_emb(q, freqs_cis), apply_rotary_emb(k, freqs_cis)

        attn_weights = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(self.head_dim)
        if mask is not None:
            attn_weights += mask

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_output = torch.einsum("bhqk,bkhd->bqhd", attn_probs, v).reshape(batch_size, seq_len, -1)
        return self.out_proj(attn_output)
'''

'''
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads

        # Projection layers
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, freqs_cis, mask=None):
        batch_size, seq_len, hidden_size = hidden_states.size()

        # Project to query, key, value
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply rotary embeddings (adjust dimensions if needed)
        freqs_cis = freqs_cis[:, :seq_len, :, :].unsqueeze(0)
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # Compute attention
        attn_weights = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(self.head_dim)
        if mask is not None:
            attn_weights += mask

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_output = torch.einsum("bhqk,bkhd->bqhd", attn_probs, v)

        # Reshape and project back
        attn_output = attn_output.contiguous().view(batch_size, seq_len, hidden_size)
        return self.out_proj(attn_output)
'''

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, freqs_cis, mask=None):
        batch_size, seq_len, _ = hidden_states.size()

        # Project hidden states
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply rotary embeddings
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # Compute attention weights
        attn_weights = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(self.head_dim)
        if mask is not None:
            attn_weights += mask

        attn_probs = F.softmax(attn_weights, dim=-1)

        # Compute attention output
        attn_output = torch.einsum("bhqk,bkhd->bqhd", attn_probs, v)
        attn_output = attn_output.reshape(batch_size, seq_len, -1)

        return self.out_proj(attn_output)




class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.intermediate(hidden_states)
        hidden_states = F.silu(hidden_states)
        return self.output(hidden_states)

class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.ffn = FeedForward(config)
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(self, hidden_states, freqs_cis, mask=None):
        attn_output = self.self_attn(self.attn_norm(hidden_states), freqs_cis, mask)
        hidden_states = hidden_states + attn_output
        ffn_output = self.ffn(self.ffn_norm(hidden_states))
        return hidden_states + ffn_output


class Llama3Model(PreTrainedModel):
    config_class = Llama3Config

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads,
            config.max_position_embeddings,
            theta=config.rope_theta,
        )

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.tok_embeddings(input_ids)
        seq_len = input_ids.size(1)
        freqs_cis = self.freqs_cis[:seq_len].to(hidden_states.device)

        for layer in self.layers:
            hidden_states = layer(hidden_states, freqs_cis, attention_mask)

        hidden_states = self.norm(hidden_states)
        return hidden_states




''' original code
class Llama3ForCausalLM(Llama3Model):
    def __init__(self, config: Llama3Config):
        super().__init__(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        hidden_states = super().forward(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return (loss, logits) if loss is not None else logits
'''

# run 돌아갔지만, acc 박살
'''
class Llama3ForCausalLM(Llama3Model):
    def __init__(self, config: Llama3Config):
        super().__init__(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Get hidden states from the base model
        hidden_states = super().forward(input_ids, attention_mask)
        
        # Calculate logits
        logits = self.lm_head(hidden_states)
        
        # Initialize loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )

        # Return logits and optional loss as a dictionary
        return {"loss": loss, "logits": logits}
'''

class Llama3ForCausalLM(Llama3Model):
    def __init__(self, config: Llama3Config):
        super().__init__(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Get hidden states from the base model
        hidden_states = super().forward(input_ids, attention_mask)
        
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        # Calculate loss
        loss = None
        if labels is not None:
            # Ensure labels match the sequence length
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )
        
        return {"loss": loss, "logits": logits}




'''
class Llama3ForCausalLM(Llama3Model):
    def __init__(self, config: Llama3Config):
        super().__init__(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Get hidden states from the base model
        hidden_states = super().forward(input_ids, attention_mask)

        # Compute logits using the lm_head
        logits = self.lm_head(hidden_states)

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        # Return (loss, logits) if loss is calculated, otherwise return logits
        return (loss, logits) if loss is not None else logits
'''



class Llama3ForSequenceClassification(Llama3Model):
    def __init__(self, config: Llama3Config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        hidden_states = super().forward(input_ids, attention_mask)
        logits = self.classifier(hidden_states[:, 0, :])  # Use the first token ([CLS])
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return (loss, logits) if loss is not None else logits


class Llama3ForTokenClassification(Llama3Model):
    def __init__(self, config: Llama3Config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        hidden_states = super().forward(input_ids, attention_mask)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return (loss, logits) if loss is not None else logits


class Llama3ForQuestionAnswering(Llama3Model):
    def __init__(self, config: Llama3Config):
        super().__init__(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)  # Start and end logits

    def forward(self, input_ids=None, attention_mask=None, start_positions=None, end_positions=None):
        hidden_states = super().forward(input_ids, attention_mask)
        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        loss = None
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2

        return (loss, start_logits, end_logits) if loss is not None else (start_logits, end_logits)


