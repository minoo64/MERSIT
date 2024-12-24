from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class Llama3Config(PretrainedConfig):
    """
    Configuration class for Llama3. Stores model hyperparameters and architecture details.
    """

    model_type = "llama3"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        num_attention_heads: int = 32,
        num_hidden_layers: int = 32,
        intermediate_size: int = 16384,
        max_position_embeddings: int = 2048,
        rope_theta: float = 50000.0,
        norm_eps: float = 1e-5,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.norm_eps = norm_eps
        self.dropout = dropout

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Ensure compatibility with PretrainedConfig.from_pretrained().
        """
        return super().from_pretrained(*args, **kwargs)
