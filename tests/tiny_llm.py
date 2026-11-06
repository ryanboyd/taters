"""
A two-layer, 32-wide causal language model with a sixty-word vocabulary and
a chat template, written into a temporary folder, so the "ask a language
model" tests run offline in under a second. Its answers are noise: the
tests are about the plumbing (the template, padding, batching, the memory
plan, the resume), never about what a real model would say.
"""
from __future__ import annotations

from pathlib import Path

from tiny_encoder import WORDS

SPECIALS = ["<pad>", "<unk>", "<s>", "</s>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
VOCAB = SPECIALS + [w for w in WORDS if not w.startswith("[") and not w.startswith("##")]

#: ChatML, with the thinking switch Qwen3's template has: with thinking off
#: an empty think block opens the reply, the way that model does it
CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and not enable_thinking %}<think>\n\n</think>\n\n"
    "{% endif %}{% endif %}")


def build(folder: Path, *, hidden: int = 32, layers: int = 2, seed: int = 0,
          max_positions: int = 128) -> Path:
    """Write the tiny checkpoint (model + tokenizer + template) into ``folder``."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    vocab = {w: i for i, w in enumerate(VOCAB)}
    raw = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, bos_token="<s>", eos_token="<|im_end|>", pad_token="<pad>",
        unk_token="<unk>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>", "<think>", "</think>"])
    tokenizer.chat_template = CHAT_TEMPLATE
    config = LlamaConfig(vocab_size=len(VOCAB), hidden_size=hidden, num_hidden_layers=layers,
                         num_attention_heads=2, num_key_value_heads=2,
                         intermediate_size=hidden * 2, max_position_embeddings=max_positions,
                         bos_token_id=vocab["<s>"], eos_token_id=vocab["<|im_end|>"],
                         pad_token_id=vocab["<pad>"], tie_word_embeddings=False)
    torch.manual_seed(seed)
    model = LlamaForCausalLM(config)
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    return folder
