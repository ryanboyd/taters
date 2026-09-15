"""
A two-layer, 32-wide BERT with a sixty-word vocabulary, written into a
temporary folder, so every transformer test runs offline in well under a
second. Random weights are fine: the tests are about the plumbing (layers,
pooling, windows, batching, manifests), not about the meanings a real
encoder learns. Tests that need a real model are marked slow and skipped by
default.
"""
from __future__ import annotations

from pathlib import Path

WORDS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
         "the", "a", "and", "of", "to", "in", "it", "was", "is", "that",
         "cat", "dog", "fish", "bird", "sat", "ran", "swam", "flew", "on",
         "mat", "rug", "sea", "sky", "good", "bad", "happy", "sad", "very",
         "quite", "today", "yesterday", "meeting", "deadline", "email", "boss",
         "office", "budget", "potato", "gravy", "butter", "salt", "dinner",
         "supper", "kitchen", "recipe", ".", ",", "!", "?", "##s", "##ing",
         "##ed", "study", "found", "people"]


def build_classifier(folder: Path, *, labels=("food", "work"), regression: bool = False,
                     multi_label: bool = False,
                     hidden: int = 32, layers: int = 2, seed: int = 0,
                     max_positions: int = 64) -> Path:
    """
    Write a tiny *sequence-classification* checkpoint into ``folder`` -- the
    shape of a classifier somebody published on the hub, with ``id2label``
    in its config (or a one-output regression head when ``regression``).
    Untrained, so its predictions mean nothing; what the tests check is that
    Taters reads the head, the labels and the windows correctly.
    """
    import torch
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    vocab = folder / "vocab.txt"
    vocab.write_text("\n".join(WORDS) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(str(vocab), do_lower_case=True)
    if regression:
        config = BertConfig(vocab_size=len(WORDS), hidden_size=hidden,
                            num_hidden_layers=layers, num_attention_heads=2,
                            intermediate_size=hidden * 2,
                            max_position_embeddings=max_positions,
                            num_labels=1, problem_type="regression")
    else:
        config = BertConfig(vocab_size=len(WORDS), hidden_size=hidden,
                            num_hidden_layers=layers, num_attention_heads=2,
                            intermediate_size=hidden * 2,
                            max_position_embeddings=max_positions,
                            num_labels=len(labels),
                            id2label={i: lab for i, lab in enumerate(labels)},
                            label2id={lab: i for i, lab in enumerate(labels)},
                            **({"problem_type": "multi_label_classification"}
                               if multi_label else {}))
    torch.manual_seed(seed)
    model = BertForSequenceClassification(config)
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    return folder


def build(folder: Path, *, hidden: int = 32, layers: int = 2, seed: int = 0,
          max_positions: int = 64) -> Path:
    """Write the tiny checkpoint (model + tokenizer) into ``folder``."""
    import torch
    from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    vocab = folder / "vocab.txt"
    vocab.write_text("\n".join(WORDS) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(str(vocab), do_lower_case=True)
    config = BertConfig(vocab_size=len(WORDS), hidden_size=hidden,
                        num_hidden_layers=layers, num_attention_heads=2,
                        intermediate_size=hidden * 2,
                        max_position_embeddings=max_positions)
    torch.manual_seed(seed)
    model = BertForMaskedLM(config)
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    return folder
