"""
What every transformer step shares: finding and loading an encoder, picking
layers, pooling, encoding sentences in batches, and surviving a full GPU.

Three steps sit on this -- embeddings from a pre-trained encoder, adapting
an encoder to a corpus, fine-tuning a predictor -- and the choices they
have in common are the ones a paper has to report the same way each time:
which layers, pooled how, at what length, in what precision. Made once
here, tested once, and the report of every step names them from the same
constants. torch and transformers are imported inside the functions that
need them, so importing this module costs nothing and the wizard can read
signatures without loading either.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

__all__ = ["CURATED_ENCODERS", "CURATED_SENTENCE_MODELS", "LAYER_CHOICES",
           "POOLINGS", "PRECISIONS", "ENCODER_MODEL_TYPES",
           "cached_hub_encoders", "cached_hub_sentence_models",
           "ResolvedEncoder", "torch_missing_reason", "resolve_encoder",
           "load_encoder", "parse_layers", "pool_hidden", "encode_sentences",
           "run_with_oom_fallback", "autocast_for", "freeze_below",
           "payload_digests", "set_threads", "ENCODER_KIND", "PREDICTOR_KIND",
           "window_length"]

PathLike = Union[str, Path]

#: The manifest tags of the two kinds of transformer model Taters saves.
ENCODER_KIND = "taters-encoder"
PREDICTOR_KIND = "taters-text-predictor-model"

#: Encoders offered by name, with what to say about each. distilroberta is
#: the default: a good trade of quality for speed, and the base of the
#: sentence-transformers models most people already use. MiniLM is the one
#: to reach for on a laptop without a GPU.
CURATED_ENCODERS: Tuple[Tuple[str, str], ...] = (
    ("distilroberta-base",
     "82M parameters, 6 layers. The default: good features per second."),
    ("sentence-transformers/all-MiniLM-L6-v2",
     "22M parameters, 6 layers. The smallest and fastest; fine on a CPU. "
     "Meaning-tuned (a sentence-transformers model), so its raw embeddings "
     "already carry meaning."),
    ("distilbert-base-uncased",
     "66M parameters, 6 layers. Lower-cases everything."),
    ("roberta-base",
     "125M parameters, 12 layers. Stronger, about twice as slow."),
    ("bert-base-uncased",
     "110M parameters, 12 layers. The classic; lower-cases everything."),
    ("microsoft/deberta-v3-base",
     "184M parameters, 12 layers. The strongest base-size encoder, and the "
     "slowest here."),
)

#: Meaning-tuned models offered by name -- sentence-transformers checkpoints,
#: trained so that texts meaning the same get similar vectors. A different
#: list from the raw encoders above on purpose: the two pickers used to look
#: identical, and a person choosing a model for the meaning-tuned step should
#: not be shown distilroberta as the first row. all-roberta-large-v1 leads
#: because it is the step's long-standing default; mpnet is the better trade
#: for most people; the MiniLMs are for a laptop.
CURATED_SENTENCE_MODELS: Tuple[Tuple[str, str], ...] = (
    ("sentence-transformers/all-roberta-large-v1",
     "355M parameters, 1024-wide vectors. The default: the strongest here, "
     "and the slowest."),
    ("sentence-transformers/all-mpnet-base-v2",
     "110M parameters, 768-wide vectors. The usual recommendation: nearly "
     "as strong, about three times faster."),
    ("sentence-transformers/all-MiniLM-L12-v2",
     "33M parameters, 384-wide vectors. Fast; fine on a CPU."),
    ("sentence-transformers/all-MiniLM-L6-v2",
     "22M parameters, 384-wide vectors. The smallest and fastest."),
    ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
     "278M parameters, 768-wide vectors. For text in fifty-odd languages, "
     "or several at once."),
)

#: The `model_type` values of encoders -- models this module can read
#: hidden states from and adapt. A Whisper, a Qwen or a wav2vec2 checkpoint
#: in the same cache is not one, and offering it would fail at load time.
ENCODER_MODEL_TYPES = frozenset({
    "bert", "roberta", "distilbert", "deberta", "deberta-v2", "electra",
    "albert", "xlm-roberta", "mpnet", "camembert", "xlm", "longformer",
    "big_bird", "funnel", "convbert", "squeezebert", "mobilebert", "ernie",
    "nomic_bert", "modernbert"})


def cached_hub_configs(cache_dir: Optional[PathLike] = None) -> List[Tuple[str, dict]]:
    """
    Every checkpoint in the Hugging Face cache with its config: ``(hub id,
    config dict)``. Offline and cheap; the pickers filter it their own way
    (encoders by model type, classifiers by head).
    """
    if cache_dir is None:
        from ..helpers.settings import model_cache_dir

        cache_dir = model_cache_dir()
    root = Path(cache_dir)
    if not root.is_dir():
        return []
    out: List[Tuple[str, dict]] = []
    for folder in sorted(root.glob("models--*")):
        hub_id = folder.name[len("models--"):].replace("--", "/")
        configs = sorted(folder.glob("snapshots/*/config.json"))
        if not configs:
            continue
        try:
            cfg = json.loads(configs[-1].read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(cfg, dict):
            out.append((hub_id, cfg))
    return out


def cached_hub_encoders(cache_dir: Optional[PathLike] = None) -> List[Tuple[str, str]]:
    """
    The encoders already downloaded to this machine: ``(hub id, note)``.

    Reads the Hugging Face cache (``HF_HUB_CACHE``, or ``~/.cache/huggingface/
    hub``) rather than asking the hub, so it works offline and costs
    nothing. Only checkpoints whose config names an encoder architecture
    are listed (see :data:`ENCODER_MODEL_TYPES`); the note says the layers
    and width, which is what tells a base-size model from a large one.
    """
    out: List[Tuple[str, str]] = []
    for hub_id, cfg in cached_hub_configs(cache_dir):
        if str(cfg.get("model_type", "")) not in ENCODER_MODEL_TYPES:
            continue
        # a checkpoint with a classification head is somebody's finished
        # classifier, not a base to build on. it has a picker of its own
        # (Settings -> saved models -> import a Hugging Face classifier)
        if any(str(a).endswith("ForSequenceClassification")
               for a in (cfg.get("architectures") or [])):
            continue
        layers = cfg.get("num_hidden_layers") or cfg.get("n_layers")
        width = cfg.get("hidden_size") or cfg.get("dim")
        bits = [b for b in (f"{layers} layers" if layers else "",
                            f"{width} wide" if width else "") if b]
        out.append((hub_id, "downloaded" + (f" · {' · '.join(bits)}" if bits else "")))
    return out


#: The file a sentence-transformers checkpoint carries and a raw encoder does
#: not. Reading it, rather than trusting the ``sentence-transformers/`` prefix
#: of the hub id, is what lets a meaning-tuned model from any organization --
#: or a folder someone saved themselves -- be recognized for what it is.
SENTENCE_MODEL_MARKER = "config_sentence_transformers.json"


def cached_hub_sentence_models(cache_dir: Optional[PathLike] = None) -> List[Tuple[str, str]]:
    """
    The meaning-tuned models already downloaded to this machine: ``(hub id, note)``.

    The sibling of :func:`cached_hub_encoders`, reading the same cache the
    same offline way, but keeping only checkpoints that carry
    :data:`SENTENCE_MODEL_MARKER` -- the file sentence-transformers writes
    beside its weights. A raw encoder in the same cache is left out, because
    the step this feeds promises a model trained for meaning, and offering a
    row that quietly is not one would break that promise.
    """
    if cache_dir is None:
        from ..helpers.settings import model_cache_dir

        cache_dir = model_cache_dir()
    root = Path(cache_dir)
    out: List[Tuple[str, str]] = []
    for hub_id, cfg in cached_hub_configs(root):
        folder = root / f"models--{hub_id.replace('/', '--')}"
        if not any(folder.glob(f"snapshots/*/{SENTENCE_MODEL_MARKER}")):
            continue
        layers = cfg.get("num_hidden_layers") or cfg.get("n_layers")
        width = cfg.get("hidden_size") or cfg.get("dim")
        bits = [b for b in (f"{layers} layers" if layers else "",
                            f"{width}-wide vectors" if width else "") if b]
        out.append((hub_id, "downloaded" + (f" · {' · '.join(bits)}" if bits else "")))
    return out


#: How hidden states become one vector per token position. The last layer
#: is specialized to the pretraining objective (predicting masked words);
#: the penultimate layers carry more general meaning and transfer better as
#: frozen features, which is why ``second_to_last`` is the default and the
#: choice DLATK made for the same reason.
LAYER_CHOICES = ("second_to_last", "last", "last4_mean", "last4_concat")
#: How token vectors become one vector per sentence. Mean pooling over the
#: real tokens beats the ``[CLS]`` vector for an encoder that was never
#: fine-tuned to put a sentence's meaning there.
POOLINGS = ("mean", "cls", "max")
PRECISIONS = ("auto", "fp32", "fp16")


def torch_missing_reason() -> str:
    """Why the transformer steps cannot run, or ``''``."""
    import importlib.util

    missing = [m for m in ("torch", "transformers") if importlib.util.find_spec(m) is None]
    if not missing:
        return ""
    return (f"The transformer steps need {' and '.join(missing)}, which "
            f"{'is' if len(missing) == 1 else 'are'} not installed. torch is "
            f"installed separately so you get the build for your GPU: see "
            f"the install guide.")


# ---------------------------------------------------------------------------
# Where an encoder comes from
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedEncoder:
    """
    An encoder as something transformers can load, plus what to call it.

    ``source`` is the folder or Hugging Face name to load; ``kind`` says
    where it came from (``hub``, ``folder``, ``encoder`` for a Taters-adapted
    encoder, ``predictor`` for the encoder inside a fine-tuned predictor);
    ``manifest`` the Taters model file it was found through, if any;
    ``label`` what the report calls it; ``base_model`` the name of the
    original pre-trained model, walked back through Taters manifests.
    """

    source: str
    kind: str
    label: str
    base_model: str
    manifest: Optional[Path] = None


def _payload_folder(manifest: Path, doc: dict, kind_id: str) -> Path:
    """The weights folder a Taters manifest names, beside it or in the library."""
    names = [str(n) for n in (doc.get("payload") or [])]
    if not names:
        raise ValueError(f"{manifest.name} names no weights folder (its 'payload' "
                         f"list is empty).")
    beside = manifest.with_name(Path(names[0]).name)
    if beside.is_dir():
        return beside
    try:
        from ..helpers.library import KINDS, find_payload

        found = find_payload(KINDS[kind_id], names[0],
                             doc.get("payload_digests") or {})
    except Exception:
        found = None
    if found is not None:
        return found
    raise ValueError(
        f"{manifest.name} needs its weights folder {names[0]}, which is not "
        f"beside it and not in your library. Put the folder next to the model "
        f"file, or add the model to your library (Settings) so any run can "
        f"find it.")


def resolve_encoder(spec: PathLike) -> ResolvedEncoder:
    """
    Turn what the user named into something ``from_pretrained`` accepts.

    Accepts a Hugging Face model name, a folder holding a checkpoint
    (``config.json`` inside), a Taters encoder manifest (``.json`` of kind
    ``taters-encoder``: its checkpoint folder is used), or a Taters
    predictor manifest (the encoder inside its payload). Refusals name what
    was tried.
    """
    text = str(spec).strip()
    if not text:
        raise ValueError("no encoder was named: give a Hugging Face model name, "
                         "a checkpoint folder, or a Taters model file")
    path = Path(text)
    if path.suffix.lower() == ".json" and path.is_file():
        doc = json.loads(path.read_text(encoding="utf-8"))
        kind = doc.get("kind")
        if kind == ENCODER_KIND:
            folder = _payload_folder(path, doc, "encoders")
            return ResolvedEncoder(source=str(folder), kind="encoder",
                                   label=str(doc.get("name") or path.stem),
                                   base_model=str(doc.get("base_model") or path.stem),
                                   manifest=path)
        if kind == PREDICTOR_KIND:
            folder = _payload_folder(path, doc, "models") / "encoder"
            if not folder.is_dir():
                raise ValueError(f"{path.name}'s weights hold no 'encoder' folder; "
                                 f"re-train the predictor to write one.")
            return ResolvedEncoder(source=str(folder), kind="predictor",
                                   label=str(doc.get("name") or path.stem),
                                   base_model=str(doc.get("base_model") or path.stem),
                                   manifest=path)
        raise ValueError(f"{path.name} is a {kind!r} file, not a text encoder or a "
                         f"fine-tuned text predictor.")
    if path.is_dir():
        if not (path / "config.json").is_file():
            raise ValueError(f"{path} holds no config.json, so it is not a "
                             f"transformers checkpoint folder.")
        return ResolvedEncoder(source=str(path), kind="folder", label=path.name,
                               base_model=_base_of(path) or path.name)
    if path.suffix.lower() == ".json":
        raise FileNotFoundError(f"model file not found: {path}")
    return ResolvedEncoder(source=text, kind="hub", label=text, base_model=text)


def _base_of(folder: Path) -> str:
    """The pre-trained model a checkpoint folder descends from, when its
    config remembers (transformers writes ``_name_or_path``)."""
    try:
        cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        return str(cfg.get("_name_or_path") or "")
    except (OSError, ValueError):
        return ""


def load_encoder(spec: PathLike, *, device: str = "auto", for_mlm: bool = False,
                 verbose: bool = False):
    """
    Load an encoder and its tokenizer onto the resolved device.

    Returns ``(model, tokenizer, resolved, device_name, fallback_reason)``.
    ``for_mlm`` loads the masked-language-model head too (adaptation
    trains it); otherwise the bare encoder. The device is resolved through
    the same rule every GPU step uses, and a genuinely unusable card falls
    back to the CPU with the reason rather than failing at the first batch.
    """
    missing = torch_missing_reason()
    if missing:
        raise ImportError(missing)
    import torch
    from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

    from ..helpers.gpu import resolve_device

    resolved = resolve_encoder(spec)

    def probe():
        torch.zeros(1, device="cuda") + 1

    device_name, reason = resolve_device(device, backend="torch", probe=probe)
    if verbose:
        print(f"[transformers] loading {resolved.label} on {device_name}")
        if reason:
            print(f"[transformers] {reason}")
    # we pass the cache dir explicitly on top of exporting it to the environment,
    # in case a script imported transformers before taters got around to setting it
    from ..helpers.settings import model_cache_dir

    cache = {"cache_dir": str(model_cache_dir())} if resolved.kind == "hub" else {}
    tokenizer = AutoTokenizer.from_pretrained(resolved.source, **cache)
    loader = AutoModelForMaskedLM if for_mlm else AutoModel
    model = loader.from_pretrained(resolved.source, **cache)
    model.to(device_name)
    model.eval()
    return model, tokenizer, resolved, device_name, reason


# ---------------------------------------------------------------------------
# Layers, pooling, precision
# ---------------------------------------------------------------------------

def parse_layers(spec: str, n_layers: int) -> Tuple[List[int], str]:
    """
    Which hidden states to read, and how to combine several.

    ``hidden_states`` has ``n_layers + 1`` entries: index 0 is the embedding
    lookup, which is not a layer and is refused. Returns non-negative
    indices into that tuple and ``"mean"`` or ``"concat"``. An explicit
    list (``"-1,-2"`` or ``"5,6"``) is averaged.
    """
    text = str(spec).strip().lower()
    n = int(n_layers)
    if n < 1:
        raise ValueError("the encoder reports no layers")
    if text == "second_to_last":
        idx = [n - 1] if n >= 2 else [n]
        return idx, "mean"
    if text == "last":
        return [n], "mean"
    if text in ("last4_mean", "last4_concat"):
        k = min(4, n)
        return list(range(n - k + 1, n + 1)), ("concat" if text.endswith("concat") else "mean")
    try:
        raw = [int(p) for p in text.replace(";", ",").split(",") if p.strip()]
    except ValueError:
        raise ValueError(
            f"layers must be one of {', '.join(LAYER_CHOICES)} or a list of layer "
            f"numbers like '-1,-2' or '5,6', not {spec!r}") from None
    if not raw:
        raise ValueError("no layers were named")
    out = []
    for r in raw:
        i = n + 1 + r if r < 0 else r
        if r == 0 or i < 1 or i > n:
            raise ValueError(
                f"layer {r} does not exist: this encoder has layers 1..{n} "
                f"(or -1..-{n} counting from the top); layer 0 is the embedding "
                f"lookup, not a layer")
        out.append(i)
    return sorted(set(out)), "mean"


def pool_hidden(hidden, mask, pooling: str):
    """
    One vector per sequence from ``[batch, tokens, dim]`` hidden states.

    ``mean`` averages over the real tokens (the attention mask, so padding
    never leaks in); ``cls`` takes the first position; ``max`` the
    element-wise maximum over real tokens.
    """
    import torch

    if pooling == "cls":
        return hidden[:, 0, :]
    m = mask.unsqueeze(-1).to(hidden.dtype)
    if pooling == "mean":
        summed = (hidden * m).sum(dim=1)
        counts = m.sum(dim=1).clamp(min=1.0)
        return summed / counts
    if pooling == "max":
        very_low = torch.finfo(hidden.dtype).min
        return hidden.masked_fill(m == 0, very_low).max(dim=1).values
    raise ValueError(f"pooling must be one of {', '.join(POOLINGS)}, not {pooling!r}")


def autocast_for(device_name: str, precision: str):
    """A context that runs in half precision when asked and the device can;
    fp16 on a CPU is slower and less exact, so ``auto`` means fp16 only on
    CUDA."""
    import contextlib

    import torch

    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {', '.join(PRECISIONS)}, not {precision!r}")
    use_half = (precision == "fp16") or (precision == "auto" and device_name.startswith("cuda"))
    if use_half and device_name.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def run_with_oom_fallback(fn: Callable[[int], object], batch_size: int, *,
                          verbose: bool = False, what: str = "the batch"):
    """
    Call ``fn(batch_size)``; on a CUDA out-of-memory error, halve and retry.

    Below one the error is re-raised with advice, because at that point
    the model itself does not fit: a shorter ``max_length``, a smaller
    encoder, or the CPU are the remaining moves.
    """
    size = int(batch_size)
    while True:
        try:
            return fn(size)
        except Exception as e:      # cuda's OutOfMemoryError is just a RuntimeError
            if "out of memory" not in str(e).lower():
                raise
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
            if size <= 1:
                raise RuntimeError(
                    f"the GPU ran out of memory on {what} even one item at a time. "
                    f"Use a shorter max_length, a smaller encoder (all-MiniLM-L6-v2), "
                    f"or device='cpu'.") from e
            size = max(1, size // 2)
            if verbose:
                print(f"[transformers] GPU out of memory; retrying {what} at "
                      f"batch size {size}")


def freeze_below(model, train_layers: int) -> Tuple[int, int]:
    """
    Freeze the embeddings and all but the top ``train_layers`` layers.

    ``0`` trains everything (the usual fine-tuning). Returns ``(trainable,
    frozen)`` parameter counts for the report. Works on the encoder
    modules transformers names ``embeddings`` and ``encoder.layer`` (BERT,
    RoBERTa, DistilBERT via ``transformer.layer``, DeBERTa).
    """
    base = getattr(model, "base_model", model)
    layers = None
    for attr in ("encoder.layer", "transformer.layer", "layers"):
        obj = base
        ok = True
        for part in attr.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                ok = False
                break
        if ok:
            layers = obj
            break
    if int(train_layers) > 0 and layers is not None:
        keep = int(train_layers)
        for p in base.parameters():
            p.requires_grad = False
        for layer in list(layers)[-keep:]:
            for p in layer.parameters():
                p.requires_grad = True
        # any heads sitting outside the base model (a predictor's head, the MLM
        # head) get to train too. but only when there *is* an outside: a bare
        # encoder is its own base model, and its parameter names have no prefix
        # that we could use to tell them apart
        if base is not model:
            prefix = getattr(model, "base_model_prefix", "") or "\x00"
            for name, p in model.named_parameters():
                if not name.startswith(prefix):
                    p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


def set_threads(workers: int) -> int:
    """torch's CPU thread count from the usual ``workers`` setting (0 = auto:
    all but one core), returned for the report."""
    import torch

    n = int(workers) if int(workers) > 0 else max(1, (os.cpu_count() or 2) - 1)
    try:
        torch.set_num_threads(n)
    except Exception:
        pass
    return n


# ---------------------------------------------------------------------------
# Encoding sentences
# ---------------------------------------------------------------------------

def _windows(ids: List[int], max_length: int, stride: int) -> List[List[int]]:
    """Split a too-long token sequence into overlapping windows of at most
    ``max_length`` tokens (special tokens included), never cutting text off:
    a sentence longer than the model's window is averaged over its parts."""
    body = max(1, int(max_length) - 2)
    if len(ids) <= body:
        return [ids]
    step = max(1, body - int(stride))
    out = []
    start = 0
    while True:
        out.append(ids[start:start + body])
        if start + body >= len(ids):
            break
        start += step
    return out


def encode_sentences(model, tokenizer, sentences: Sequence[str], *,
                     layers: str = "second_to_last", pooling: str = "mean",
                     max_length: int = 512, batch_size: int = 32,
                     device_name: str = "cpu", precision: str = "auto",
                     stride: int = 32, verbose: bool = False,
                     on_batch: Optional[Callable[[int, int], None]] = None):
    """
    One vector per sentence: ``(vectors [n, d] float32, token_counts [n])``.

    Every sentence is tokenized without truncation; one longer than
    ``max_length`` is windowed with ``stride`` tokens of overlap and its
    windows' vectors averaged, so nothing is silently cut. Windows are
    sorted by length and batched, the chosen layers are read from
    ``hidden_states`` and pooled over the attention mask, and a GPU that
    runs out of memory halves the batch and carries on.
    """
    import numpy as np
    import torch

    n_layers = int(getattr(model.config, "num_hidden_layers", 0) or
                   getattr(model.config, "n_layers", 0))
    idx, combine = parse_layers(layers, n_layers)
    if pooling not in POOLINGS:
        raise ValueError(f"pooling must be one of {', '.join(POOLINGS)}, not {pooling!r}")
    sentences = [str(s) for s in sentences]
    if not sentences:
        dim = int(model.config.hidden_size) * (len(idx) if combine == "concat" else 1)
        return np.zeros((0, dim), dtype=np.float32), []

    # first, token ids per sentence, then we chop those into windows. we keep
    # track of which sentence each window came from so we can put it back later
    encoded = tokenizer(sentences, add_special_tokens=False, truncation=False)["input_ids"]
    token_counts = [len(ids) for ids in encoded]
    windows: List[Tuple[int, List[int]]] = []
    for s_i, ids in enumerate(encoded):
        for w in _windows(ids, max_length, stride):
            windows.append((s_i, w))
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def with_specials(ids: List[int]) -> List[int]:
        head = [cls_id] if cls_id is not None else []
        tail = [sep_id] if sep_id is not None else []
        return head + list(ids) + tail

    order = sorted(range(len(windows)), key=lambda i: len(windows[i][1]))
    dim = int(model.config.hidden_size) * (len(idx) if combine == "concat" else 1)
    window_vectors = np.zeros((len(windows), dim), dtype=np.float32)

    def run(size: int):
        done = 0
        for start in range(0, len(order), size):
            chunk = order[start:start + size]
            seqs = [with_specials(windows[i][1]) for i in chunk]
            width = max(len(s) for s in seqs)
            input_ids = torch.full((len(seqs), width), int(pad_id), dtype=torch.long)
            mask = torch.zeros((len(seqs), width), dtype=torch.long)
            for r, s in enumerate(seqs):
                input_ids[r, :len(s)] = torch.tensor(s, dtype=torch.long)
                mask[r, :len(s)] = 1
            input_ids = input_ids.to(device_name)
            mask = mask.to(device_name)
            with torch.inference_mode(), autocast_for(device_name, precision):
                out = model(input_ids=input_ids, attention_mask=mask,
                            output_hidden_states=True)
                states = out.hidden_states
                pooled = [pool_hidden(states[i].float(), mask, pooling) for i in idx]
                vec = torch.cat(pooled, dim=1) if combine == "concat" \
                    else torch.stack(pooled, dim=0).mean(dim=0)
            window_vectors[chunk] = vec.detach().cpu().numpy().astype(np.float32)
            done += len(chunk)
            if on_batch is not None:
                on_batch(done, len(order))
        return None

    run_with_oom_fallback(run, batch_size, verbose=verbose, what="a batch of sentences")

    # now we fold the windows back into sentences: each sentence is the mean
    # of its windows
    vectors = np.zeros((len(sentences), dim), dtype=np.float32)
    counts = np.zeros(len(sentences), dtype=np.int64)
    for w_i, (s_i, _ids) in enumerate(windows):
        vectors[s_i] += window_vectors[w_i]
        counts[s_i] += 1
    counts[counts == 0] = 1
    vectors /= counts[:, None]
    return vectors, token_counts


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------

def payload_digests(folder: Path) -> Dict[str, str]:
    """sha256 of every file under a checkpoint folder, by relative path --
    what a manifest records so the library can find the folder again."""
    from ..helpers.provenance import file_digest

    folder = Path(folder)
    return {p.relative_to(folder).as_posix(): file_digest(p)
            for p in sorted(folder.rglob("*")) if p.is_file()}


def estimate_minutes(n_texts: int, avg_tokens: float, *, epochs: int, passes: int,
                     params_m: float, device_name: str) -> float:
    """
    A rough wall-time guess for a training run, for the note before it.

    Scaled from distilroberta (82M) fine-tuning 1,000 texts of 200 tokens
    for one epoch: about 20 s on a recent GPU, about 6 minutes on an 8-core
    CPU. Linear in texts, tokens, epochs, passes (folds + final) and
    parameters; a guess within a factor of two, said as such.
    """
    base = 20.0 / 60.0 if device_name.startswith("cuda") else 6.0
    scale = (max(1, n_texts) / 1000.0) * (max(1.0, avg_tokens) / 200.0) \
        * max(1, epochs) * max(1, passes) * (max(1.0, params_m) / 82.0)
    return round(base * scale, 1)


def human_count(n: int) -> str:
    """``82.1M`` for parameter counts."""
    n = int(n)
    if n >= 1_000_000_000:
        return f"{n / 1e9:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def chunk_ids(ids: Sequence[int], body: int) -> List[List[int]]:
    """A text's token ids in consecutive windows of at most ``body`` tokens
    (no overlap: every token is seen once). Empty for an empty text."""
    ids = list(ids)
    if not ids:
        return []
    return [ids[i:i + body] for i in range(0, len(ids), body)]


def truncation_share(token_counts: Sequence[int], max_length: int) -> float:
    """The share of sentences (or texts) longer than the window -- what the
    report says was windowed or cut."""
    counts = list(token_counts)
    if not counts:
        return 0.0
    body = max(1, int(max_length) - 2)
    return sum(1 for c in counts if c > body) / len(counts)


def round_cells(values, rounding: Optional[int]) -> List[object]:
    """Floats for a CSV row, rounded when asked, NaN as blank."""
    out: List[object] = []
    for v in values:
        f = float(v)
        if math.isnan(f):
            out.append(None)
        else:
            out.append(round(f, int(rounding)) if rounding is not None else f)
    return out


def window_length(tokenizer, asked: int) -> Tuple[int, str]:
    """
    The window length to actually tokenize at, and what to say if we moved it.

    The model's own limit is the authority here, and it is not always the
    number its config advertises: RoBERTa reports ``max_position_embeddings``
    of 514 but accepts 512, because its positions start at an offset. Ask for
    514 and the embedding lookup goes out of range mid-run.

    So we ask the tokenizer, which knows the real figure, rather than trusting
    the config or hard-coding 512 -- Longformer and ModernBERT take thousands,
    and a fixed cap would quietly throw away most of what they can read.

    A tokenizer that does not know its limit writes a sentinel in the
    quintillions, and needs no special case: it is never smaller than a length
    anybody asked for, so the comparison below simply leaves it alone. (It got
    one anyway at first, and a mutation check pointed out that nothing could
    reach it.)

    Returns ``(length, note)``; the note is empty when nothing was changed,
    and is meant to be shown. Silently scoring at a different length than
    somebody typed is how a setting comes to look broken.
    """
    limit = int(getattr(tokenizer, "model_max_length", 0) or 0)
    if 0 < limit < asked:
        return limit, (f"max_length {asked} is longer than this model accepts; "
                       f"reading in windows of {limit} tokens instead.")
    return asked, ""
