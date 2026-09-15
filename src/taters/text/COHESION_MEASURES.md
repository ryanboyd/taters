# The cohesion measures, explained

This guide documents every column that `analyze_cohesion` ("Text cohesion"
in the TUI) writes: what it measures, how it is computed, how to read it,
where it comes from, and — where relevant — how it deliberately differs
from TAACO 2.1.3, the reference implementation these measures were audited
against.

**Lineage.** The measure families follow the Tool for the Automatic Analysis
of Cohesion (TAACO: Crossley, Kyle & McNamara, 2016, *Behavior Research
Methods*) and the Coh-Metrix tradition behind it (Graesser, McNamara,
Louwerse & Cai, 2004). Column names match TAACO 2.1.3 wherever the measure
survives, so results can be read against that literature. Nothing was copied
from TAACO (it is CC BY-NC-SA licensed); the measures were reimplemented
from their published definitions, and a line-by-line audit of TAACO 2.1.3
found a number of defects that this implementation deliberately does not
reproduce. Each is flagged below as **Differs from TAACO**.

**The one rule to remember: empty cells are NA, not zero.** A one-paragraph
document has no paragraph-to-paragraph cohesion to measure, so its paragraph
columns are empty. TAACO writes `0.0` there, which downstream statistics
cannot tell apart from "measured, and found zero cohesion" (5.6% of TAACO's
own sample corpus is affected). When you aggregate these columns, treat
empty as missing.

## How the text is prepared

* **Tokens** come from the same tokenizer settings as every other taters
  text step (`engine`/`tokenizer`: potts + NLTK by default, Stanza
  optionally). Text is lowercased before tagging, as in the other tagged
  taters streams — a known tradeoff (e.g., bare "i" can mistag) accepted for
  cross-step consistency.
* **A token counts only if it contains a letter or digit.** *Differs from
  TAACO*: its punctuation filter mixed POS tags with literal characters that
  can never equal a tag, so quotes, hyphens and `%` flowed into the counts —
  `%` is Penn-tagged NN and counted as a **noun** in every noun, content and
  argument index.
* **Sentences** are split by NLTK's punkt (or the model, under Stanza).
* **Paragraphs** are runs of blank lines. *Differs from TAACO*, which
  breaks on **every** newline — under which hard-wrapped text (every PDF
  extraction) makes each visual line a "paragraph" and reduces the
  paragraph indices to noise.
* **Lemmas** are always used for overlap and TTR (WordNet POS-guided under
  NLTK; the model's lemma under Stanza).

### The nine word classes

| class | contents |
|---|---|
| `all` | every counted token's lemma |
| `noun` | NN/NNS/NNP/NNPS (proper nouns included, as TAACO) |
| `pronoun` | PRP/PRP$ **plus unattended demonstratives** (below) |
| `adj` | JJ/JJR/JJS |
| `verb` | content verbs only: VB* minus modals and every form of *be*. *Differs from TAACO*: TAACO also demoted auxiliary *have/do* via the dependency parse; without one, auxiliary *have/do* stay content here. |
| `adv` | RB/RBR/RBS |
| `cw` (content) | nouns + adjectives + content verbs + **deadjectival adverbs** (an *-ly* adverb whose stem has an adjective sense in WordNet — TAACO used a COCA-derived list we cannot ship; WordNet is the substitute) |
| `fw` (function) | every other counted token |
| `argument` | nouns + pronouns (TAACO's definition) |

**Demonstratives** (*this/that/these/those*): "attended" when the next
content-bearing token is a noun or adjective ("**that** car"), otherwise
"unattended" ("I like **that**") — a pronominal use, which joins the pronoun
and argument classes, as in TAACO. *Differs from TAACO*: attendedness there
came from the dependency parse; here it is the next-token heuristic above.

## 1. TTR and density (15 columns)

All computed over **lemmas**, whole-document.

| column | definition |
|---|---|
| `lemma_ttr` | lemma types / lemma tokens |
| `lemma_mattr` | moving-average TTR, window `mattr_window` (default 50; Covington & McFall 2010). Documents shorter than the window fall back to plain TTR. |
| `lexical_density_tokens` | content tokens / all tokens |
| `lexical_density_types` | content types / all types |
| `content_ttr`, `function_ttr`, `function_mattr`, `noun_ttr`, `verb_ttr`, `adj_ttr`, `adv_ttr`, `prp_ttr`, `argument_ttr` | class-restricted TTRs |
| `bigram_lemma_ttr`, `trigram_lemma_ttr` | TTR over lemma n-grams built **within sentences**. *Differs from TAACO*, which built n-grams over the flat document, manufacturing n-grams that span sentence boundaries. |

Interpretation: higher TTR = more lexical diversity = **less** repetition,
i.e. usually *less* cohesion. TTR falls mechanically with text length;
prefer `lemma_mattr` when documents differ in length. A related, word-form-
based family lives in the "Lexical diversity" step (`analyze_lexical_richness`);
these are the lemma/class-based TAACO versions.

## 2. Lexical overlap (108 columns)

The heart of the tool. For each of the nine classes, each unit (sentence,
paragraph) and each window (adjacent, adjacent-two), three indices:

* `adjacent_overlap_<class>_<unit>` — total overlapping **types** / total
  types of the earlier segments. Note the denominator: this is
  type-normalized (TAACO's definition, kept), *not* a share of running
  words.
* `..._div_seg` — mean number of overlapping types per comparison
  (unbounded above).
* `adjacent_overlap_binary_...` — share of comparisons with at least one
  overlapping type.

"Adjacent-two" (`_2_`) compares each segment against the union of the next
**two**; it needs at least three segments.

Reading them: higher = more lexical continuity between neighboring
sentences/paragraphs = more surface cohesion. The `binary` variant is the
most robust to length; `div_seg` the least (it grows with segment size).

A sentence with no members of a class still occupies its slot: "no
adjectives here" is an absence the adjective-overlap denominators are
defined over (TAACO semantics, kept — but be aware that class-sparse texts
pull these indices toward 0 for reasons of class *presence*, not cohesion).

*Differs from TAACO*: (a) documents with too few segments are NA, not 0;
(b) the two-segment window is built as a fresh union — TAACO appended
segment *i+2* **into its shared sentence list**, so the mere act of
computing one index changed the input of the next; its published `_2_` and
semantic values depend on which options were enabled, ours do not.

## 3. Synonym overlap (8 columns)

`syn_overlap_<unit>_<noun|verb>` and `..._prop`, via WordNet (through nltk;
TAACO shipped WordNet-derived files keyed by inflected forms, noisy enough
that *was* mapped to *washington*).

* The unsuffixed column is **TAACO's definition, kept for comparability**:
  for each type of segment *i*, one hit per word of segment *i+1* whose
  synonym set contains it, averaged over segment pairs. Know what it is: a
  multiply-counting, **unbounded count** (audit finding 4.15) that is not
  comparable across texts with different sentence lengths.
* `_prop` is the normalized companion this implementation adds: source
  types matched by at least one next-segment word, over total source types
  — the same scale as the lexical overlap indices. Prefer it.

An out-of-vocabulary lemma is its own synonym set, so exact repetition
always overlaps. Synonym overlap is by construction ≥ the corresponding
lemma overlap.

## 4. Semantic similarity (4 columns, optional)

`semantic_1_all_sent`, `semantic_2_all_sent`, `semantic_1_all_para`,
`semantic_2_all_para`: the mean cosine similarity between sentence-
transformer embeddings of adjacent segments (`_1_`) or between a segment
and the normalized mean of the next two (`_2_`). Range roughly 0–1 for
running text; higher = adjacent segments are about more similar things —
cohesion of *meaning* where the overlap indices measure cohesion of
*wording*.

Controlled by `semantic_model` (default: MiniLM; `"none"` disables and the
step runs without any model; the `device` setting says where it runs).

*Differs from TAACO*, deliberately and entirely: TAACO uses frozen 2014
COCA-derived LSA/word2vec spaces (license-restricted, fixed vocabulary,
unknown words silently dropped) and an LDA index that the audit proved is
**destroyed by an implementation bug** — `safe_divide(x, div)` is applied
to the loop *index* rather than the topic value, so its `lda_*` columns are
a near-constant (SD 0.019 across its own 520-text sample) carrying no
topical information. There is nothing valid there to be compatible with;
the LDA family is dropped, and LSA/word2vec are replaced by modern
embeddings. Do not compare `semantic_*` values numerically to published
`lsa_*` values; directionally they measure the same construct.

## 5. Connectives (22 + 3 columns)

Each column is an **incidence**: occurrences / all counted words. The
category lists ship as editable text files (Manage my library → Connectives
lists); each file is one column, named by the file. Add your own file, get
your own column. The shipped categories follow Halliday & Hasan's (1976)
cohesion taxonomy and the Coh-Metrix connective classes (additive, causal,
logical, temporal; positive = extending, negative = contrasting), plus
TAACO's structural categories (basic connectives, conjunctions,
disjunctions, subordinators, coordinating conjuncts, sentence linking,
order, reason & purpose, opposition, determiners).

**Provenance, per entry.** Inside every list file, entries are grouped by
where the item appears: **[H&H]** Halliday & Hasan's (1976, ch. 5)
conjunction inventory; **[PDTB]** the Penn Discourse Treebank 2.0 inventory
of 100 explicit discourse connectives (Prasad et al., 2008); and **[taters]**
for items curated here that appear in neither (mostly multiword purpose and
sequencing markers). Roughly four fifths of the shipped entries trace to one
of the two published inventories. Nothing is copied from TAACO (its lists
are inside its CC BY-NC-SA source), so counts will differ modestly from
TAACO's even where the construct is the same. A methods section can cite:
*"connective categories after Halliday & Hasan (1976) and Graesser et al.
(2004); item inventories drawn from Halliday & Hasan (1976, ch. 5) and the
PDTB 2.0 explicit connectives (Prasad et al., 2008), as shipped with taters
(full lists included in the package)."*

Matching rules:

* Words and phrases match **within a sentence only**. *Differs from
  TAACO*, which counted with `str.count` over the punctuation-stripped,
  joined document: it both undercounted adjacent repeats ("so so" counted
  once) and manufactured phrase matches across sentence boundaries
  ("…ends in. Fact is…" matched *in fact*).
* Ambiguous words carry a **POS-tag constraint** in the list file
  (`yet<TAB>CC`): conjunction *yet* counts, "not yet" does not. This
  replaces TAACO's dependency-parse caveats — which were themselves part
  dead code (computed, then ignored for the causal categories) and part
  broken (an unsplit caveat string, so membership tests were substring
  matches: "in" matched inside "since"). Where Penn tags cannot
  discriminate (prepositional vs conjunctive *for*, *as*), the entry is
  either omitted or counted unconditionally — the file's comments say
  which.

The shipped categories, column by column (each is occurrences per word;
the file of the same name holds the exact entries and their citations):

| column | what it counts |
|---|---|
| `basic_connectives` | the core coordinators/subordinators (*and, but, or, because, if, so*) |
| `conjunctions` | additive coordination (*and, but*) |
| `disjunctions` | alternatives (*or*) |
| `lexical_subordinators` | subordinating conjunctions introducing a dependent clause (*although, because, unless, while…*) |
| `coordinating_conjuncts` | clause coordinators (*and, but, or, nor, yet, so*) |
| `addition` | simple additives (*also, moreover, in addition…*) |
| `sentence_linking` | conjunctive adverbs linking across sentences (*however, therefore, meanwhile…*) |
| `order` | sequencing (*first, next, finally…*) |
| `reason_and_purpose` | reason/purpose markers (*because, in order to, so that…*) |
| `all_causal` | causal connectives, extending and contrasting |
| `positive_causal` | causal connectives that extend (*because, therefore, as a result…*) |
| `opposition` | adversatives (*but, however, whereas, despite…*) |
| `determiners` | *a, an, the* and the demonstrative determiners (TAACO parity; an incidence, not a cohesion device) |
| `all_additive` | additive connectives, extending and contrasting |
| `all_logical` | logical connectives (*and, or, if, then, therefore…*) |
| `positive_logical` | logical connectives that extend |
| `negative_logical` | logical connectives that contrast or negate |
| `all_temporal` | temporal connectives (*when, until, meanwhile, next…*) |
| `positive_intentional` | goal/intention markers (*so that, in order to…*) |
| `all_positive` | extending connectives across all categories |
| `all_negative` | contrasting connectives across all categories |
| `all_connective` | the union view: every category in one incidence |

`all_demonstratives`, `attended_demonstratives`,
`unattended_demonstratives` are computed from the class machinery (not
lists), as incidences per word.

## 6. Givenness (4 columns)

How much of the text refers back to already-given information.

| column | definition |
|---|---|
| `pronoun_density` | pronoun tokens (unattended demonstratives included) / all words |
| `pronoun_noun_ratio` | pronoun tokens / noun tokens |
| `repeated_content_lemmas` | content tokens whose lemma occurs more than once in the document / **content tokens**. *Differs from TAACO*, which divided this content-word count by ALL words (audit 4.13) — a mixed-class ratio. |
| `repeated_content_and_pronoun_lemmas` | the same over content + pronoun tokens |

Higher = more anaphora and repetition = more given information; texts high
on these typically presume more shared context.

## Dropped families, and why

* **LDA similarity** — provably broken in the reference (see §4); no valid
  published values exist to be comparable with.
* **Source-text overlap** (TAACO's `source*` options, 26 columns) — needs a
  second input text per run (a different input contract) and its keyness
  reference lists are COCA-derived and unshippable. A candidate for a later
  step of its own.
* TAACO's `lsa_*`, `lda_*`, `word2vec_*` column *names* — replaced by
  `semantic_*` to avoid claiming a method that is not used.

## Practical notes

* ~150 columns is a lot of correlated measurement. For most analyses pick
  one representative per family (e.g. `adjacent_overlap_cw_sent`,
  `lemma_mattr`, `all_connective`, `semantic_1_all_sent`,
  `pronoun_density`) rather than modeling all of them.
* Very short documents produce many NA cells by design; filter on `nwords`
  before aggregating.
* Conversation transcripts gathered turn-by-turn have no blank lines, so
  each document is one paragraph: sentence-level indices are the
  meaningful ones there.

## References

* Crossley, S. A., Kyle, K., & McNamara, D. S. (2016). The tool for the
  automatic analysis of text cohesion (TAACO). *Behavior Research Methods,
  48*, 1227–1237.
* Graesser, A. C., McNamara, D. S., Louwerse, M. M., & Cai, Z. (2004).
  Coh-Metrix: Analysis of text on cohesion and language. *Behavior Research
  Methods, Instruments, & Computers, 36*, 193–202.
* Halliday, M. A. K., & Hasan, R. (1976). *Cohesion in English*. Longman.
* Prasad, R., Dinesh, N., Lee, A., Miltsakaki, E., Robaldo, L., Joshi, A., &
  Webber, B. (2008). The Penn Discourse TreeBank 2.0. *Proceedings of
  LREC 2008*.
* Covington, M. A., & McFall, J. D. (2010). Cutting the Gordian knot: The
  moving-average type–token ratio (MATTR). *Journal of Quantitative
  Linguistics, 17*, 94–100.
