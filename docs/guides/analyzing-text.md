# Analyzing Text

Language has been the heartbeat of Taters from day one. My background is heavily influenced by this area, and so are the project's roots: the **psychology of verbal behavior**. What follows is some general information to help you grapple with the idea of using text for psychometric purposes regardless of the analytic method used, ranging from **dictionary-based** analyses to **transformer-based** methods, and everything in-between and beyond.

If you're new to this space and want a single starting point, I'd suggest starting here:

* Kennedy, B., Ashokkumar, A., Boyd, R. L., & Dehghani, M. (2022). Text analysis for Psychology: Methods, principles, and practices. In M. Dehghani & R. L. Boyd (Eds.), *The handbook of language analysis in psychology* (pp. 3–62). The Guilford Press. [link](https://osf.io/h2b8t_v1)

For broader field context and the "verbal behavior" perspective:

* Boyd, R. L., & Schwartz, H. A. (2021). Natural language analysis and the psychology of verbal behavior: The past, present, and future states of the field. *Journal of Language and Social Psychology, 40*(1), 21–41. [https://doi.org/10.1177/0261927X20967028](https://doi.org/10.1177/0261927X20967028)

* Boyd, R. L., & Markowitz, D. M. (2025). Verbal behavior and the future of social science. *American Psychologist, 80*(3), 411–433. [https://doi.org/10.1037/amp0001319](https://doi.org/10.1037/amp0001319)

---

## Using text analysis methods in Taters

Most methods accept **analysis-ready CSVs** (`text_id,text`), raw CSVs (with `text_cols` / optional `id_cols` / optional `group_by`), or a folder of `.txt`. Below, you'll find short descriptions and information drawn from the full API. Note that this page is really intended as a starting point to wrap your head around some core concepts — it is not exhaustive in the least, and necessarily reflects my own perspectives, knowledge, and limitations of both.

---

## Dictionary-based analyses

Dictionary methods treat language as **evidence of attention and style**: the words we use (including the "little" ones) systematically reflect cognitive, affective, and social processes. Decades of work show that transparent category counts can be highly diagnostic — especially with **function words** and psychologically motivated lexicons. A few starting points:

* Pennebaker, J. W. (2011). *The secret life of pronouns: What our words say about us*. Bloomsbury.

* Tausczik, Y. R., & Pennebaker, J. W. (2010). The psychological meaning of words: LIWC and computerized text analysis methods. *Journal of Language and Social Psychology, 29*(1), 24–54. [https://doi.org/10.1177/0261927X09351676](https://doi.org/10.1177/0261927X09351676)

* Boyd, R. L. (2017). Psychological text analysis in the digital humanities. In S. Hai-Jew (Ed.), *Data Analytics in Digital Humanities* (pp. 161–189). Springer International Publishing. [https://doi.org/10.1007/978-3-319-54499-1_7](https://doi.org/10.1007/978-3-319-54499-1_7)

Taters ships with a number of published dictionaries — moral foundations, personal
values, stereotype content, the General Inquirer, and more — already in your
library and ready to pick. See
[Built-in dictionaries and archetypes](bundled-dictionaries.md) for further information
on each one, how to cite them.

---

## Sentiment (VADER)

VADER is a lexicon plus a short list of rules — boosters (*very*), negation, punctuation, capitalization, contrastive *but* — rather than a model. Nothing is trained and nothing is fitted, so the same text always scores the same, it runs at thousands of texts a second on a CPU, and the numbers are reproducible by anybody with the same lexicon. It was built and validated on social media and holds up well beyond it.

Four columns come out: `vader_pos`, `vader_neg` and `vader_neu`, the share of the text in each band (they sum to 1), and `vader_compound`, the single summary score from −1 to +1 that most analyses use. The compound score is **not** the mean of the other three — it is a normalized sum of word valences, so a long mild text and a short vehement one can land in the same place.

A text is scored whole, as with the other per-text measures. VADER was designed on sentence-length material, so a long document's compound score is a blunter instrument than a tweet's; if sentence-level sentiment is the question, split the text into sentences before gathering and each will get its own row.

`lexicon_file` swaps in your own valences, in VADER's tab-separated format (`token`, mean valence, standard deviation, ratings), and `emoji_lexicon` does the same for emoji. Both are worth knowing about and worth reporting: a custom lexicon makes the numbers yours rather than VADER's, which is sometimes exactly the point.

* Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. In *Proceedings of the Eighth International AAAI Conference on Weblogs and Social Media* (pp. 216–225). [https://doi.org/10.1609/icwsm.v8i1.14550](https://doi.org/10.1609/icwsm.v8i1.14550)

---

## Lexical richness analyses

Lexical richness/diversity asks: *how varied is a speaker's vocabulary use?* Classic measures (e.g., TTR, RTTR/CTTR, Herdan's C, Yule's K/I) capture type–token structure, while modern, length-robust metrics (MTLD, MATTR, HD-D, VOCD/D) reduce text-length bias and are widely used in psycholinguistics and language assessment. In **Taters**, these metrics are computed per text (or per group such as `source,speaker`) with tokenization and reproducibility controls (window sizes, draws, seeds). A few helpful references and codebases:

* The original `lexicalrichness` package (reference implementation and API ideas). [https://github.com/LSYS/lexicalrichness](https://github.com/LSYS/lexicalrichness)

* McCarthy, P. M., & Jarvis, S. (2007). vocd: A theoretical and empirical evaluation. *Language Testing, 24*(4), 459–488. [https://doi.org/10.1177/0265532207080767](https://doi.org/10.1177/0265532207080767)

* McCarthy, P. M., & Jarvis, S. (2010). MTLD, vocd-D, and HD-D: A validation study of sophisticated approaches to lexical diversity assessment. *Behavior Research Methods, 42*(2), 381–392. [https://doi.org/10.3758/BRM.42.2.381](https://doi.org/10.3758/BRM.42.2.381)

> Tip: Results can vary with tokenization choices (e.g., handling of hyphens) and window/draw parameters. For strict comparability with prior work or other toolkits, keep those settings consistent and document them in your analysis.

---

## Transformer-based analyses

### Archetypes (theory-driven, embedding-based)

<!-- Filed here because it runs on sentence embeddings. Note that the wizard's
     checklist groups it with the dictionaries instead, under "Content
     categories & sentiment", because that is what it measures. Both are
     true; the guide is organized by machinery and the checklist by what you
     get out of it. -->

Archetype analysis encodes each text with a **Sentence-Transformers** model and measures similarity to **curated seed phrases** (one CSV per construct). The model handles nuance; your archetype definitions provide **direction** in embedding space. Two
validated archetype dictionaries — resilience and suicidality — ship with
Taters and are listed, with their citations, under
[Built-in dictionaries and archetypes](bundled-dictionaries.md#archetype-dictionaries).
Recent examples:

* Varadarajan, V., Lahnala, A., Ganesan, A. V., Dey, G., Mangalik, S., Bucur, A.-M., Soni, N., Rao, R., Lanning, K., Vallejo, I., Flek, L., Schwartz, H. A., Welch, C., & Boyd, R. L. (2024). Archetypes and entropy: Theory-driven extraction of evidence for suicide risk. In *Proceedings of CLPsych 2024* (pp. 278–291). [https://aclanthology.org/2024.clpsych-1.28](https://aclanthology.org/2024.clpsych-1.28)

* Lahnala, A., Varadarajan, V., Flek, L., Schwartz, H. A., & Boyd, R. L. (2025). Unifying the extremes: Developing a unified model for detecting and predicting extremist traits and radicalization. *Proceedings of the International AAAI Conference on Web and Social Media, 19*, 1051–1067. https://doi.org/10.1609/icwsm.v19i1.35860

* Boyd, R. L., Vallejo, I., & Lanning, K. (n.d.). Toward an integrative science of suicidality: Understanding suicide risk factors through real-world natural language. Journal of Psychopathology and Clinical Science. [https://doi.org/10.1037/abn0001157](https://doi.org/10.1037/abn0001157)


* Atari, M., Omrani, A., & Dehghani, M. (2023). Contextualized construct representation: Leveraging psychometric scales to advance theory-driven text analysis. *OSF*. [https://doi.org/10.31234/osf.io/m93pd](https://doi.org/10.31234/osf.io/m93pd)

* Chen, Y., Li, S., Li, Y., & Atari, M. (2024). Surveying the dead minds: Historical-psychological text analysis with contextualized construct representation (CCR) for classical Chinese. In Y. Al-Onaizan, M. Bansal, & Y.-N. Chen (Eds.), *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing* (pp. 2597–2615). Association for Computational Linguistics. [https://doi.org/10.18653/v1/2024.emnlp-main.151](https://doi.org/10.18653/v1/2024.emnlp-main.151)

* Simchon, A., Hadar, B., & Gilead, M. (2023). A computational text analysis investigation of the relation between personal and linguistic agency. *Communications Psychology, 1*(1), 23. [https://doi.org/10.1038/s44271-023-00020-1](https://doi.org/10.1038/s44271-023-00020-1)

### Sentence embeddings: a model trained to capture meaning

Two steps on the checklist turn a text into a row of numbers with a transformer, and they look alike. The difference is what the model was trained for. A *sentence-transformers* model was trained so that texts meaning the same thing get similar numbers, so its vectors are ready-made "meaning" features: two texts close in this space say similar things, and the vectors work well as predictors straight away. That is the sentence-embeddings step, and the usual choice. The *transformer-embeddings* step reads the same kind of vector straight out of any encoder -- one that was never trained to capture meaning -- and is the specialist's tool: it earns its keep when you have adapted or fine-tuned an encoder to your own corpus, or want a particular encoder or layer. Start with sentence embeddings unless you have one of those reasons.

When you want **task-agnostic** semantic features (for clustering, similarity, regression/classification), use `taters.text.extract_sentence_embeddings`. Taters tokenizes into sentences, embeds each sentence, and averages to a single vector per row. Optional L2 normalization makes cosine comparisons straightforward. Example applications:

* Kjell, O. N. E., Sikström, S., Kjell, K., & Schwartz, H. A. (2022). Natural language analyzed with AI-based transformers predict traditional subjective well-being measures approaching the theoretical upper limits in accuracy. *Scientific Reports, 12*, 3918. [https://doi.org/10.1038/s41598-022-07520-w](https://doi.org/10.1038/s41598-022-07520-w)

* Nilsson, A. H., Schwartz, H. A., Rosenthal, R. N., McKay, J. R., Vu, H., Cho, Y.-M., Mahwish, S., Ganesan, A. V., & Ungar, L. (2024). Language-based EMA assessments help understand problematic alcohol consumption. *PLOS ONE, 19*(3), e0298300. [https://doi.org/10.1371/journal.pone.0298300](https://doi.org/10.1371/journal.pone.0298300)

### Transformer embeddings: raw features from any encoder

The sentence-embeddings step runs on a sentence-transformers model, and asks which one right after the checklist — a list of meaning-tuned models (those already downloaded first, then the usual names: `all-roberta-large-v1` is the default, `all-mpnet-base-v2` nearly as strong and about three times faster, the MiniLMs for a laptop, `paraphrase-multilingual-mpnet-base-v2` for other languages) with a line to type any other. The "Transformer embeddings (raw, any encoder)" step does the same thing — split each text into sentences, encode every sentence, average — with any Hugging Face encoder, an encoder you adapted to your corpus (below), or the encoder inside a predictor you fine-tuned, and lets you choose *how it is read*: which hidden layers (`second_to_last` by default; the last layer of a model that was never fine-tuned is specialized to predicting masked words, and the layers below it transfer better as features), and how token vectors are pooled (mean over the real tokens by default; `[CLS]` means little in an encoder never trained to use it). A sentence longer than the window is windowed with overlap and averaged, never cut. The output is `transformer_embeddings.csv`: `token_count`, `sentence_count`, then `e_1 … e_d`, aggregated per speaker for recordings like the sentence embeddings are. The encoder setting is a list, not a text box: your own adapted encoders and fine-tuned predictors first, then every encoder already downloaded to this machine (so a second run costs no download), then the usual names marked as not yet downloaded — and a line to type any other Hugging Face name or checkpoint folder. Tick both embedding steps and you are asked twice in a row, once per step; each question is worded for its kind of model and names the step it is for.

* Tenney, I., Das, D., & Pavlick, E. (2019). BERT rediscovers the classical NLP pipeline. In A. Korhonen, D. Traum, & L. Màrquez (Eds.), *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics* (pp. 4593–4601). Association for Computational Linguistics. [https://doi.org/10.18653/v1/P19-1452](https://doi.org/10.18653/v1/P19-1452)


* Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In K. Inui, J. Jiang, V. Ng, & X. Wan (Eds.), *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)* (pp. 3982–3992). Association for Computational Linguistics. [https://doi.org/10.18653/v1/D19-1410](https://doi.org/10.18653/v1/D19-1410)


* Ganesan, A. V., Matero, M., Ravula, A. R., Vu, H., & Schwartz, H. A. (2021). Empirical evaluation of pre-trained transformers for human-level NLP: The role of sample size and dimensionality. In K. Toutanova, A. Rumshisky, L. Zettlemoyer, D. Hakkani-Tur, I. Beltagy, S. Bethard, R. Cotterell, T. Chakraborty, & Y. Zhou (Eds.), *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies* (pp. 4515–4532). Association for Computational Linguistics. [https://doi.org/10.18653/v1/2021.naacl-main.357](https://doi.org/10.18653/v1/2021.naacl-main.357)


### Adapting an encoder to your corpus

Under "Wrangle Language Models", *Adapt an existing model to my texts* asks first which encoder to start from — your own adapted encoders and the models already on the machine are listed before the usual names — and then continues that encoder's own masked-word pretraining on your corpus — no labels — so that it speaks your texts' dialect before it embeds or predicts anything (domain-adaptive pretraining; Gururangan et al., 2020). A tenth of the texts is held out and never trained on; the report gives the held-out loss and perplexity before and after under the same masks, the loss and learning-rate curves, the share of texts cut into windows, the words your corpus uses that the base tokenizer breaks into pieces, and a methods paragraph with every setting. The adapted encoder is saved to your library as a *text encoder* and is then a base for embeddings or fine-tuning. It is slow on a CPU: distilroberta over a few thousand short texts takes an hour or more there and a few minutes on a GPU; `adapt_train_layers: 2` trains only the top two layers and roughly halves it.

* Gururangan, S., Marasović, A., Swayamdipta, S., Lo, K., Beltagy, I., Downey, D., & Smith, N. A. (2020). Don’t stop pretraining: Adapt language models to domains and tasks. In D. Jurafsky, J. Chai, N. Schluter, & J. Tetreault (Eds.), *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 8342–8360). Association for Computational Linguistics. [https://doi.org/10.18653/v1/2020.acl-main.740](https://doi.org/10.18653/v1/2020.acl-main.740)


### Training an encoder from scratch

*Train a model from scratch → A transformer, from random weights* is the heavy-duty door, and it is offered rather than withheld: someone with a large corpus and a serious GPU may well want an encoder that knows nothing but their texts. It starts from random weights and learns a **tokenizer** from the corpus too — byte-level BPE, learned from the training texts alone — so a corpus of clinical notes or forum posts gets a vocabulary of its own words, whole, rather than one borrowed from web text that shatters them into pieces. The objective is masked-language modeling in the RoBERTa style (Devlin et al., 2019; Liu et al., 2019): dynamic masking, no next-sentence task. You choose a size — *small* (4 layers, 256 wide, about 10M parameters; an overnight run on one GPU), *base* (the BERT/RoBERTa shape, 12 layers, 768 wide, about 110M; days), or *custom* — and before anything runs the app says what it will cost in hours for your corpus on your machine, and asks. Training stops when the held-out loss stops falling and keeps the best epoch; with more than one GPU visible, all of them are used from the one process (`DataParallel`; not as efficient as a distributed launch, and the report says so).

Two things worth knowing before you choose it. A language model learns language from **quantity**: below a few million words the result will be worse at everything than any pretrained model you could have adapted, and the report says so in as many words when the corpus is small — the step does not refuse a small corpus, because testing the machinery on one is a legitimate thing to do, but it will not let a toy pass for a tool. And the result is an ordinary text encoder to everything downstream: it lands in your library beside the adapted ones (marked *trained from scratch*, with its final perplexity rather than a before-and-after), and embeds text or seeds fine-tuning like any other.

* Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). RoBERTa: A robustly optimized BERT pretraining approach. *arXiv:1907.11692*. [https://arxiv.org/abs/1907.11692](https://arxiv.org/abs/1907.11692)

### Fine-tuning a predictor

*Fine-tune a transformer to predict outcomes from text* asks for the encoder first, then the spreadsheet, then which columns to predict, and trains that encoder end to end to predict them: a numeric column is a regression, a column of labels a classification, and several columns train one model with one shared encoder and a head per outcome (multi-task learning; a row missing one outcome still trains the others). The discipline is the ridge's: the rows are dealt into folds, each fold trains a fresh model with a validation slice for early stopping and predicts the rows it never saw, and **every reported number is out of fold** — R², r, ρ, RMSE and MAE for a measurement, accuracy against the always-the-commonest baseline, AUC, macro F1, log loss and the confusion matrix for a category, each with per-fold means and standard errors. Its metrics table carries the ridge's and the classifier's columns so the kinds of model compare in one table; its report has the loss curves per fold, a predicted-versus-observed scatter, the confusion matrix as a picture, and the methods paragraph. The final model, trained on every row for the median best epoch, is saved with its heads; add it to your library and it scores new texts from the checklist, writing predicted classes as **the data's own labels** — and you can rename a class (`0` → `control`) or change how the model is applied per model under Settings → Manage Taters data → Manage saved models. Start it from a Hugging Face encoder, an encoder you adapted, or a predictor you fine-tuned earlier (a warm start on new data: its encoder continues and its heads are reused for outcomes with the same name). Length is handled the way the adaptation step handles it: a text longer than `predictor_max_length` tokens (256 by default) is not cut but read in windows, every window trains with the text's labels and is weighted so that a long text still counts once, and the text's prediction — out of fold, and later on new data — is the mean over its windows. A tweet is one window; a 3,000-word essay is a dozen, and scoring it costs a dozen ordinary batches, not one giant one.

Expectations, honestly: distilroberta over a thousand texts, three epochs, five folds and the final model is about ten minutes on a recent GPU and thirty to sixty on an eight-core CPU; `predictor_train_layers: 2` halves it and MiniLM is about five times faster again. Below a few hundred labeled texts a ridge over embeddings usually does as well, and the report will tell you.

* Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In J. Burstein, C. Doran, & T. Solorio (Eds.), *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)* (pp. 4171–4186). Association for Computational Linguistics. [https://doi.org/10.18653/v1/N19-1423](https://doi.org/10.18653/v1/N19-1423)


---

## Word vectors

A word-vector model learns, from nothing but which words keep company with which, that *grave* sits near *death* and far from *picnic*. Trained on the corpus under study it captures how *these* writers use words — what "home" means in a bereavement forum is not what it means in a real-estate listing — and a pre-trained set (GloVe, word2vec, fastText) brings a general sense of the language to a corpus too small to train on.

* Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. In C. J. C. Burges, L. Bottou, M. Welling, Z. Ghahramani, & K. Q. Weinberger (Eds.), *Advances in Neural Information Processing Systems 26* (pp. 3111–3119). Curran Associates. [http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
)
* Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching word vectors with subword information. *Transactions of the Association for Computational Linguistics, 5*, 135–146. [https://doi.org/10.1162/tacl_a_00051](https://doi.org/10.1162/tacl_a_00051)

* Pennington, J., Socher, R., & Manning, C. (2014). GloVe: Global vectors for word representation. In A. Moschitti, B. Pang, & W. Daelemans (Eds.), *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)* (pp. 1532–1543). Association for Computational Linguistics. [https://doi.org/10.3115/v1/D14-1162](https://doi.org/10.3115/v1/D14-1162)


**In Taters:** two doors. "Wrangle Language Models" on the main menu (*Train a model from scratch → Word embeddings*) trains word2vec or fastText on your own texts (`pip install "taters[vectors]"` for gensim), or brings in a pre-trained file; the "Word vectors: train on these texts" row on the feature checklist does the same inside an ordinary pipeline. Either way you get three things:

* **Features** — `word_vectors.csv`: per text, the mean vector of its words (`wv_1 … wv_k`) and how many words were in the vocabulary (`in_vocab_count`). When the saved model later *scores* texts, it can add one `sim_<dictionary>__<category>` column for every category of every *concept dictionary* set on it — an apply setting, chosen per model under Settings → Manage saved models rather than at training time, because training a model and scoring texts against concepts are different acts. Concepts are spelled as LIWC-22 dictionaries — the same `.dicx`/`.csv` files the dictionary analyzer counts with, picked from the same library: columns are categories, rows are terms, a cell is the term's weight in the category (`X` is 1; write `2` to count *happy* twice as heavily as *calm*), `*` wildcards and multi-word terms exactly as LIWC reads them. Each term is resolved against the model's vocabulary the way LIWC resolves it against text — `abrad*` becomes every vocabulary word starting with *abrad*, a phrase becomes the mean of its words — and counts as one term however many words it matched; the category's vector is the weighted mean of its terms' vectors, and each text's column is the cosine to it. A hypothesis becomes one column per category; a dictionary of sixteen categories becomes sixteen.
* **A model** — `models/word_vectors.json` with its matrix beside it as `.npy`. Add it to your library from the finish screen and it scores any new dataset from the checklist ("Score with models I already have"), reading text exactly as it was trained (lower-cased, lemmatized or not, the same tokenizer). A ridge fitted on these features carries the model and re-applies it to a new corpus — retraining there would give different dimensions.
* **Evidence** — `word_vectors_report.md` beside the model: a methods paragraph ready to paste, every setting, the vocabulary's coverage of the corpus and the words that fell below the minimum count, the training loss per epoch (table and plot), and the nearest neighbors of a dozen probe words, drawn as word clouds under `features/figures/wordclouds/word_vectors/`. The probes are a dozen common content words by default (*time, people, work, life, home, family, money, love, food, friend, think, good*), skipping any the model never learned — a fixed list rather than the corpus's most frequent words, because those are always function words, and the neighbors of *the* are *and*, *that* and *with* in every corpus ever trained. Replace the list with the words your study turns on (`wv_probes`); for a corpus in another language, that is the way to get probes at all. Nothing is ever removed from the *training* text: the model is trained on the words as people wrote them, and — unlike the frequency steps — not lemmatized unless you ask (`wv_lemmatize`), because *was* and *be*, *cats* and *cat*, keep different company, and that company is what a word vector is. The neighbors are the evidence that the model learned what you think it learned; look at them before you trust a `sim_` column.

How words are averaged is a setting of the *model*, not of the pipeline: `tokens` (every occurrence), `types` (each distinct word once) or `sif` (smooth inverse frequency — the commonest words count least; needs the counts a trained model has, so not for imported vectors), with an option to scale every word vector to unit length first. The concept dictionaries are stored *inside* the model — their terms and weights, not a path — so a model applied on another machine measures the same concepts; change them, and the averaging, per model under Settings → Manage Taters data → Manage saved models, and every later run of the model uses the new values. The report's *Concepts* table says, per category, how many of its terms the model knew, the weight behind its vector, and the terms that missed; a category no word of which the model knows is refused by name rather than written as a blank column.

Two honest limits. Words outside the vocabulary are skipped, for fastText as much as word2vec (the subword vectors that would guess at them are hundreds of megabytes and mostly noise for feature extraction), so `in_vocab_count` is worth a glance. And a corpus of a few hundred short texts is way too little for training: the report's loss curve and neighbors will tell you whether the vectors mean anything, and a pre-trained set is often the better choice below a few hundred thousand words.

---

## Readability analyses

Readability formulas (Flesch-Kincaid and their many cousins) estimate how demanding a text is to process, mostly from word and sentence length. They are old, blunt, and surprisingly durable — useful as covariates, as sanity checks, and in any research where "how hard is this to read?" is itself the question. Taters computes the standard battery via `textstat`, one row per text.

* DuBay, W. H. (2004). *The principles of readability*. Impact Information.

* Crossley, S. A., Skalicky, S., & Dascalu, M. (2019). Moving beyond classic readability formulas: New methods and new models. *Journal of Research in Reading, 42*(3-4), 541–561. [https://doi.org/10.1111/1467-9817.12283](https://doi.org/10.1111/1467-9817.12283)

**In Taters:** the "Readability scores" step in the app, or `analyze_readability` from Python/CLI. 

---

## Word and phrase frequencies (n-grams)

Sometimes the most informative thing you can do is simply count: which words and phrases does this corpus actually use, how often, and how tightly do the phrases hang together? Taters builds a corpus-wide frequency list of words and phrases (1- to n-grams), with collocation statistics — NPMI and logDice — that tell you whether "snow leopard" is a real unit of meaning or two words that happened to sit together. The frequency list is also the vocabulary that feeds the document-term matrix and topic model below.

* Bouma, G. (2009). Normalized (pointwise) mutual information in collocation extraction. *Proceedings of the Biennial GSCL Conference*, 31–40.

**In Taters:** the "N-gram frequency list" step. Options you'll actually care about: lemmatization, part-of-speech tagged counting (the verb "felt" vs. the noun "felt"), stop lists, and a NLTK-or-Stanza engine choice for tagging. It streams, so a corpus far bigger than your RAM is fine. Punctuation is not counted unless you ask (`keep_punctuation`): a frequency list whose top term is "." is counting sentence boundaries, not words. Emoticons go with the punctuation — they have no letter in them — so turn it on if they are what you study. The setting is shared with the document-term matrix and the topic model, which must read text exactly the way the list did.

The list is also drawn as one word cloud of the most frequent terms, under `features/figures/wordclouds/ngram_frequencies/`, leaving out phrases that only repeat words already in the picture ("of the" beside "of" and "the").

---

## Document-term matrices

The document-term matrix (DTM) is the workhorse representation of classic computational text analysis: one row per document, one column per term, cells holding counts (or binary presence, relative frequencies, or TF-IDF). It is the bridge between "a folder of texts" and nearly every multivariate method you might want to run.

**In Taters:** the "Document-term matrix" step. It takes its vocabulary from the frequency list above (so the two steps always agree about tokenization and lemmatization), scans with longest-match-wins so "health behaviors" doesn't also count as "health," and names its output file after the weighting so your count, binary, and TF-IDF matrices can coexist. 

* Schwartz, H. A., Eichstaedt, J. C., Kern, M. L., Dziurzynski, L., Ramones, S. M., Agrawal, M., Shah, A., Kosinski, M., Stillwell, D., Seligman, M. E. P., & Ungar, L. H. (2013). Personality, gender, and age in the language of social media: The open-vocabulary approach. *PLOS ONE, 8*(9), e73791. [https://doi.org/10.1371/journal.pone.0073791](https://doi.org/10.1371/journal.pone.0073791)

* Son, Y., Bayas, N., & Schwartz, H. A. (2018). Causal explanation analysis on social media. In E. Riloff, D. Chiang, J. Hockenmaier, & J. Tsujii (Eds.), *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing* (pp. 3350–3359). Association for Computational Linguistics. [https://doi.org/10.18653/v1/D18-1372](https://doi.org/10.18653/v1/D18-1372)


---

## Topic modeling: the Meaning Extraction Method (MEM)

The MEM is a simple, durable idea: build a document-term matrix over frequent content words, run a PCA with varimax rotation, and read the rotated components as *themes* — clusters of words that rise and fall together across documents. It has been a mainstay of psychological text analysis for a good reason: the themes are transparent (you can read the loadings), and the scores drop straight into ordinary statistics.

* Chung, C. K., & Pennebaker, J. W. (2008). Revealing dimensions of thinking in open-ended self-descriptions: An automated meaning extraction method for natural language. *Journal of Research in Personality, 42*(1), 96–132. [https://doi.org/10.1016/j.jrp.2007.04.006](https://doi.org/10.1016/j.jrp.2007.04.006)

* Boyd, R. L. (2017). Psychological text analysis in the digital humanities. In S. Hai-Jew (Ed.), *Data Analytics in Digital Humanities* (pp. 161–189). Springer.

* Markowitz, D. M. (2021). The meaning extraction method: An approach to evaluate content patterns from large-scale language data. *Frontiers in Communication, 6*. [https://doi.org/10.3389/fcomm.2021.588823](https://doi.org/10.3389/fcomm.2021.588823)


**In Taters:** the "Topic model: meaning extraction method" step. It fits the themes, writes per-document theme scores, the term-by-theme loadings, two variance tables (see below) — and a **reusable model file**. That last part matters: import the model into your library and pick "Score with models I already have" and you can score a *new* dataset on the *same* themes later, with the original vocabulary, tokenizer settings, and standardization frozen inside the model. The PCA itself is exact (not approximated), streams so corpus size is not a memory problem, and its varimax matches what R's `psych` package produces. 

How many themes? Left to decide for itself, the step uses parallel analysis: a theme is kept while its eigenvalue beats what a random matrix of the same shape would give at that rank, so a theme has to explain more than noise does. See [How many topics?](#how-many-topics) for the other rules and for why a fixed cutoff is risky on a wide matrix.

### Two variance tables, and why they are separate

MEM writes two, and mixing them up is the single easiest way to conclude the
math is broken when it is not.

`<name>_eigenvalues.csv` is the **spectrum**: one row per rank, with the
correlation matrix's eigenvalue, the level chance alone reaches at that rank,
and whether it was kept. Every rank is listed, not just the kept ones, so you
can see exactly where the curve crosses:

| rank | eigenvalue | chance_threshold | kept |
|---:|---:|---:|---|
| 1 | 23.25 | 3.04 | yes |
| 2 | 7.71 | 2.98 | yes |
| 62 | 2.07 | 2.06 | yes |
| 63 | 2.04 | 2.05 | |

That is the curve to judge signal by, and it is what the rule read.

`<name>_theme_variance.csv` is what each finished **theme** accounts for — its
sum of squared loadings, and that as a percent.

Two files rather than one, and the reason matters. Before rotation,
"eigenvalue" and "sum of squared loadings" are the same number. Varimax then
rotates the kept axes inside the space they span: they stop being
eigenvectors, so they stop having eigenvalues, and the variance gets
deliberately *spread around* — which is the thing that makes a theme
nameable in the first place. So the theme table comes out much flatter than
the spectrum, and theme 3 is **not** built from eigenvector 3. Both lists come
out sorted descending, and that is all they have in common. Side by side in
one table, that coincidence looks like a promise.

Each theme is also drawn as a word cloud under `features/figures/wordclouds/topic_model_mem/` — the thirty terms loading most strongly, sized by their loading, blue loading positively and red negatively — with an `index.md` beside them naming each theme's share of the variance. Reading the clouds is the fastest way to decide what to call a theme; the loadings table is what you report.

---


## Topic modeling: LDA and NMF

MEM is not the only way to find topics, and the other two answer a different
question. All three read your corpus, build their own vocabulary from it, and
give you a score per document per topic plus a word cloud of each topic — so
you can run any of them, or all three, the same way.

**LDA** (latent Dirichlet allocation) is what most papers mean by "topic
model". It tells a story about how the corpus got written: every document is a
mixture of topics, every topic a distribution over words, and fitting runs that
story backwards. You get the *proportion* of each document belonging to each
topic, and those proportions add up to one — which reads naturally ("this
interview was mostly about work") and has one consequence worth knowing about
before you analyze them; see below.

**NMF** (non-negative matrix factorization) asks something simpler: split the
matrix into two non-negative pieces whose product is close to it. No
probabilities, nothing that has to sum to one. On short texts — tweets,
open-ended survey answers, single utterances — LDA often struggles because
there is not enough of each document to infer a mixture from, and NMF's topics
come out sharper. Its weights are just weights, which makes them more
straightforward to use as predictors.

**MEM** finds the dimensions along which word use *co-varies*, and a document
scores positive or negative on each. MEM's themes are contrasts; LDA's topics
are ingredients. A study reporting both is not doing the same thing twice.

### If you have used MALLET

LDA in DLATK runs through MALLET, which is a Java program driven by a gensim
wrapper that gensim removed in version 4. Nothing here shells out to Java, and
nothing here needs a JDK. Taters uses variational Bayes, the same family
scikit-learn and gensim's own `LdaModel` use — comparable topics, but not the
same algorithm as MALLET's Gibbs sampling, so the numbers will not match and
you should not report them as though they do.

### Each model builds its own matrix

LDA is only defined over word *counts*; give it anything else and it would fit
happily and mean nothing, so it refuses. NMF wants **tf-idf**, because without
it the first factor goes to whatever words are merely common. MEM takes either,
plus one-hot. Since they genuinely want different matrices — and often
different vocabulary sizes — each builds its own, in a folder named after its
results file (`topic_model_lda_matrix/` beside `topic_model_lda.csv`). You can
look at it: it is the vocabulary your topics came out of.

### Which words get to be in the vocabulary

A topic model does not look at every word in your corpus — it takes a
vocabulary off the top of a frequency list, and `vocab_top_n` decides how many.
What it ranks by matters more than it sounds.

Topic models here rank by **share of documents**, not raw count. A word one
transcript repeats four hundred times tops the frequency ranking and can tell
you nothing, because it only ever describes that one transcript. A word used
once each by half your participants is what a topic is actually made of. You
can switch back with `vocab rank by`, and the standalone document-term matrix
still ranks by frequency, because a feature table generally does want the
commonest terms.

The other lever is the stop list. Function words are the most frequent words in
any corpus by a wide margin, so without one every topic comes out as "the, and,
of". Punctuation and English stop words are applied by default; the picker is
the same one the n-gram step uses, and you can manage the lists under Settings.

### How many topics?

The one setting with no good default. Ask for five and you get broad topics;
ask for fifty on the same corpus and you get narrow ones. Neither is wrong —
they're different questions.

All three models take **0** to mean *you decide*, over that model's own matrix,
at the moment of fitting.

**MEM's two cheap rules** read numbers the fit already has, so they cost
nothing extra:

| rule | keeps a theme while |
|---|---|
| `parallel` (default) | its eigenvalue beats what a random matrix of the **same shape** would give at that rank |
| `kaiser` | its eigenvalue beats a fixed `kaiser_cutoff` |

Why parallel is the default: a document-term matrix is *wide*, and wide
matrices throw up big eigenvalues out of pure noise. On a real corpus here —
938 documents, 515 terms — random data of that shape reaches **3.0**. So the
textbook Kaiser cutoff of 1.0 kept 196 themes, 1.5 kept 122, and parallel
analysis kept 82. A fixed number can't know the shape of your data; parallel
analysis measures it.

If you do use `kaiser`, pick a cutoff above that ceiling or it isn't really
doing anything. It's roughly `(1 + sqrt(terms / documents)) ** 2`, so it climbs
as your vocabulary grows against your corpus.

**Two rules that actually fit things.** These work on all three models, and
they cost one fit per candidate count:

| rule | maximizes |
|---|---|
| `coherence` | how often a topic's top words show up in the same documents |
| `coherence_exclusivity` | the harmonic mean of that and *exclusivity* — whether those words are this topic's, or everybody's |

Use the second one. Coherence alone loves a handful of topics built out of
common words, because common words co-occur everywhere. Exclusivity alone loves
whatever splits the corpus into the most distinctive-looking pieces. They fail
in opposite directions, which is exactly why you want both — and it's a
*harmonic* mean, so a great score on one can't paper over a terrible score on
the other.

`coherence_exclusivity` needs `coherence_metric="npmi"`, and says so if you ask
for UMass. NPMI is bounded between −1 and 1, so the two balance as they are.
UMass isn't bounded, so balancing it would mean rescaling against whichever
counts happened to be in your sweep — and then the winner changes when you add
a candidate. That's not a number to put in a paper.

Either way, the evidence lands next to your results: `<stem>_k_selection.csv`
(every metric at every count), `<stem>_k_selection.png` (the chosen score, log
x-axis), a report with the top words of every fit, and — for the balanced rule
— `<stem>_k_tradeoff.png`, coherence on the left axis and exclusivity on the
right, so you can see the trade-off you're picking from.

**Then read the words.** These are a guide, not a verdict. A model can score
beautifully and still cut your corpus somewhere useless, and the best score is
often at more topics than anyone wants to interpret. The literature this comes
from hands you the plot so a *person* looks at it; the automatic answer is a
convenience on top of that, not a replacement for it.

Two practical notes. Counts your corpus is too small for get skipped and named
rather than killing the run, so the default list can reach 2000 without
breaking a 60-document study. And a sweep on LDA or NMF is one fit per count —
that's the slow road. MEM is much cheaper, because changing the theme count
just re-slices an eigendecomposition it already did.

Worth knowing: the coherence curve is usually much sharper for LDA than for
NMF. NMF merges topics in the tail rather than the head, so its top words stay
coherent even when the count is wrong, and a flat NMF curve means "coherence
cannot separate these" rather than "any of these is fine".

### Using topic proportions in an analysis

LDA's proportions sum to one for every document, so the last topic is just
one minus all the others — it carries nothing the rest do not. Two things
follow, and the standard answers to both are already how Taters works.

**Do not put all of them in one model at once.** With every topic in a single
regression there is no unique answer: many different sets of coefficients fit
identically. Schwartz et al. (2013), the open-vocabulary paper that put 2,000
LDA topics in front of a personality outcome, were explicit about it — they ran
*a separate regression for each feature*, with age and gender as covariates,
and read each coefficient as that feature's correlation. Taters' correlations
and group differences work the same way, one measure at a time, with optional
control variables and a correction for multiple comparisons. Nothing to change.

**For prediction, reduce first.** The same paper fed its topics to a ridge
regression only after running them through PCA. That is worth doing for two
reasons at once: it cuts hundreds of topics down to something a model can
learn from, and it makes the sum-to-one problem disappear on its own — the
redundant direction has exactly zero variance, so PCA drops it without anybody
having to think about it. In Taters that is the `stats_pca` setting on the
analysis step; set it to the name of your topic feature set, or to `all`.

If you are reporting individual topics, one more habit of phrasing: "more of
Topic 3" always means "more of Topic 3 *relative to everything else this person
could have talked about*", because the shares are of one whole. That is a real
finding and a slightly different sentence from the one people usually write.

None of this applies to MEM themes or NMF factors — neither is constrained to
sum to anything.

### Supertopics: reducing topics further

Fifty topics is a lot to read, and a lot to hand a regression. The analysis
stage can reduce them with PCA, and what comes out is called a **supertopic** —
a combination of topics that vary together across documents. The columns are
named for what they are and where they came from —
`topic_model_lda_Supertopic_1`, `topic_model_mem_Supertopic_1` — rather than
the generic `Component_1`, so a results table says what it holds without a
lookup, and a column quoted in a paper still says which model produced it.

That naming only applies when the set being reduced really is a topic model.
Reduce readability indices, or dictionary categories, or embedding dimensions
and you get `readability_Component_1` and so on, because that is what those
are. Reduce a set that *mixes* — topics together with readability, or two
different topic models thrown in together, which is what the combined "all" set
does — and you get components too: a blend of two kinds of measure is not a
supertopic of either.

The awkward part used to be working out what a supertopic *was*. Its features
are topics, so its word cloud could only name them — `Topic_7`, `Topic_22` —
which is a second puzzle stacked on the first one. So two things are drawn now:

- **The supertopic in its own words**, under `supertopics/`. The topics'
  word profiles are composed through the PCA loadings, so you get an ordinary
  word cloud of terms.
- **Each contributing topic's own cloud**, under `themes/`, the same way a
  ridge over topics already got them.

A supertopic is a *contrast*, not a bag of topics: loading positively on one
topic and negatively on another means "high on this = much of the first, little
of the second". Both ends are the finding, so both are drawn — blue rises with
the supertopic, red falls with it, exactly as a MEM theme's cloud reads.

One consequence worth expecting: a word sitting equally in the topics at both
ends cancels out and disappears. That is correct — it does not distinguish the
ends, so it is not what the supertopic is about — and it has a pleasant side
effect of quietly removing the words that are simply common.

### Applying topics to a second study

Every topic model saves a reusable model file, and applying it takes no
vocabulary or tokenizer settings at all — they come out of the model, because
the model *is* the instrument. That is what makes two studies comparable:
measure the second corpus with the first one's topics, rather than fitting new
topics and hoping `Topic_3` means the same thing in both. Fitting again would
give you a different number of topics in a different order.

### Column names

MEM writes `Theme_1..Theme_k`, LDA writes `Topic_1..Topic_k`, NMF writes
`Factor_1..Factor_k`. Deliberately different, so that running more than one
leaves you with three readable sets of columns rather than a collision.

## Parts of speech

Sometimes the grammar is the signal: how often someone uses verbs, adjectives, or particular tag *sequences* (syntactic n-grams like determiner-noun) carries stylistic and psychological information independent of which words fill the slots. Taters tags each text (NLTK or Stanza; Penn or Universal tagsets) and writes one row per document of tag frequencies.

**In Taters:** the "Parts of speech" step. 

---

## Text cohesion

Cohesion is how a text hangs together on the surface: repeated words between neighboring sentences, connectives ("however," "because," "meanwhile"), pronouns referring back to given information. It is the observable machinery that supports *coherence* — the reader's mental model — and it has a deep literature behind it (Coh-Metrix, and TAACO more recently). Taters computes the TAACO family of indices — lexical overlap between adjacent sentences and paragraphs across nine word classes, connectives by category, givenness, synonym overlap via WordNet, and semantic similarity via sentence embeddings — roughly 150 columns per text.

Two things to know before you use it. First, ~150 correlated measures is a lot; the shipped guide (`COHESION_MEASURES.md`, also printed by `python -m taters.text.analyze_cohesion --explain`) documents every column — what it measures, how to read it, and which one to pick per family. Second, this is a reimplementation from the published definitions, not a port: a careful audit of the reference tool found real bugs, and where Taters deliberately deviates (empty cells instead of fake zeros, punctuation that doesn't count as nouns), the guide says exactly how and why.

* Halliday, M. A. K., & Hasan, R. (2014). *Cohesion in English*. Routledge. [https://doi.org/10.4324/9781315836010](https://doi.org/10.4324/9781315836010)

* Graesser, A. C., McNamara, D. S., Louwerse, M. M., & Cai, Z. (2004). Coh-Metrix: Analysis of text on cohesion and language. *Behavior Research Methods, Instruments, & Computers, 36*(2), 193–202. [https://doi.org/10.3758/BF03195564](https://doi.org/10.3758/BF03195564)

* Crossley, S. A., Kyle, K., & McNamara, D. S. (2016). The tool for the automatic analysis of text cohesion (TAACO): Automatic assessment of local, global, and text cohesion. *Behavior Research Methods, 48*(4), 1227–1237. [https://doi.org/10.3758/s13428-015-0651-7](https://doi.org/10.3758/s13428-015-0651-7)


**In Taters:** the "Text cohesion" step. The connectives category lists are editable files in your library — add a file, get a column. 

---

## Practical notes

* **Interpretability vs. nuance:** Dictionaries are directly interpretable; embeddings are flexible and expressive. Many projects benefit from **both**.
* **Construct validity:** Whether counting or embedding, the **theory** matters. Tie features to constructs you can define, defend, and, ideally, test across datasets.
* **Reproducibility:** Taters standardizes I/O, uses "don't overwrite unless asked," and writes predictable outputs under `./features/*/` — so your analyses are easy to rerun and audit later.
