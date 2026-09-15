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

The sentence-embeddings step uses one sentence-transformers model. The "Transformer embeddings (raw, any encoder)" step does the same thing — split each text into sentences, encode every sentence, average — with any Hugging Face encoder, an encoder you adapted to your corpus (below), or the encoder inside a predictor you fine-tuned, and lets you choose *how it is read*: which hidden layers (`second_to_last` by default; the last layer of a model that was never fine-tuned is specialized to predicting masked words, and the layers below it transfer better as features), and how token vectors are pooled (mean over the real tokens by default; `[CLS]` means little in an encoder never trained to use it). A sentence longer than the window is windowed with overlap and averaged, never cut. The output is `transformer_embeddings.csv`: `token_count`, `sentence_count`, then `e_1 … e_d`, aggregated per speaker for recordings like the sentence embeddings are. The encoder setting is a list, not a text box: your own adapted encoders and fine-tuned predictors first, then every encoder already downloaded to this machine (so a second run costs no download), then the usual names marked as not yet downloaded — and a line to type any other Hugging Face name or checkpoint folder.

* Tenney, I., Das, D., & Pavlick, E. (2019). BERT rediscovers the classical NLP pipeline. In A. Korhonen, D. Traum, & L. Màrquez (Eds.), *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics* (pp. 4593–4601). Association for Computational Linguistics. [https://doi.org/10.18653/v1/P19-1452](https://doi.org/10.18653/v1/P19-1452)


* Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In K. Inui, J. Jiang, V. Ng, & X. Wan (Eds.), *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)* (pp. 3982–3992). Association for Computational Linguistics. [https://doi.org/10.18653/v1/D19-1410](https://doi.org/10.18653/v1/D19-1410)


* Ganesan, A. V., Matero, M., Ravula, A. R., Vu, H., & Schwartz, H. A. (2021). Empirical evaluation of pre-trained transformers for human-level NLP: The role of sample size and dimensionality. In K. Toutanova, A. Rumshisky, L. Zettlemoyer, D. Hakkani-Tur, I. Beltagy, S. Bethard, R. Cotterell, T. Chakraborty, & Y. Zhou (Eds.), *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies* (pp. 4515–4532). Association for Computational Linguistics. [https://doi.org/10.18653/v1/2021.naacl-main.357](https://doi.org/10.18653/v1/2021.naacl-main.357)


### Adapting an encoder to your corpus

Under "Train a model", *Adapt a language model to my texts* asks first which encoder to start from — your own adapted encoders and the models already on the machine are listed before the usual names — and then continues that encoder's own masked-word pretraining on your corpus — no labels — so that it speaks your texts' dialect before it embeds or predicts anything (domain-adaptive pretraining; Gururangan et al., 2020). A tenth of the texts is held out and never trained on; the report gives the held-out loss and perplexity before and after under the same masks, the loss and learning-rate curves, the share of texts cut into windows, the words your corpus uses that the base tokenizer breaks into pieces, and a methods paragraph with every setting. The adapted encoder is saved to your library as a *text encoder* and is then a base for embeddings or fine-tuning. It is slow on a CPU: distilroberta over a few thousand short texts takes an hour or more there and a few minutes on a GPU; `adapt_train_layers: 2` trains only the top two layers and roughly halves it.

* Gururangan, S., Marasović, A., Swayamdipta, S., Lo, K., Beltagy, I., Downey, D., & Smith, N. A. (2020). Don’t stop pretraining: Adapt language models to domains and tasks. In D. Jurafsky, J. Chai, N. Schluter, & J. Tetreault (Eds.), *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 8342–8360). Association for Computational Linguistics. [https://doi.org/10.18653/v1/2020.acl-main.740](https://doi.org/10.18653/v1/2020.acl-main.740)


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


**In Taters:** two doors. "Train a model" on the main menu trains word2vec or fastText on your own texts (`pip install "taters[vectors]"` for gensim), or brings in a pre-trained file; the "Word vectors: train on these texts" row on the feature checklist does the same inside an ordinary pipeline. Either way you get three things:

* **Features** — `word_vectors.csv`: per text, the mean vector of its words (`wv_1 … wv_k`), how many words were in the vocabulary (`in_vocab_count`), and one `sim_<dictionary>__<category>` column for every category of every *concept dictionary* you tick. Concepts are spelled as LIWC-22 dictionaries — the same `.dicx`/`.csv` files the dictionary analyzer counts with, picked from the same library: columns are categories, rows are terms, a cell is the term's weight in the category (`X` is 1; write `2` to count *happy* twice as heavily as *calm*), `*` wildcards and multi-word terms exactly as LIWC reads them. Each term is resolved against the model's vocabulary the way LIWC resolves it against text — `abrad*` becomes every vocabulary word starting with *abrad*, a phrase becomes the mean of its words — and counts as one term however many words it matched; the category's vector is the weighted mean of its terms' vectors, and each text's column is the cosine to it. A hypothesis becomes one column per category; a dictionary of sixteen categories becomes sixteen.
* **A model** — `models/word_vectors.json` with its matrix beside it as `.npy`. Add it to your library from the finish screen and it scores any new dataset from the checklist ("Score with models I already have"), reading text exactly as it was trained (lower-cased, lemmatised or not, the same tokenizer). A ridge fitted on these features carries the model and re-applies it to a new corpus — retraining there would give different dimensions.
* **Evidence** — `word_vectors_report.md` beside the model: a methods paragraph ready to paste, every setting, the vocabulary's coverage of the corpus and the words that fell below the minimum count, the training loss per epoch (table and plot), and the nearest neighbors of your concept seeds and the most frequent words, drawn as word clouds under `features/figures/wordclouds/word_vectors/`. The neighbors are the evidence that the model learned what you think it learned; look at them before you trust a `sim_` column.

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


**In Taters:** the "Topic model: meaning extraction method" step. It fits the themes, writes per-document theme scores, the term-by-theme loadings, an eigenvalue table — and a **reusable model file**. That last part matters: import the model into your library and pick "Score with models I already have" and you can score a *new* dataset on the *same* themes later, with the original vocabulary, tokenizer settings, and standardization frozen inside the model. The PCA itself is exact (not approximated), streams so corpus size is not a memory problem, and its varimax matches what R's `psych` package produces. 

How many themes? Left to decide for itself, the step uses parallel analysis: a theme is kept while its eigenvalue beats what a random matrix of the same size would give at that rank, so a theme has to explain more than noise does. The older Kaiser rule (every eigenvalue above 1) is available as `mem_retain: kaiser`; on a wide matrix it keeps most of them — one real corpus came back with 101 themes. The model file records which rule decided and the eigenvalues beside their chance levels.

Each theme is also drawn as a word cloud under `features/figures/wordclouds/topic_model_mem/` — the thirty terms loading most strongly, sized by their loading, blue loading positively and red negatively — with an `index.md` beside them naming each theme's share of the variance. Reading the clouds is the fastest way to decide what to call a theme; the loadings table is what you report.

---

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
