# Analyzing Text

Language has been the heartbeat of Taters from day one. Or is it the backbone? Or the nervous system? I... don't know. Let's start again — language is a big part of Taters. My background is heavily influenced by this area, and so are the project's roots: the **psychology of verbal behavior**. What follows is some general information to help you grapple with the idea of using text for psychometric purposes regardless of the analytic method used, ranging from **dictionary-based** analyses to **transformer-based** methods, and everything in-between and beyond.

If you're new to this space and want a single starting point, I'd suggest starting here:

* Kennedy, B., Ashokkumar, A., Boyd, R. L., & Dehghani, M. (2022). Text analysis for Psychology: Methods, principles, and practices. In M. Dehghani & R. L. Boyd (Eds.), *The handbook of language analysis in psychology* (pp. 3–62). The Guilford Press. [link](https://osf.io/h2b8t_v1)

For broader context on how we usually think about this stuff, see:

* Boyd, R. L., & Schwartz, H. A. (2021). Natural language analysis and the psychology of verbal behavior: The past, present, and future states of the field. *Journal of Language and Social Psychology, 40*(1), 21–41. [https://doi.org/10.1177/0261927X20967028](https://doi.org/10.1177/0261927X20967028)

* Boyd, R. L., & Markowitz, D. M. (2025). Verbal behavior and the future of social science. *American Psychologist, 80*(3), 411–433. [https://doi.org/10.1037/amp0001319](https://doi.org/10.1037/amp0001319)

When I started doing this kind of work, there weren't a whole lot of people who were deep into the "text analysis in psychology" world. Now, especially with the advent of LLMs, everyone is deeply interested in measuring things about people from words. I could point you toward a huge list of people who have been doing amazing work in this space for years, many of them longer than I've been doing it. There are a massive number of perspectives on language, language-as-data, language-as-behavior, and on and on, so I won't bury you with details here. Across these disciplines and traditions, you'll find a lot of disagreements, some more superficial than others, but everyone agrees: language is cool.

None of this is the "final word" on anything related to measuring people through language. But, these are fairly broad ideas coming from a lot of different traditions, and I've been fascinated and inspired by work coming from a lot of very smart people working on a lot of very difficult questions. So, keep that in mind as you read everything below.

---

## Using text analysis methods in Taters

Most methods accept **analysis-ready CSVs** (`text_id,text`), raw CSVs (with `text_cols` / optional `id_cols` / optional `group_by`), or a folder of documents (e.g., `.txt`, `.docx`, or `.pdf`). Below, you'll find short descriptions and information drawn from the full API. Note that this page is really intended as a starting point to wrap your head around some core concepts — it is not exhaustive in the least, and necessarily reflects my own perspectives, knowledge, and limitations of both.

---

## Dictionary-based analyses

Dictionary methods treat language as **evidence of attention and style**: the words we use (including the "little" ones) systematically reflect cognitive, affective, and social processes. Decades of work have shown that transparent category counts can be highly diagnostic — especially with **function words** and psychologically motivated lexicons. A few starting points:

* Pennebaker, J. W. (2011). *The secret life of pronouns: What our words say about us*. Bloomsbury.

* Tausczik, Y. R., & Pennebaker, J. W. (2010). The psychological meaning of words: LIWC and computerized text analysis methods. *Journal of Language and Social Psychology, 29*(1), 24–54. [https://doi.org/10.1177/0261927X09351676](https://doi.org/10.1177/0261927X09351676)

* Boyd, R. L. (2017). Psychological text analysis in the digital humanities. In S. Hai-Jew (Ed.), *Data Analytics in Digital Humanities* (pp. 161–189). Springer International Publishing. [https://doi.org/10.1007/978-3-319-54499-1_7](https://doi.org/10.1007/978-3-319-54499-1_7)

Taters ships with a number of published dictionaries — moral foundations, personal
values, stereotype content, the General Inquirer, and more — already in your
library and ready to pick. See
[Built-in dictionaries and archetypes](bundled-dictionaries.md) for further information
on each one, how to cite them.

Dictionary-based methods have gotten short shrift in recent years. You'll often hear reviewers ask "What is this, the stone age? Why not just use ChatGPT?" The truth is that dictionary-based methods are a hammer in the toolkit. Sure, it's a blunt tool that isn't very elegant. But, hoo boy, you can get an *awful* lot done with just a hammer. And sometimes, "good enough" really is good enough, depending on your research task.

---

## Sentiment (VADER)

VADER is a lexicon plus a short list of rules — boosters (*very*), negation, punctuation, capitalization, contrastive *but* — rather than a "model" in the sense that you might think of something on Hugging Face. Nothing is trained and nothing is fitted, so the same text always scores the same, it runs at thousands of texts a second on a CPU, and the numbers are reproducible by anybody with the same lexicon. It was built and validated on social media and holds up well beyond it.

Four columns come out: `vader_pos`, `vader_neg` and `vader_neu`, the relative share of the text in each band (they sum to 1), and `vader_compound`, the single summary score from −1 to +1 that most analyses use. Note that the compound score is **not** the mean of the other three: it is a normalized sum of word valences, so a long mild text and a short vehement one can end up with scores that can be compared.

A text is scored whole, as with the other per-text measures. VADER was designed on sentence-length material (e.g., Twitter data), so a long document's compound score is a blunter instrument than a tweet's; if sentence-level sentiment is the question, split the text into sentences before gathering and each will get its own row.

`lexicon_file` swaps in your own valences, in VADER's tab-separated format (`token`, mean valence, standard deviation, ratings), and `emoji_lexicon` does the same for emoji. Both are worth knowing about and worth reporting: a custom lexicon makes the numbers yours rather than VADER's, which is sometimes exactly the point.

* Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. In *Proceedings of the Eighth International AAAI Conference on Weblogs and Social Media* (pp. 216–225). [https://doi.org/10.1609/icwsm.v8i1.14550](https://doi.org/10.1609/icwsm.v8i1.14550)

As a side note, I actually met one of the authors (I'm 96% sure that it was C. J. Hutto) during my very first time attending ICWSM. He seemed extremely nice, and I still think that VADER is a great name for their approach.

---

## Lexical richness analyses

Lexical richness/diversity asks: *how varied is a speaker's vocabulary use?* Classic measures (e.g., TTR, RTTR/CTTR, Herdan's C, Yule's K/I) capture type–token structure, while modern, length-robust metrics (MTLD, MATTR, HD-D, VOCD/D) reduce text-length bias and are widely used in psycholinguistics and language assessment. In **Taters**, these metrics are computed on a per-text basis (or per-group such as `source,speaker`) with tokenization and reproducibility controls (window sizes, draws, seeds). A few helpful references and codebases:

* The original `lexicalrichness` package (reference implementation and API ideas). [https://github.com/LSYS/lexicalrichness](https://github.com/LSYS/lexicalrichness)

* McCarthy, P. M., & Jarvis, S. (2007). vocd: A theoretical and empirical evaluation. *Language Testing, 24*(4), 459–488. [https://doi.org/10.1177/0265532207080767](https://doi.org/10.1177/0265532207080767)

* McCarthy, P. M., & Jarvis, S. (2010). MTLD, vocd-D, and HD-D: A validation study of sophisticated approaches to lexical diversity assessment. *Behavior Research Methods, 42*(2), 381–392. [https://doi.org/10.3758/BRM.42.2.381](https://doi.org/10.3758/BRM.42.2.381)

> Tip: Results can vary with tokenization choices (e.g., handling of hyphens) and window/draw parameters. For strict comparability with prior work or other toolkits, keep those settings consistent and document them in your analysis. Tokenization, everywhere, is always a series of judgment calls and there is no objectively correct choice here. Just standards and norms that are generally defensible and well-understood.

---

## Entropy analyses

Entropy is a word that scares people off, and it should not. Well, okay, it's
a little bit scary sounding. The universe is full of entropy. And, as we all know,
[in space, no one can hear you scream](https://en.wikipedia.org/w/index.php?title=In_space_no_one_can_hear_you_scream).
Thus, entropy is scary, right? Right. Well, okay, maybe not. Strip away the
physics and the information theory: the concept of entropy is asking something very ordinary:
**how surprised should I be by the next word?**

Imagine you are reading somebody's writing one word at a time and trying to
guess what comes next. If they only ever use one word — "turkey turkey turkey turkey" —
you will never be wrong, and there is nothing to be surprised by. Entropy is
zero. If they use eight different words and each one is equally likely, you
will be wrong most of the time, and the entropy would be three bits. Eight equally likely
options is exactly three yes/no questions' worth of uncertainty, which is what
a "bit" means.

That's the whole idea. Everything below is a variation on it.

### Why bother? Taters already measures vocabulary diversity!

Fair question, although you could have asked it in a profoundly more polite way.
Nevertheless... the [lexical richness](#lexical-richness-analyses) measures
above are indeed asking something similar. But, here are a few reasons why you
might consider entropy over classical measures of lexical richness:

**It puts the old measures on one map.** Type-token ratio, Simpson's D and
Yule's K look like three unrelated things named after three different people.
Actually, they are the same idea, but read at three different settings of one
dial. Turn the dial one way and you are just counting distinct words; turn it
the other way and only the single most common word matters; the middle is
Shannon's entropy. Taters reports everything across the whole dial rather than
just three points on it, so instead of arguing about which index to use you can
look at the overall picture.

**It tells you when your text is too short to answer.** This is the big one,
and I'll get into that a bit more below.

**It can tell a rambling text from a repetitive one.** Vocabulary measures
don't know a ramble from a coherent, well-articulated train of thought. More
on that below too.

### Reading the numbers: bits, and "effective number of words"

Every entropy measure provided by Taters is reported twice, because bits are hard
to have intuitions about. "Your participant's writing has 6.4 bits of word entropy"
means nothing to most people. So, beside it, you get the same number as an
**effective number of words** — 6.4 bits is about 84 — which reads as: *this person wrote as
though they were drawing from a pool of 84 equally likely words.* Same fact,
friendlier units. Use whichever you prefer; use the effective number if you are
putting it in a table if somebody else has to read it so that you don't make them sad.

### The short-text problem

Here is the thing that trips people up, and it's why Taters gives you a boatload
of different columns for Shannon entropy instead of plain old vanilla Shannon entropy.

**The simple way of measuring entropy underestimates it. Your text length matters
majorly for traditional entropy measures.*

Try it: take a thousand words that are all equally likely, and write down two
hundred of them at random. The "true" answer is 9.97 bits. Count it up the
obvious way — tally what you actually saw, work out the entropy of that — and
you get **7.44**. Okay, you don't really have to do it. But if you want to, go ahead.
What you'll find is that you are ~2.5 bits too low, and it's not exactly obvious that
the number is a major underestimate.

Now imagine one participant wrote fifty words and another wrote five thousand.
The second person will score higher on the simple measure *even if their
vocabulary is no more varied*, because they gave you more chances to see it.
If you correlate that with anything, you may be measuring how much people
typed/spoke/etc. rather than anything approaching their entropy. Well, not like
*their* entropy, like not their *physical* entropy, but the entropy of their words,
at least. I wonder if the two things are correlated... hmmm. Sounds like a study worth doing.

Anyways, this is not a small effect, and it is way worse for entropy than for type-token
ratio, which is the measure people usually (and rightly) warn you about.

So Taters gives you four corrections alongside the simple count. On that same
sample of two hundred:

| what you get | answer | how it works |
|---|---:|---|
| the simple count | 7.44 | tally what you saw |
| Miller–Madow | 8.09 | add a fudge factor based on how many words you saw |
| Grassberger | 8.91 | a smarter fudge, applied to each word's count |
| NSB | 9.80 | work out which distributions *could* have produced this |
| Chao–Shen | 9.89 | estimate how much you *missed*, then reweight |
| *(the truth)* | *9.97* | |

You are not expected to choose between them on theory. And yes, that previous
table actually has the phrase "smarter fudge" in it. Surprising, right? Now *that's*
entropy, baby. Alright, now, let's start here:

**Look at `ent_word_coverage` first.** This is the share of the vocabulary your
text actually showed you, from 0 to 1. Roughly: if somebody used most of their
words more than once, you saw a lot of their vocabulary and coverage is high.
If nearly every word appeared exactly once, you barely scratched it and
coverage is near zero.

* **Coverage high (say above 0.8)?** All five numbers will be close together.
  Use the plain `ent_word_shannon_bits` and stop worrying.
* **Coverage low?** They will spread out, and *that spread is your error bar*.
  Report one of the corrected ones — `ent_word_chao_shen_bits` holds up best on
  the short, lopsided texts that real people write. Be sure to say which corrected
  version of the score you used when reporting your results. If you do, your name
  will go down in legend and song for being a reliable reporter of results. If you
  *don't*, well, I don't even want to say what Reviewer 2 will do to you. But it
  won't be pretty.

For what it is worth, in a real study that I did on 938 open-ended responses, the mean
coverage was 0.59, the simple count said 6.42 bits and Chao–Shen said 7.25.
Short survey responses are definitely one of the cases where this matters.

> A note on NSB. It is the only one of the four that needs to be told something
> it cannot work out for itself: how many different words the person *could*
> have used. Guess too low and you'll get an underestimate. Guess too high and...
> yup, you'll get an overestimate.  Taters does not ask you to guess this value.
> Instead, it estimates that number from your own text (the `chao1`
> column, which is worth a look in its own right — it is "how many more words
> would I have seen if they had kept writing?").

### Note: "varied" is not the same as "unpredictable"

Take these two texts:

```
the cat sat on the mat the cat sat on the mat the cat sat on the mat
the on mat the sat sat the cat the the cat sat mat the mat on on cat
```

Same words. Same number of each. Every vocabulary measure invented *ever* calls
them identical, and so does every entropy column in the first half of the
table — they all come out to 2.25163 bits, exactly the same.

But one of them is a phrase repeated three times and the other is word salad,
and you would obviously like to know which is which.

That is what the **conditional entropy** columns are for. Instead of asking
"how surprising is the next word", they ask **"how surprising is the next word
once I have seen the one before it?"** In the first text, once you have seen
"the cat" you know "sat" is coming — so given the previous word, there is
almost nothing left to be surprised by: 0.32 bits. In the shuffled version,
knowing the previous word barely helps: 1.60 bits. It's full of surprises.

So:

* `ent_word_shannon_bits` and friends → **how varied** the vocabulary is.
  Order does not matter; shuffle the text and nothing changes.
* `ent_word_conditional2_bits`, `..._conditional3_bits`, `..._rate_bits` →
  **how predictable** the writing is. Order is the whole point.

These measure genuinely different things. A bureaucrat writing in stock phrases
can have a wide vocabulary and be entirely predictable; somebody with a small
vocabulary can put it together in ways you wouldn't ever anticipate.

The three `ent_compress_*` columns measure the same predictability the "crude"
way: they run the text through zip, bzip2 and LZMA and report how well it
squashed, in bits per character. Repetitive text compresses well. It sounds
like a hack and it works surprisingly well, and it has one real advantage — it
makes no assumptions whatsoever about what a "word" is, so it doesn't care
whether the tokenizer understood your language.

### Two units: words and characters

Everything is computed twice.

**Words** are what you would expect: lowercased, punctuation and digits
stripped, split on spaces. This is the same treatment the lexical richness
measures use, so the two sets of columns describe the same stream of words and
you can compare them.

**Characters** keep the punctuation and the spaces, and count letter-by-letter:

```
raw text:   The cat sat. The cat ran!
word units: the | cat | sat | the | cat | ran
char units: t h e _ c a t _ s a t . _ t h e _ c a t _ r a n !
```

Character entropy is worth having for two reasons. It needs no tokenizer, so it
still works on a language Taters tokenizes badly. And it picks up style that
word measures throw away — punctuation habits, spelling, how long the words
are. English character entropy comes out around 4.2 bits, which is the figure
Shannon got in 1951, so it is a good sanity check that nothing has gone wrong.
Side note about Shannon: if you're looking for a good book that isn't *about*
Shannon, but touches on bigger topics that have to do with him, I can recommend
[The Idea Factory](https://en.wikipedia.org/wiki/The_Idea_Factory) by Jon Gertner.

Back to entropy meausures. Every column is named for its unit:
`ent_word_*` or `ent_char_*`. The three compression columns have no
unit because they work on the raw bytes.

### The rest of the columns, briefly

* `ent_word_hill_q0`, `q0_5`, `q2`, `q3`, `qinf` — the dial mentioned at the
  beginning, in effective-number-of-words units. `q0` is simply how many different
  words there were. `q2` is Simpson's index, inverted. `qinf` is driven by the
  single commonest word. Low numbers mean a few words dominate.
* `ent_word_renyi_*` — the same dial in bits, if you prefer bits.
* `ent_word_tsallis_*` — the same idea again in a different formalism. Present
  because some literatures use it; ignore it if yours does not.
* `ent_word_shannon_evenness` — 0 to 1, how close to "all words equally
  common" this text is. Pielou's evenness, if you want the name.
* `ent_word_berger_parker` — the commonest word's share of the text, straight.
* `ent_word_chao1` — estimated vocabulary including the words you did not see.
* `ent_word_tokens`, `ent_char_tokens` — how much text there was. These are
  kept out of the statistics as **bookkeeping** on purpose: length is the
  thing you most want to hold constant here, and the thing you least want
  sneaking in as a predictor. See
  [controls in the statistics guide](stats.md).

### Two things that will look like bugs (but, hey, they're not)

**A conditional entropy came out negative.** Entropy cannot be negative, so
this clearly means that Taters is a flaming dumpster of failure, right? Wrong.
This is the measure telling you that it needs more text. Counting pairs of words needs
more text than counting single words, so on a short response, the pair estimate
is the more broken of the two, and the subtraction goes the wrong way. Taters
leaves it negative rather than rounding it up to zero, because a negative
number is obviously wrong and a zero doesn't really let you know that something
was off. Treat it as "too short to say", and check `ent_word_tokens` to be sure.

**All five Shannon columns disagree wildly.** Check `ent_word_coverage`. If it
is near zero, they *should* disagree — see above.

### I didn't read any of that, I'm too busy for your rambling (which has insanely high entropy, by the way). What do I actually need to know?

1. Look at `ent_word_coverage` before you look at anything else.
2. Use `ent_word_chao_shen_bits` rather than the plain one unless your texts
   are long.
3. If you care about *style* rather than *vocabulary*, the conditional and
   compression columns are the ones you want.

References, for the methods above:

* Hill, M. O. (1973). Diversity and evenness: A unifying notation and its consequences. *Ecology, 54*(2), 427–432. [https://doi.org/10.2307/1934352](https://doi.org/10.2307/1934352)

* Chao, A., & Shen, T.-J. (2003). Nonparametric estimation of Shannon's index of diversity when there are unseen species in sample. *Environmental and Ecological Statistics, 10*, 429–443. [https://doi.org/10.1023/A:1026096204727](https://doi.org/10.1023/A:1026096204727)

* Grassberger, P. (2003). Entropy estimates from insufficient samplings *(arXiv:physics/0307138)*. arXiv. [https://doi.org/10.48550/arXiv.physics/0307138](https://doi.org/10.48550/arXiv.physics/0307138)

* Nemenman, I., Shafee, F., & Bialek, W. (2001). Entropy and inference, revisited. *Advances in Neural Information Processing Systems, 14*. [https://proceedings.neurips.cc/paper/2001/hash/d46e1fcf4c07ce4a69ee07e4134bcef1-Abstract.html](https://proceedings.neurips.cc/paper/2001/hash/d46e1fcf4c07ce4a69ee07e4134bcef1-Abstract.html)

* Shannon, C. E. (1951). Prediction and entropy of printed English. *The Bell System Technical Journal, 30*(1), 50–64. [https://doi.org/10.1002/j.1538-7305.1951.tb01366.x](https://doi.org/10.1002/j.1538-7305.1951.tb01366.x)


---

## Transformer-based analyses

### Archetypes (theory-driven, embedding-based)

<!-- Filed here because it runs on sentence embeddings. Note that the wizard's
     checklist groups it with the dictionaries instead, under "Content
     categories & sentiment", because that is what it measures. Both are
     true; the guide is organized by machinery and the checklist by what you
     get out of it. -->

Archetype analysis encodes each text with a **Sentence-Transformers** model and measures similarity to **curated seed phrases** (for formatting, as an example, see: [https://github.com/ryanboyd/archetypes/blob/main/example_archetypes/Suicidality-Archetypes.csv](https://github.com/ryanboyd/archetypes/blob/main/example_archetypes/Suicidality-Archetypes.csv)). The model handles nuance; your archetype definitions provide **direction** in embedding space. Two
validated archetype dictionaries — resilience and suicidality — come built-in with
Taters and are listed, with their citations, under
[Built-in dictionaries and archetypes](bundled-dictionaries.md#archetype-dictionaries).
Recent examples:

* Varadarajan, V., Lahnala, A., Ganesan, A. V., Dey, G., Mangalik, S., Bucur, A.-M., Soni, N., Rao, R., Lanning, K., Vallejo, I., Flek, L., Schwartz, H. A., Welch, C., & Boyd, R. L. (2024). Archetypes and entropy: Theory-driven extraction of evidence for suicide risk. In *Proceedings of CLPsych 2024* (pp. 278–291). [https://aclanthology.org/2024.clpsych-1.28](https://aclanthology.org/2024.clpsych-1.28)

* Lahnala, A., Varadarajan, V., Flek, L., Schwartz, H. A., & Boyd, R. L. (2025). Unifying the extremes: Developing a unified model for detecting and predicting extremist traits and radicalization. *Proceedings of the International AAAI Conference on Web and Social Media, 19*, 1051–1067. https://doi.org/10.1609/icwsm.v19i1.35860

* Boyd, R. L., Vallejo, I., & Lanning, K. (2026). Toward an integrative science of suicidality: Understanding suicide risk factors through real-world natural language. Journal of Psychopathology and Clinical Science. [https://doi.org/10.1037/abn0001157](https://doi.org/10.1037/abn0001157)

Also, around the same time that we were playing with this idea, other really smart people had similar thoughts (I'm looking at you, Mohammad Atari!). Which makes sense: the archetypes method was inspired by past work from the same origin lab, namely, [this paper](https://link.springer.com/article/10.3758/s13428-017-0875-9). Here are some other papers to read that use the same idea in different and meaningful ways:

* Atari, M., Omrani, A., & Dehghani, M. (2023). Contextualized construct representation: Leveraging psychometric scales to advance theory-driven text analysis. *OSF*. [https://doi.org/10.31234/osf.io/m93pd](https://doi.org/10.31234/osf.io/m93pd)

* Chen, Y., Li, S., Li, Y., & Atari, M. (2024). Surveying the dead minds: Historical-psychological text analysis with contextualized construct representation (CCR) for classical Chinese. In Y. Al-Onaizan, M. Bansal, & Y.-N. Chen (Eds.), *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing* (pp. 2597–2615). Association for Computational Linguistics. [https://doi.org/10.18653/v1/2024.emnlp-main.151](https://doi.org/10.18653/v1/2024.emnlp-main.151)

* Simchon, A., Hadar, B., & Gilead, M. (2023). A computational text analysis investigation of the relation between personal and linguistic agency. *Communications Psychology, 1*(1), 23. [https://doi.org/10.1038/s44271-023-00020-1](https://doi.org/10.1038/s44271-023-00020-1)

Also, you should check out there site here: [https://psychologicaltextanalysis.com/](https://psychologicaltextanalysis.com/)

### Sentence embeddings: a model trained to capture meaning

Have you ever wished that "meaning" could be represented geometrically? Of course you have. Who hasn't? Using semantic embeddings, you can turn a text into a row of numbers with a transformer, and they become comparable. This is the idea behind semantic embeddings. Note that there are multiple popular forms of this. This section focuses explicitly on *sentence-transformers* models, which are a teensy bit different from many "off-the-shelf" models that you might see people use. *sentence-transformers* models are typically trained so that texts meaning the same thing get similar numbers, and things that mean different things get pushed apart. So, a sentence-transformer model will give you embeddings that help you quantify similarity in "meaning": two texts close in this space say similar things, and the vectors work well as predictors straight away.

The *transformer-embeddings* module in Taters computes the same kind of vector straight out of any encoder -- one that was never post-trained to capture "meaning" in the way that a sentence-transformer model would. Transformer embeddings are the specialist's tool, which means that you should probably start with sentence embeddings unless you have a good reason and know what you're doing.

When you want **task-agnostic** semantic features (for clustering, similarity, regression/classification), use `taters.text.extract_sentence_embeddings`. Taters tokenizes into sentences, embeds each sentence, and averages to a single vector per row. Optional L2 normalization makes cosine comparisons straightforward. Example applications:

* Kjell, O. N. E., Sikström, S., Kjell, K., & Schwartz, H. A. (2022). Natural language analyzed with AI-based transformers predict traditional subjective well-being measures approaching the theoretical upper limits in accuracy. *Scientific Reports, 12*, 3918. [https://doi.org/10.1038/s41598-022-07520-w](https://doi.org/10.1038/s41598-022-07520-w)

* Nilsson, A. H., Schwartz, H. A., Rosenthal, R. N., McKay, J. R., Vu, H., Cho, Y.-M., Mahwish, S., Ganesan, A. V., & Ungar, L. (2024). Language-based EMA assessments help understand problematic alcohol consumption. *PLOS ONE, 19*(3), e0298300. [https://doi.org/10.1371/journal.pone.0298300](https://doi.org/10.1371/journal.pone.0298300)

### Transformer embeddings: oh no, now we're getting serious

The sentence-embeddings module (described above) runs on a sentence-transformers model, and asks which one right after the checklist — a list of meaning-tuned models (those already downloaded first, then the usual names: `all-roberta-large-v1` is the default, `all-mpnet-base-v2` nearly as strong and about three times faster, the MiniLMs for a laptop, `paraphrase-multilingual-mpnet-base-v2` for other languages) with a line to type any other. The "Transformer embeddings (raw, any encoder)" step does the same thing — split each text into sentences, encode every sentence, average — with any Hugging Face encoder, an encoder you adapted to your corpus (below), or the encoder inside a predictor you fine-tuned, and lets you choose *how it is read*: which hidden layers (`second_to_last` by default; the last layer of a model that was never fine-tuned is specialized to predicting masked words, and the layers below it transfer better as features), and how token vectors are pooled (mean over the real tokens by default; `[CLS]` means little in an encoder never trained to use it). A sentence longer than the window is windowed with overlap and averaged.

Okay, that was a lot of dense information, I know. Just know that the way that this is implemented is pretty standard stuff for how people in my world would typically use them. It might not be the way that you would do it, but it's worth thinking about your research goal / task at hand and deciding whether this is the right approach. 

The output is `transformer_embeddings.csv`: `token_count`, `sentence_count`, then `e_1 … e_d`, aggregated for the whole text. 

* Tenney, I., Das, D., & Pavlick, E. (2019). BERT rediscovers the classical NLP pipeline. In A. Korhonen, D. Traum, & L. Màrquez (Eds.), *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics* (pp. 4593–4601). Association for Computational Linguistics. [https://doi.org/10.18653/v1/P19-1452](https://doi.org/10.18653/v1/P19-1452)

* Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In K. Inui, J. Jiang, V. Ng, & X. Wan (Eds.), *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)* (pp. 3982–3992). Association for Computational Linguistics. [https://doi.org/10.18653/v1/D19-1410](https://doi.org/10.18653/v1/D19-1410)

* Ganesan, A. V., Matero, M., Ravula, A. R., Vu, H., & Schwartz, H. A. (2021). Empirical evaluation of pre-trained transformers for human-level NLP: The role of sample size and dimensionality. In K. Toutanova, A. Rumshisky, L. Zettlemoyer, D. Hakkani-Tur, I. Beltagy, S. Bethard, R. Cotterell, T. Chakraborty, & Y. Zhou (Eds.), *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies* (pp. 4515–4532). Association for Computational Linguistics. [https://doi.org/10.18653/v1/2021.naacl-main.357](https://doi.org/10.18653/v1/2021.naacl-main.357)

For what it's worth, that last paper by Ganesan et al. is worth a read. The [HLAB](https://humanlanguage.org/) has the idea of using transformer-based models to predict human traits down to a science. Yes, yes, I know what I said. But, so far as I'm aware, they've effectively perfected the idea of using transformer embeddings as features for ridge regressions and have shown time and again their exceptional ability to predict human traits. Read their papers.

### Adapting an encoder to your corpus

Off-the-shelf encoders learned language (mostly) from the internet. Your corpus might not be the internet. Maybe it is but, still, this section is probably relevant to you. If you're working with clinical notes, or therapy transcripts, or posts from one very specific forum, the model represents language in a way that probably isn't quite the same as what you're showing it. Sometimes, that's not a problem. Sometimes it is. But, regardless, you can usually get better results by "adapting" the model to your corpus.

Under "Wrangle Language Models", pick *Adapt an existing model to my texts*. It asks which encoder to start from (your own adapted encoders and anything already on your machine get listed first, then the usual suspects), and then just... keeps training it on your texts. It plays the same "masked word" game that it was originally trained with, no labels needed, but continues learning its shape based on your dataset. This is called domain-adaptive pretraining, and the canonical reference is below:

* Gururangan, S., Marasović, A., Swayamdipta, S., Lo, K., Beltagy, I., Downey, D., & Smith, N. A. (2020). Don't stop pretraining: Adapt language models to domains and tasks. In D. Jurafsky, J. Chai, N. Schluter, & J. Tetreault (Eds.), *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 8342–8360). Association for Computational Linguistics. [https://doi.org/10.18653/v1/2020.acl-main.740](https://doi.org/10.18653/v1/2020.acl-main.740)

What you get back is an ordinary text encoder, saved to your library, that you can then use for embeddings or fine-tuning like any other. You also get a report, because you should want evidence that the adaptation actually did something: a tenth of your texts are held out and never trained on, and the report tells you the held-out loss and perplexity *before and after* under identical masking. If those numbers didn't move, the adaptation didn't help. You also get the loss and learning-rate curves, how many of your texts were long enough to be cut into windows, and — my favorite part — the words your corpus uses that the base tokenizer shatters into fragments. So cool, right? That list is usually a pretty good argument for why you adapted in the first place.

One practical warning: this is slow on a CPU. Over a few thousand short texts, DistilRoBERTa takes an hour or more on a CPU, or just a few minutes a GPU. If you're stuck on CPU, `adapt_train_layers: 2` trains only the top two layers and roughly halves the wait time. Neat!

### Training an encoder from scratch (are you sure? you probably don't want to do this)

*Train a model from scratch → A transformer, from random weights* is really heavy-duty. It's there because somebody with a big corpus and a serious GPU might genuinely want an encoder that has never seen anything but their texts. Most people do not want this. I'm telling you up front so you can skip to the next section with a clear conscience. And, honestly, if you're getting into that kind of hardcore model training, you probably don't want to use Taters to do it. You can almost certainly get the job done better (and faster) using a much more bespoke setup and pipeline.

Still here? Okay, no problem. Here's what it does. It starts from random weights and also learns its own **tokenizer** from your corpus — byte-level BPE, trained on your training texts only. That's really the major appeal to doing this yourself: a corpus of clinical notes or forum posts ends up with a vocabulary made of its own words, whole, instead of one borrowed from web text that chops them into fragments. The training objective is masked-language modeling in the RoBERTa style: dynamic masking and no next-sentence task. But, remember, the major models out there that you see everyone else using have seen *tons* of text, so they probably do pretty okay on your text too.

You get to pick a size for the model that you train. *Small* is 4 layers and 256 wide, about 10M parameters — an overnight run on one GPU. *Base* is what you'd expect if you're familiar with things like BERT or RoBERTa (12 layers and 768 wide, about 110M parameters), but it will take days to train. There's also a *custom* option if you know what you're doing. Before anything starts, the app estimates how much time it might take for your corpus, on your machine, and asks whether you're sure. Are you sure? Again, you probably aren't. You probably don't want to train your own model. Seriously. 

Training stops on its own when the held-out loss stops falling, and keeps the best epoch rather than the last one. If you have more than one GPU visible, it uses all of them from a single process (`DataParallel` — not as efficient as a proper distributed launch. Again, rolling your own is probably a better idea than using Taters).

Two things to know before you commit.

**A language model learns language from quantity.** Below a few million words, the thing you train will be worse at everything than any pretrained model you could have adapted instead. Taters won't stop you — testing the machinery on a small corpus is a perfectly okay thing to do — but the report is going to tell you that your model probably isn't going to be good for much.

**The result is just an encoder.** It lands in your library next to the adapted ones (marked *trained from scratch*, reporting its final perplexity rather than a before-and-after, since there's no "before"), and everything downstream treats it like any other encoder.

* Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). RoBERTa: A robustly optimized BERT pretraining approach. *arXiv:1907.11692*. [https://arxiv.org/abs/1907.11692](https://arxiv.org/abs/1907.11692)

### Fine-tuning a predictor

Everything above produces *features* — numbers you then feed to a statistical model. Fine-tuning skips the middleman: you train the encoder itself to predict your outcome directly.

*Fine-tune a transformer to predict outcomes from text* asks for the encoder first, then your spreadsheet, then which columns to predict. A numeric column becomes a regression; a column of labels becomes a classification. Pick several columns and you get one model with one shared encoder and a separate head per outcome (i.e., "multi-task learning"). A row missing one outcome still gets used to train the others rather than the whole row being thrown out.

What happens during training is that your rows get assigned into folds. Each fold trains a fresh model, holds back a validation slice for early stopping, and predicts only the rows it never saw. **Every number in the report is out of fold.** For a measurement that means R², r, ρ, RMSE and MAE; for a category it means accuracy against the "always-guess-the-most-common-class" baseline, AUC, macro F1, log loss and a confusion matrix. Reports provide per-fold means and standard errors, so you can see whether one lucky fold did all of the actual work. The metrics table provides the same columns as the ridge and classifier models use, so you can compare the two kinds of models directly against each other rather than squinting at two spreadsheets trying to figure out which numbers to compare.

The report gives you loss curves per fold, a predicted-versus-observed scatter, the confusion matrix as a picture, and a methods paragraph you can paste.

The final model is trained on every row for the median best epoch and saved with its heads. Add it to your library and it'll score new texts from the checklist — and it writes predicted classes using **your data's own labels**, not `0` and `1`. If you do want to rename a class (`0` → `control`) or change how the model gets applied, that's something that you can change on a per-model basis under the settings menu (Settings → Manage Taters data → Manage saved models).

You can start from a Hugging Face encoder, an encoder you adapted, or a predictor you fine-tuned earlier. That last one is a warm start: its encoder continues training and its heads are reused for any outcome with the same name.

Long texts are handled the way adaptation handles them (see above). Anything longer than `predictor_max_length` tokens (256 by default) isn't truncated — it's read in overlapping windows. Every window trains with the text's labels and gets weighted so a long text still counts once overall, and the text's prediction is the mean across its windows. A tweet is one window. A 3,000-word essay is a dozen, and scoring it costs a dozen ordinary batches rather than one enormous one.

**Set your expectations.** Let's say that you want to fine-tune DistilRoBERTa against a thousand texts — three epochs, five folds, plus the final model. That will probably take about ten minutes on a recent GPU, or thirty to sixty minutes on an eight-core CPU. `predictor_train_layers: 2` roughly halves that, and MiniLM is about five times faster again. Below a few hundred labeled texts, a ridge regression over embeddings usually does just as well for a fraction of the trouble, and it's a lot easier to break down into something interpretable (in my opinion). The report will tell you when that's the case.

* Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In J. Burstein, C. Doran, & T. Solorio (Eds.), *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)* (pp. 4171–4186). Association for Computational Linguistics. [https://doi.org/10.18653/v1/N19-1423](https://doi.org/10.18653/v1/N19-1423)


---

## Word vectors

A word-vector model learns, from nothing but which words keep company with which, that the word *grave* means something a lot closer to the word *death*, and neither of those words is anywhere close to the word *picnic*. Nobody told it that that was the case, it just read a lot of texts and started to notice that all on its own. Cool, right? Right.

Why train your own instead of downloading somebody else's? Because words mean different things in different places. What "home" means in a bereavement forum is not what it means in a real-estate listing, and a model trained on your corpus captures *your* writers' usage. The flip side: if your corpus is small, a pre-trained set (GloVe, word2vec, fastText) brings a general sense of the language that your texts alone can't supply.

* Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. In C. J. C. Burges, L. Bottou, M. Welling, Z. Ghahramani, & K. Q. Weinberger (Eds.), *Advances in Neural Information Processing Systems 26* (pp. 3111–3119). Curran Associates. [http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

* Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching word vectors with subword information. *Transactions of the Association for Computational Linguistics, 5*, 135–146. [https://doi.org/10.1162/tacl_a_00051](https://doi.org/10.1162/tacl_a_00051)

* Pennington, J., Socher, R., & Manning, C. (2014). GloVe: Global vectors for word representation. In A. Moschitti, B. Pang, & W. Daelemans (Eds.), *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)* (pp. 1532–1543). Association for Computational Linguistics. [https://doi.org/10.3115/v1/D14-1162](https://doi.org/10.3115/v1/D14-1162)

**In Taters:** there are a couple of ways to get to the same result. "Wrangle Language Models" on the main menu (*Train a model from scratch → Word embeddings*) trains word2vec or fastText on your texts, or imports a pre-trained file. You'll need `pip install "taters[vectors]"` for gensim. Alternatively, the "Word vectors: train on these texts" row on the feature checklist does the same thing inside an ordinary pipeline. You probably want to train your model separately so that you can actually look over the results and think about whether it did a good job, rather than blindly run it as part of a larger pipeline only to find out that it didn't do a good job after like 48 hours of grinding away on your dataset.

Either way, you end up with a few things as a result:

**Features** (`word_vectors.csv`). Per text: the mean vector of its words (`wv_1 … wv_k`) and how many of its words the model actually knew (`in_vocab_count`). Glance at that second column — if the number is small for a given text, it's almost certainly a problem.

When the saved model later *scores* texts, it can also add a `sim_<dictionary>__<category>` column for every category of every **concept dictionary** attached to it. This is where it gets fun. Concepts are written as LIWC-22 dictionaries — the same `.dicx`/`.csv` files the dictionary analyzer uses, from the same library. Columns are categories, rows are terms, and a cell is that term's weight in that category (`X` means 1; write `2` if you want *happy* to count twice as heavily as *calm*). Wildcards (`abrad*`) and multi-word terms work exactly as LIWC reads them. Each term gets resolved against the model's vocabulary the way LIWC resolves it against text — `abrad*` becomes every vocabulary word starting with *abrad*, a phrase becomes the mean of its words — and counts as one term no matter how many words it matched. The category's vector is the weighted mean of its terms' vectors, and your text's column is the cosine to it. So a hypothesis becomes one column, and a sixteen-category dictionary becomes sixteen.
Remember when I gave you the DDR paper above? Wait, what? You haven't spent your evenings reading this page in its entirety? Yeesh, okay, fine, [here it is again](https://link.springer.com/article/10.3758/s13428-017-0875-9). That's the citation you'll want to use for this general method/approach.

Note that concept dictionaries are an **apply** setting, chosen per model under Settings → Manage saved models, not something you decide at training time. Training a model and scoring texts against concepts are two different acts and shouldn't be tangled together.

**A model** (`models/word_vectors.json`, with its matrix beside it as `.npy`). Add it to your library from the finish screen and it'll score any new dataset from the checklist under "Score with models I already have", reading text exactly the way it was trained — same lowercasing, same lemmatization decision, same tokenizer. This matters more than it sounds: a ridge fitted on these features carries the model with it and re-applies it to the new corpus. Retraining vectors on the new corpus would give you different dimensions meaning different things, and your model would be nonsense.

**Evidence** (`word_vectors_report.md`). A methods paragraph ready to paste, every setting, how much of your corpus the vocabulary covers, which words fell below the minimum count, training loss per epoch as both a table and a plot, and the nearest neighbors of a dozen probe words drawn as word clouds under `features/figures/wordclouds/word_vectors/`.

About those probes: the defaults are a fixed list of common content words (*time, people, work, life, home, family, money, love, food, friend, think, good*), skipping any the model never learned. Fixed, rather than "your corpus's most frequent words," because the most frequent words in any corpus are [function words](https://www.researchgate.net/publication/237378690_The_Psychological_Functions_of_Function_Words), and the nearest neighbors of *the* are *and*, *that* and *with* in every corpus that has ever been trained. That's a beautiful word cloud of absolutely nothing. Replace the list with words your study actually turns on (`wv_probes`) — and if your corpus is in another language, replacing them is the only way to get useful probes at all.

**Look at the neighbors before you trust a `sim_` column.** They are the evidence for whether your model learned what you think it learned.

One thing that surprises people: nothing is ever stripped from the *training* text. The model trains on words as people wrote them, and — unlike the frequency steps — it does not lemmatize unless you ask (`wv_lemmatize`). That's on purpose. *Was* and *be*, *cats* and *cat*, keep different company, and the company a word keeps is the entire thing a word vector is made of. But, you can still lemmatize if you want. There are legitimate reasons to do so, it's just not my typical default. Your call.

**How words get averaged** is a property of the *model*, not the pipeline: `tokens` (every occurrence counts), `types` (each distinct word once) or `sif` (smooth inverse frequency, where the commonest words count least — this needs the counts a trained model has, so it isn't available for imported vectors). There's also an option to scale every word vector to unit length first. The concept dictionaries are stored *inside* the model — the actual terms and weights, not a file path — so a model applied on a different machine measures the same concepts. Change the dictionaries or the averaging per model under Settings → Manage Taters data → Manage saved models, and every later run uses the new values.

The report's *Concepts* table tells you, per category, how many of its terms the model knew, how much weight is behind its vector, and which terms missed. A category where the model knows none of the terms gets refused by name rather than quietly written as a column of blanks.

**Two honest limits.** Words outside the vocabulary are skipped — and yes, that includes fastText, even though its subword vectors could technically guess at them. Those subword vectors are hundreds of megabytes and mostly noise for feature extraction, so we don't ship them. That's what `in_vocab_count` is for. And a corpus of a few hundred short texts is far too small to train on: the loss curve and the neighbor clouds will make that obvious, and below a few hundred thousand words a pre-trained set is usually the better call.

---

## Readability analyses

Readability formulas — Flesch-Kincaid and its many cousins — estimate how demanding a text is to read, mostly from word and sentence length. They are old, blunt, and weirdly durable. Useful as covariates, useful as sanity checks, and genuinely the point when "how hard is this to read?" is your actual research question. Taters computes the standard battery via `textstat`, one row per text.

* DuBay, W. H. (2004). *The principles of readability*. Impact Information.

* Crossley, S. A., Skalicky, S., & Dascalu, M. (2019). Moving beyond classic readability formulas: New methods and new models. *Journal of Research in Reading, 42*(3-4), 541–561. [https://doi.org/10.1111/1467-9817.12283](https://doi.org/10.1111/1467-9817.12283)

**In Taters:** the "Readability scores" step in the app, or `analyze_readability` from Python/CLI.

---

## Word and phrase frequencies (n-grams)

Sometimes the most informative thing you can do is just *count*, you know? Which words and phrases does this corpus actually use? How often? And when two words sit next to each other, is that a real unit of meaning or a coincidence?

Taters builds a corpus-wide frequency list of words and phrases (1- to n-grams) with collocation statistics — NPMI and logDice — that answer exactly that last question. They're what tells you "snow leopard" is a thing and "snow yesterday" is not. This frequency list is also the vocabulary that feeds the document-term matrix and the topic models below, so it's doing double duty.

* Bouma, G. (2009). Normalized (pointwise) mutual information in collocation extraction. *Proceedings of the Biennial GSCL Conference*, 31–40.

**In Taters:** the "N-gram frequency list" step. The options you'll actually care about are lemmatization, part-of-speech tagged counting (so the verb "felt" and the noun "felt" don't get merged), stop lists, and whether to tag with NLTK or Stanza. It streams, so a corpus far bigger than your RAM is fine.

Punctuation is not counted unless you ask for it (`keep_punctuation`). A frequency list whose top term is "." is counting sentence boundaries, not words, which helps nobody. Well, almost nobody, unless you study sentence boundaries. I'm sure that somebody has made a lucrative career out of sentence boundaries, but they're probably not using Taters. Emoticons get thrown out with the punctuation — there's no letter in `:-)` — so switch it on if emoticons are what you study (which is even more lucrative than studying sentence boundaries). This setting is shared with the document-term matrix and the topic models, because all three have to read text the exact same way or their vocabularies won't line up.

The list also gets drawn as a word cloud of the most frequent terms under `features/figures/wordclouds/ngram_frequencies/`, leaving out phrases that only repeat words already in the picture ("of the" sitting next to "of" and "the" is not useful information).

---

## Document-term matrices

The document-term matrix (DTM) is the workhorse of classic computational text analysis: one row per document, one column per term, and cells holding counts — or binary presence, or relative frequencies, or TF-IDF, depending on what you're doing. It's the bridge between "a folder of texts" and nearly every multivariate method you might want to run.

**In Taters:** the "Document-term matrix" step. It takes its vocabulary from the frequency list above, so the two steps can never disagree about tokenization or lemmatization. It scans with longest-match-wins, so "health behaviors" doesn't *also* get counted as "health". And it names the output file after the weighting, so your count, binary and TF-IDF matrices can live side by side without stepping on each other.

* Schwartz, H. A., Eichstaedt, J. C., Kern, M. L., Dziurzynski, L., Ramones, S. M., Agrawal, M., Shah, A., Kosinski, M., Stillwell, D., Seligman, M. E. P., & Ungar, L. H. (2013). Personality, gender, and age in the language of social media: The open-vocabulary approach. *PLOS ONE, 8*(9), e73791. [https://doi.org/10.1371/journal.pone.0073791](https://doi.org/10.1371/journal.pone.0073791)

* Son, Y., Bayas, N., & Schwartz, H. A. (2018). Causal explanation analysis on social media. In E. Riloff, D. Chiang, J. Hockenmaier, & J. Tsujii (Eds.), *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing* (pp. 3350–3359). Association for Computational Linguistics. [https://doi.org/10.18653/v1/D18-1372](https://doi.org/10.18653/v1/D18-1372)

---

## Topic modeling: the Meaning Extraction Method (MEM)

The MEM is a simple idea that has aged extremely well: build a document-term matrix over frequent content words, run a PCA with varimax rotation, and read the rotated components as *themes* — clusters of words that rise and fall together across documents.

It's been a mainstay of psychological text analysis for good reasons. The themes are transparent, because you can just read the loadings and see what's in them. And the scores drop straight into ordinary statistics without any special handling.

* Chung, C. K., & Pennebaker, J. W. (2008). Revealing dimensions of thinking in open-ended self-descriptions: An automated meaning extraction method for natural language. *Journal of Research in Personality, 42*(1), 96–132. [https://doi.org/10.1016/j.jrp.2007.04.006](https://doi.org/10.1016/j.jrp.2007.04.006)

* Boyd, R. L. (2017). Psychological text analysis in the digital humanities. In S. Hai-Jew (Ed.), *Data Analytics in Digital Humanities* (pp. 161–189). Springer.

* Markowitz, D. M. (2021). The meaning extraction method: An approach to evaluate content patterns from large-scale language data. *Frontiers in Communication, 6*. [https://doi.org/10.3389/fcomm.2021.588823](https://doi.org/10.3389/fcomm.2021.588823)

**In Taters:** the "Topic model: meaning extraction method" step. It fits the themes and writes per-document theme scores, the term-by-theme loadings, two variance tables (more on those in a second) — and a **reusable model file**.

That last one matters more than it might look. Import the model into your library, pick "Score with models I already have", and you can score a *new* dataset on the *same* themes, with the original vocabulary, tokenizer settings and standardization frozen inside the model. That's what makes two studies comparable.

A few implementation notes for the curious: the PCA is exact rather than approximated, it streams so corpus size isn't a memory problem, and its varimax matches what R's `psych` package produces.

**How many themes?** Left to decide for itself, MEM uses parallel analysis: a theme is kept while its eigenvalue beats what a random matrix of the same shape would produce at that rank. In other words, a theme has to explain more than noise does. See [How many topics?](#how-many-topics) for the other rules, and for why a fixed cutoff is a bad idea on a wide matrix.

### Two variance tables, and why they are separate

MEM writes two variance tables, and mixing them up is the single easiest way to conclude that the math is broken when it absolutely is not.

`<name>_eigenvalues.csv` is the **spectrum**. One row per rank, with the correlation matrix's eigenvalue, the level chance alone reaches at that rank, and whether it was kept. Every rank is listed, not just the kept ones, so you can see exactly where the curve crosses over:

| rank | eigenvalue | chance_threshold | kept |
|---:|---:|---:|---|
| 1 | 23.25 | 3.04 | yes |
| 2 | 7.71 | 2.98 | yes |
| 62 | 2.07 | 2.06 | yes |
| 63 | 2.04 | 2.05 | |

That's the curve you judge signal by, and it's what the selection rule actually read.

`<name>_theme_variance.csv` is a different thing: what each finished **theme** accounts for — its sum of squared loadings, and that as a percent.

Why two files instead of one? Because before rotation, "eigenvalue" and "sum of squared loadings" are the same number. Varimax then rotates the kept axes inside the space they span. They stop being eigenvectors, so they stop having eigenvalues, and the variance gets deliberately *spread around* — which is exactly the thing that makes a theme nameable in the first place.

So the theme table comes out much flatter than the spectrum, and **theme 3 is not built from eigenvector 3**. Both lists happen to come out sorted descending, and that is all they have in common. Put side by side in one table, that coincidence looks like a promise, and it isn't one.

Each theme also gets drawn as a word cloud under `features/figures/wordclouds/topic_model_mem/` — the thirty terms loading most strongly, sized by their loading, blue for positive and red for negative — with an `index.md` beside them naming each theme's share of the variance. Reading the clouds is far and away the fastest way to decide what to call a theme. The loadings table is what you report.

---

## Topic modeling: LDA and NMF

MEM isn't the only way to find topics, and the other two answer a genuinely different question. All three read your corpus, build their own vocabulary from it, and give you a score per document per topic plus a word cloud of each topic — so you can run any of them, or all three, exactly the same way.

**LDA** (latent Dirichlet allocation) is what most papers mean when they say "topic model". It tells a story about how your corpus got written: every document is a mixture of topics, every topic is a distribution over words, and fitting the model runs that story backwards. What you get is the *proportion* of each document belonging to each topic, and those proportions add up to one. That reads very naturally — "this interview was mostly about work" — and has one consequence you need to know about before you analyze them. See below.

**NMF** (non-negative matrix factorization) asks something simpler: split the matrix into two non-negative pieces whose product is close to the original. No probabilities, nothing that has to sum to one. On short texts — tweets, open-ended survey answers, single utterances — LDA often struggles, because there isn't enough of each document to infer a mixture from, and NMF's topics come out noticeably sharper. Its weights are just weights, which also makes them easier to use as predictors.

**MEM** finds the dimensions along which word use *co-varies*, and a document scores positive or negative on each.

Here's the one-liner worth remembering: **MEM's themes are contrasts; LDA's topics are ingredients.** A study reporting both is not doing the same thing twice.

### If you have used MALLET

LDA in DLATK runs through MALLET, which is a Java program driven by a gensim wrapper that gensim removed in version 4. Nothing here shells out to Java and nothing here needs a JDK.

Taters uses variational Bayes — the same family scikit-learn and gensim's own `LdaModel` use. You'll get comparable topics, but it is **not** the same algorithm as MALLET's Gibbs sampling, so the numbers will not match and you shouldn't report them as though they do.

### Each model builds its own matrix

LDA is only defined over word *counts*. Hand it anything else and it would fit perfectly happily while meaning nothing at all, so it refuses instead. NMF wants **tf-idf**, because without it the first factor just goes to whatever words happen to be common. MEM takes either of those, plus one-hot.

Since they genuinely want different matrices — and often different vocabulary sizes — each one builds its own, in a folder named after its results file (`topic_model_lda_matrix/` sits beside `topic_model_lda.csv`). You can go look at it. It's the vocabulary your topics came out of, and it's worth a glance when a topic surprises you.

### Which words get to be in the vocabulary

A topic model doesn't look at every word in your corpus. It takes a vocabulary off the top of a frequency list, and `vocab_top_n` decides how many. What that list ranks *by* matters much more than it sounds.

Topic models here rank by **share of documents**, not raw count. Think about why. A word that one transcript repeats four hundred times will top the raw frequency ranking and can tell you nothing, because it only ever describes that one transcript. A word used once each by half your participants is what a topic is actually made of. You can switch back with `vocab rank by` if you want, and the standalone document-term matrix still ranks by frequency, because a feature table generally *does* want the commonest terms.

The other lever is the stop list. Function words are the most frequent words in any corpus by an enormous margin, so without a stop list every single topic comes out as "the, and, of". Punctuation and English stop words are applied by default. The picker is the same one the n-gram step uses, and you can manage the lists under Settings.

### How many topics?

This is the one setting with no good default, and I'm not going to pretend otherwise. Ask for five topics and you get broad ones. Ask for fifty on the same corpus and you get narrow ones. Neither is wrong — they're answers to different questions.

All three models take **0** to mean *you decide for me*, over that model's own matrix, at the moment of fitting.

**MEM's two cheap rules** read numbers the fit already has, so they cost nothing extra:

| rule | keeps a theme while |
|---|---|
| `parallel` (default) | its eigenvalue beats what a random matrix of the **same shape** would give at that rank |
| `kaiser` | its eigenvalue beats a fixed `kaiser_cutoff` |

Why is parallel the default? Because a document-term matrix is *wide*, and wide matrices throw up big eigenvalues out of pure noise. Here's a real example from an actual corpus — 938 documents, 515 terms. Random data of that shape reaches **3.0**. So the textbook Kaiser cutoff of 1.0 kept 196 themes. A cutoff of 1.5 kept 122. Parallel analysis kept 82. A fixed number can't know the shape of your data; parallel analysis measures it.

If you do want `kaiser`, pick a cutoff above that noise ceiling or you aren't really filtering anything. It's roughly `(1 + sqrt(terms / documents)) ** 2`, so it climbs as your vocabulary grows relative to your corpus.

**Two rules that actually fit things.** These work on all three models, and they cost one full fit per candidate count:

| rule | maximizes |
|---|---|
| `coherence` | how often a topic's top words show up in the same documents |
| `coherence_exclusivity` | the harmonic mean of that and *exclusivity* — whether those words are this topic's, or everybody's |

Use the second one. Coherence on its own loves a handful of topics built out of common words, because common words co-occur everywhere. Exclusivity on its own loves whatever splits the corpus into the most distinctive-looking pieces. They fail in opposite directions, which is precisely why you want both — and it's a *harmonic* mean, so a great score on one can't paper over a terrible score on the other.

`coherence_exclusivity` needs `coherence_metric="npmi"`, and it'll say so if you ask for UMass. The reason: NPMI is bounded between −1 and 1, so the two halves balance as they are. UMass isn't bounded, so balancing it would mean rescaling against whichever counts happened to be in your sweep — and then the winner changes when you add one more candidate. That is not a number anybody should put in a paper.

Either way, the evidence lands right next to your results: `<stem>_k_selection.csv` (every metric at every count), `<stem>_k_selection.png` (the chosen score, log x-axis), a report with the top words of every fit, and — for the balanced rule — `<stem>_k_tradeoff.png`, with coherence on the left axis and exclusivity on the right, so you can see the trade-off you're actually picking from.

**Then read the words.** I mean it. These rules are a guide, not a verdict. A model can score beautifully and still cut your corpus somewhere completely useless, and the best-scoring count is often more topics than any human wants to interpret. The literature this comes from hands you the plot so that a *person* looks at it. The automatic answer is a convenience layered on top of that, not a replacement for it.

Two practical notes. Counts your corpus is too small for get skipped and named rather than killing the run, so the default list can reach 2000 without breaking a 60-document study. And a sweep on LDA or NMF is one full fit per count — that's the slow road. MEM is much cheaper, because changing the theme count just re-slices an eigendecomposition it already did.

One more thing worth knowing: the coherence curve is usually much sharper for LDA than for NMF. NMF merges topics in the tail rather than the head, so its top words stay coherent even when the count is wrong. A flat NMF curve means "coherence cannot separate these", not "any of these is fine".

### Using topic proportions in an analysis

LDA's proportions sum to one for every document, which means the last topic is just one minus all the others. It carries nothing the rest don't already have. Two things follow from that, and the standard answers to both are already how Taters works — so this is mostly background rather than a to-do list.

**Don't put all of them in one model at once.** With every topic in a single regression there is no unique answer: many different sets of coefficients fit the data identically well. Schwartz et al. (2013) — the open-vocabulary paper that put 2,000 LDA topics in front of a personality outcome — were explicit about this. They ran *a separate regression for each feature*, with age and gender as covariates, and read each coefficient as that feature's correlation. Taters' correlations and group differences work the same way: one measure at a time, with optional control variables and a correction for multiple comparisons. Nothing for you to change.

**For prediction, reduce first.** That same paper fed its topics to a ridge regression only after running them through PCA. That's worth doing for two reasons at once. It cuts hundreds of topics down to something a model can actually learn from, and it makes the sum-to-one problem vanish on its own — the redundant direction has exactly zero variance, so PCA drops it without anybody having to think about it. In Taters that's the `stats_pca` setting on the analysis step; set it to the name of your topic feature set, or to `all`.

And if you're reporting individual topics, one habit of phrasing: "more of Topic 3" always means "more of Topic 3 *relative to everything else this person could have talked about*", because the shares are shares of one whole. That's a real finding, and a slightly different sentence from the one people usually write.

None of this applies to MEM themes or NMF factors. Neither is constrained to sum to anything.

### Supertopics: reducing topics further

Fifty topics is a lot to read, and a lot to hand to a regression. The analysis stage can reduce them with PCA, and what comes out the other side is a **supertopic** — a combination of topics that vary together across documents.

The columns get named for what they are and where they came from: `topic_model_lda_Supertopic_1`, `topic_model_mem_Supertopic_1`, rather than a generic `Component_1`. So a results table says what it holds without a lookup, and a column quoted in a paper still says which model produced it.

That naming only kicks in when the set being reduced really is a topic model. Reduce readability indices, or dictionary categories, or embedding dimensions, and you get `readability_Component_1` and so on, because that's what those are. Reduce a set that *mixes* — topics together with readability, or two different topic models thrown in together, which is what the combined "all" set does — and you get components, because a blend of two kinds of measure is not a supertopic of either.

The genuinely awkward part used to be working out what a supertopic *was*. Its features are topics, so its word cloud could only name them — `Topic_7`, `Topic_22` — which is a second puzzle stacked on top of the first one. So now two things get drawn:

- **The supertopic in its own words**, under `supertopics/`. The topics' word profiles get composed through the PCA loadings, so you get an ordinary word cloud of actual terms.
- **Each contributing topic's own cloud**, under `themes/`, the same way a ridge over topics already got them.

A supertopic is a *contrast*, not a bag of topics. Loading positively on one topic and negatively on another means "high on this = a lot of the first, very little of the second". Both ends are the finding, so both get drawn — blue rises with the supertopic, red falls with it, exactly the way a MEM theme's cloud reads.

One consequence to expect: a word sitting equally in the topics at both ends cancels out and disappears from the cloud. That's correct — it doesn't distinguish the ends, so it isn't what the supertopic is about — and it has the pleasant side effect of quietly removing words that are simply common.

### Applying topics to a second study

Every topic model saves a reusable model file, and applying it takes no vocabulary or tokenizer settings at all. They come out of the model, because the model *is* the instrument.

That's the whole point: measure your second corpus with the first one's topics, rather than fitting fresh topics and hoping `Topic_3` means the same thing in both. It won't. Fitting again gives you a different number of topics in a different order, and comparing them is comparing nothing.

### Column names

MEM writes `Theme_1..Theme_k`, LDA writes `Topic_1..Topic_k`, NMF writes `Factor_1..Factor_k`. Deliberately different, so running more than one leaves you with three readable sets of columns instead of a collision.

---

## Parts of speech

Sometimes the grammar is the signal. How often somebody uses verbs, or adjectives, or particular tag *sequences* (syntactic n-grams, like determiner-noun) carries stylistic and psychological information that's independent of which words fill the slots.

Taters tags each text — NLTK or Stanza, Penn or Universal tagsets — and writes one row per document of tag frequencies.

**In Taters:** the "Parts of speech" step.

---

## Text cohesion

Cohesion is how a text hangs together on the surface: words repeated between neighboring sentences, connectives ("however", "because", "meanwhile"), pronouns pointing back at something already mentioned. It's the observable machinery that supports *coherence*, which is the model in the reader's head, and it has a big, deep literature behind it — Coh-Metrix, and TAACO more recently.

Taters computes the TAACO family: lexical overlap between adjacent sentences and paragraphs across nine word classes, connectives by category, givenness, synonym overlap via WordNet, and semantic similarity via sentence embeddings. That's roughly 150 columns per text.

Two things to know before you use it.

**150 correlated measures is a lot.** The shipped guide (`COHESION_MEASURES.md`, also printed by `python -m taters.text.analyze_cohesion --explain`) documents every single column: what it measures, how to read it, and which one to pick per family. Read that before you pick, or you'll end up reporting six versions of the same thing.

**This is a reimplementation, not a port.** A careful audit of the reference tool turned up real bugs. Where Taters deliberately deviates — empty cells instead of fake zeros, punctuation that doesn't count as nouns — the guide says exactly how and why, so you can decide for yourself whether you agree.

* Halliday, M. A. K., & Hasan, R. (2014). *Cohesion in English*. Routledge. [https://doi.org/10.4324/9781315836010](https://doi.org/10.4324/9781315836010)

* Graesser, A. C., McNamara, D. S., Louwerse, M. M., & Cai, Z. (2004). Coh-Metrix: Analysis of text on cohesion and language. *Behavior Research Methods, Instruments, & Computers, 36*(2), 193–202. [https://doi.org/10.3758/BF03195564](https://doi.org/10.3758/BF03195564)

* Crossley, S. A., Kyle, K., & McNamara, D. S. (2016). The tool for the automatic analysis of text cohesion (TAACO): Automatic assessment of local, global, and text cohesion. *Behavior Research Methods, 48*(4), 1227–1237. [https://doi.org/10.3758/s13428-015-0651-7](https://doi.org/10.3758/s13428-015-0651-7)

**In Taters:** the "Text cohesion" step. The connectives category lists are editable files in your library — add a file, get a column.

Side note: I've met both Art Graesser and Scott Crossley on more than one occasion. They both seem like really nice guys. If you're wanting to go deeper into the cohesion world (or anything in the educational psych space more broadly), definitely stop reading anything by me and start reading their impressive bodies of work instead. They know this space vastly better than I ever will, and I suspect that the Taters version of coherence/cohesion are probably a bit hacky compared to their work.

---

## Practical notes

* **Interpretability vs. nuance.** Dictionaries are directly interpretable; embeddings are flexible and expressive. Plenty of projects benefit from **both**, and there's no prize for purity.
* **Construct validity.** Whether you're counting or embedding, the **theory** is what matters. Tie your features to constructs you can define, defend, and ideally test across more than one dataset.
* **Reproducibility.** Taters standardizes its I/O, never overwrites unless you ask, and writes predictable outputs under `./features/*/` — so your analyses are easy to rerun, easy to audit, and easy to hand to somebody else six months from now. Including future you, who will not remember any of this.
