# Running Analyses

Everything else in Taters *measures*. This part *asks*.

Once a run finishes you have a folder of feature tables: a row per text, and
columns full of numbers — how readable it was, how much it hangs together, how
often it reaches for social words, where it sits in a 768-dimensional semantic
space. That is a genuine achievement and it is also, on its own, completely
inert. Nobody has ever had a research question whose answer is a spreadsheet.

The analysis stage exists to close that last gap. It takes those columns and the
things you already know about your participants — their condition, their scores,
their age — and answers the three questions people actually turn up with:

| the question you have | the analysis |
|---|---|
| "Do these groups *talk* differently?" | group differences |
| "Does the way people talk *track* this outcome?" | correlations |
| "Can I *predict* this outcome from language alone?" | prediction (ridge / classification) |

You do not need all three, and plenty of perfectly good studies need none of
them. The stage is optional for exactly that reason. But if you have a
spreadsheet with a group label or a score sitting next to your text, you are two
questions away from an answer, and this page is about which question to ask.

Read as far as **"That is the whole core"** and you have everything you need to
run one. The two parts after it are for when you come back with something
harder, and you can ignore them until you do. The collapsed boxes are the same deal:
the practical answer is always outside them, and what is inside is the evidence
if you want to argue with it. For how any of this actually works underneath —
the folds, the shrinkage, the exact tests — see [Statistics](stats.md).

---

## First, a word about word counts

This is the one piece of housekeeping worth understanding before any of the
analyses, because getting it wrong quietly corrupts everything downstream.

Most language measures are **percentages**. "How positive is this text" means
"what share of its words are positive words". That works beautifully on a
paragraph and falls apart on a sentence, for a reason that is easiest to see in
an example I have used for years:

> the sentence "That was a good donut" contains positive emotion words
> (*good*) and ingestion words (*donut*) both at the same rate: 1 out of 5,
> or 20%. This is an extremely high number for both categories: positive
> emotion words are typically in the 2–8% range, and ingestion words typically
> occur far less frequently (less than 1% of words in most cases).

Five words, and the text now looks like the most food-obsessed, most cheerful
document in your corpus. It is not more positive than anything. It is just
short. In larger bodies of text these behaviors smooth out into distributions
that mean something; in very short ones, a single word is a landslide.

So a **minimum of 25 or 50 words per text** is the usual recommendation, and
Taters offers that filter first, before any of the others, because it is nearly
always what you want. Ticking it adds the step that counts the words and then
uses the count as a *gate* — texts under your threshold drop out of the
analyses. It deliberately does **not** become a predictor: wanting to exclude
short texts is not the same as believing length predicts your outcome.

The same logic applies to the Meaning Extraction Method and topic models
generally. If your texts average ten words and half of those are function
words, there is not enough co-occurrence in the corpus to find themes in,
however sophisticated the statistics you point at it.

* Boyd, R. L. (2017). Psychological text analysis in the digital humanities. In
  S. Hai-Jew (Ed.), *Data Analytics in Digital Humanities* (pp. 161–189).
  Springer International Publishing.
  [https://doi.org/10.1007/978-3-319-54499-1_7](https://doi.org/10.1007/978-3-319-54499-1_7)

---

## "Do these groups talk differently?"

**What it is.** For every feature you extracted, Taters compares the groups you
name — a condition, a diagnosis, a subreddit, moderators against regular users
— and reports whether the difference is bigger than you would expect from
noise, and how big it is.

**Why you would run it.** This is the workhorse of experimental and
quasi-experimental design, and the most interpretable thing in this whole
stage. The output is a sentence you can put in a paper: *people in the
expressive-writing condition used more cognitive-processing words than
controls*. Nothing here is a black box; you can look at the group means and see
the effect with your own eyes.

**What it tells you.** One row per feature, with each group's *n*, mean and
standard deviation, an *F* and its *p*, an effect size (η²), and — this is the
part people skip — which specific pairs of groups differ, in a companion table
with mean differences, confidence intervals and Cohen's *d*. A significant *F*
across four groups tells you *something* differs somewhere. The pairwise table
tells you what.

**What to do with the results.** Sort by η², not by *p*. With a few hundred
features and a decent sample, plenty of things reach significance while
explaining almost none of the variance, and the effect size is what tells you
whether a difference is worth a paragraph in your discussion section.

Two knobs are worth knowing about. If your groups have very different spreads —
common when one group is much smaller — Welch's version of the test does not
assume they are equal, and its post-hoc partner (Games-Howell) does not either.
And if you have something you want held constant, the test becomes an ANCOVA:
see [controls](#holding-something-constant), below. That combination — a
coefficient per language feature, controls entered as covariates, and an
explicit correction because you are exploring many features at once — is
exactly what Schwartz et al. (2013) do, if you want a published precedent for
the bookkeeping.

---

## "Does the way people talk track this outcome?"

**What it is.** A correlation between every language feature and every outcome
you name.

**Why you would run it.** Because it is the fastest honest way to find out
whether there is anything in your data at all. Before you fit a model, before
you write an introduction, you want to know whether the language moves with the
thing you care about. Correlations are also the natural output when your
outcome is *continuous* — a scale score, an age, a rating — where "groups" would
mean throwing away most of your information by chopping a perfectly good
variable into halves.

**What it tells you.** One row per feature, and for each outcome a correlation,
its *p*, its corrected *p*, and the *n* it was computed on. The *n* column
matters more than it looks: features are computed per text, and a text missing
one measure still has the others, so different features can rest on different
numbers of observations.

**What to do with the results.** Read the corrected column, and read the
direction. A wall of significant correlations at *r* = .05 in a sample of
50,000 is a description of a very large sample, not a discovery.

It also helps to know the magnitudes to expect before you look. Individual
words rarely exceed *r* = .10 with a psychological outcome; aggregated
features — topics, dictionary categories — reach roughly *r* = .25 at the
individual level, and the reason is reliability rather than magic: more
reliable measures earn larger effects (Kern et al., 2016). If you want a
concrete answer to "how many participants do I need", Eichstaedt et al. (2021)
report roughly 550 users to recover 100 significant topics for openness and
about 1,800 for neuroticism — which is the most useful sample-size guidance I
know of in this literature.

Which brings us to:

---

### Testing hundreds of things at once

If you extract 200 features and test each at *p* < .05, then even if nothing
whatsoever is going on, you should expect about **ten significant results**.
They will look exactly like findings. Some of them will be interesting. All of
them will be noise.

This is not a hypothetical risk in language research, it is the default
condition — a dictionary run alone produces a hundred-odd columns, a cohesion
run 166, an embedding run several hundred. So Taters corrects for it, and makes
you choose how:

* **Benjamini–Hochberg (`fdr_bh`)** — the sensible default. Controls the
  *proportion* of your findings that are false, rather than the chance of any
  false finding at all, which is the right trade-off when you are exploring.
  Be precise about what it promises: *q* < .05 means a tolerated share of
  false positives **among your discoveries**, not a guarantee about the family
  as a whole. The original proof also assumes independent tests
  (Benjamini & Hochberg, 1995).
* **Benjamini–Yekutieli (`fdr_by`)** — the same idea, made legitimate when
  your tests are *not* independent, which is the actual situation with
  language features: it holds under positive dependency and offers a more
  conservative variant when you cannot assume even that
  (Benjamini & Yekutieli, 2001). Given how heavily intercorrelated language
  features are, this is the conservative-but-defensible choice.
* **Holm** and **Bonferroni** — control the chance of *any* false positive.
  Appropriate when you have a small confirmatory set and a reviewer to
  convince.
* **None** — a legitimate answer for a genuinely pre-registered, single-feature
  hypothesis. It is offered because "no correction" is a methodological
  position, not a mistake, and hiding it would only invite people to
  under-report what they did.

Whichever you pick is recorded in the report, because the correction *is* part
of the result.

---

## "Can I predict this outcome from language alone?"

**What it is.** Cross-validated ridge regression: a model that takes all your
language features together and predicts a number — and, crucially, a model you
can save and point at a *different* dataset later.

**Why you would run it.** Because "does this correlate?" and "can this predict?"
are different questions with different answers, and only the second one gets
you a usable instrument. A correlation tells you about the sample in front of
you. A cross-validated model tells you how well language would do on people it
has never seen — which is the claim you need if you want to *measure* something
rather than describe it.

??? info "Why ridge, and not ordinary regression"

    This is the tradition the World Well-Being Project built, and it is worth
    understanding why it looks the way it does. Language features are many, they
    are heavily collinear (three readability indices are three views of sentence
    length), and there are usually more of them than you have participants —
    literally so with transformer features, where "the number of observations is
    often smaller than the standard 768+ hidden state sizes"
    (Ganesan et al., 2021). Ordinary regression is at its worst in exactly those
    conditions: when predictors are not independent of each other, least-squares
    estimates "have a high probability of being unsatisfactory, if not incorrect"
    (Hoerl & Kennard, 1970). Ridge adds a penalty that shrinks the coefficients,
    deliberately accepting a little bias to buy a large reduction in error, and
    never has to invert a singular matrix.

Which features you hand it is your call, and the two families are worth
telling apart. Dictionary counts, readability, cohesion and the like are
*closed-vocabulary*: someone decided in advance what to count. The
**document-term matrix** — one column per frequent word or phrase, one row per
text — is the *open-vocabulary* alternative Schwartz et al. (2013) made the
case for: let the model find which words carry the signal rather than
deciding beforehand. It is wide (thousands of columns), which is exactly the
regime ridge exists for, and it is the table you would reduce to components
first if you want something you can name afterwards. Tick it on the feature
checklist like anything else; it joins the analysis table under its own name.

??? info "What this can be claimed to measure — two studies worth reading"

    The canonical demonstration of what you get for that is Park et al. (2015):
    they fitted a ridge model to the Facebook language of 66,732 people, applied
    it to 4,824 people it had never seen, and then treated its output as a
    *measurement instrument* — it agreed with self-reports at an average *r* = .38,
    matched or beat what a friend's ratings achieve (.39 against .32), told traits
    apart, and stayed stable over six months. That is the shape of the thing
    Taters' save-and-apply step is for. One thing to note about it, because it
    matters for what you can claim: those 4,824 were a held-out split of the
    *same* collection, so it shows generalization to new **people from the same
    population**.

    The strongest demonstration I know of the harder version — a model fixed in
    advance and then applied to a sample collected afterwards — is
    Kjell et al. (2026). They built a language assessment of PTSD severity on a
    development sample of 1,437 people describing their lives in automated
    interviews, **preregistered the models**, and then applied them unchanged to a
    prospective sample of 346. The preregistered models correlated with
    established PTSD measures at *r* = .38, reached AUC = .76 against a baseline
    of .61, and each standard-deviation increase in the language score was
    associated with $696.50 more mental-health-care expenditure.

    That paper is worth reading for the *method* as much as the result, because it
    answers an objection this whole section invites. If a model's score depends on
    choices you made after seeing the data — which features, which penalty, which
    fold seed — then a good out-of-fold number is not quite the guarantee it looks
    like. Preregistering the model closes that door: the instrument is fixed
    before the new sample exists, and what happens next is a test rather than a
    search. If you intend to publish a language-based measure, this is the
    standard to aim at.

---

### What does each feature table add?

Throw four feature tables at a ridge and you get one R². Which table did the
work? When the tables are analyzed together, Taters also fits each one
alone, every pair, every triple, and so on up to all of them, and the report
lines the R²s up: dictionary alone, dictionary with the matrix, dictionary
with the matrix and the themes. Reading down that table is how you learn
that the readability scores added nothing once the dictionary was in, or
that the topic-model themes carried most of it. Every combination gets its
own saved model too, so the one that earned its keep can be taken to the
next study. Turn it down to each-and-all, or off, with `set_combos` if the
run is large; it is fitted for up to five tables as it is.

---

### When the outcome is a category

If the thing you want to predict is a *class* — a diagnosis, a condition,
`male`/`female` — that is classification, not regression, and Taters treats
handing one to the other as a category error rather than a near miss: each step
refuses the other's outcomes by name and points you at its sibling. It is its
own row on the analysis checklist ("Classification model"), and it asks for a
column of labels rather than a column of numbers. One run can do both — predict
a test score *and* a diagnosis — because each keeps its own answer.

The machinery is the same idea (penalized logistic regression, cross-validated),
but what comes back is deliberately more than one number, and for a reason worth
internalizing: **accuracy alone is close to meaningless**. On an outcome where
90% of people are in one class, a model that always answers "that one" is 90%
accurate and completely worthless. So accuracy is never reported without the
accuracy of exactly that do-nothing model beside it, and the number to lead
with is the **area under the ROC curve**, which asks whether the model *ranks*
cases correctly and is unmoved by how imbalanced your classes are.

You also get precision, recall and F1 **per class**, because a model can be
excellent on the majority class and useless on the one you actually care about,
and every average hides that. And you get the confusion matrix, which is the
only output that tells you *what* the mistakes were.

??? info "A worked example: predicting depression from Facebook posts"

    For a worked example of exactly this — a ridge-penalized logistic regression,
    evaluated by 10-fold cross-validation, scored by AUC, on a real clinical
    outcome — see Eichstaedt et al. (2018). Using only the Facebook posts that 683
    emergency-department patients wrote *before* any depression diagnosis appeared
    in their medical records, it identified the 114 who would go on to be
    diagnosed at AUC = 0.69, which the authors note "falls just short of the
    customary threshold for good discrimination", and roughly matches how well
    screening questionnaires do against the same records. It still beat chance
    three months before the diagnosis was recorded (AUC = 0.62). Read it as a
    proof-of-concept complement to screening at about questionnaire accuracy —
    not as a deployed clinical tool, and not as a save-and-score-a-new-dataset
    example, since all of its results are out-of-fold within one sample.

---

## What you actually end up with

Every analysis writes tidy CSVs *and* contributes to a single `report.md` in
plain English — what was run, on how many rows, with what corrections, and what
it found. The report is written to be readable by a collaborator who was not
there when you ran it, which in practice means yourself in six months.

Every feature table also gets a table of descriptive statistics — n, missing,
mean, SD, quartiles, range, skewness, kurtosis — under `stats_descriptives/`,
whether or not you ran any statistics; it is the methods-table paragraph
written for you, and the first place to look when a measure behaves oddly.

Every result is also drawn as a word cloud, under `figures/wordclouds/`, and
the report ends with them: the features predicting a higher and a lower
score, correlated positively and negatively with each outcome, higher in one
group than another, loading on each component. Size and shade follow the
statistic, blue is positive and red negative. They are the quickest first
read of a result there is; the tables are what you cite. The
[statistics guide](stats.md#word-clouds) lists exactly what is drawn from
what.

Two things in it are worth reading before the results:

**How much of your data was used.** Every table carries the number of rows the
analysis actually used beside the number it *could* have used, and the report
says so out loud whenever those differ by much. This matters more than it
sounds: some measures are undefined for some texts rather than merely absent,
and the textbook response — drop any row with a missing value — can quietly
throw away most of a corpus. A model fitted on 52 of 938 texts will report an
R² as confidently as any other. Taters tells you.

**What was set aside.** A predictor that is missing for most of your texts is
dropped, rather than being allowed to drop the texts. The report names what
went and why.

---

!!! success "That is the whole core"

    Three questions, the answers, and the files they leave behind. If one of
    those was your question, you are done — go and read your report.

    What follows is for when you come back with a harder one: controlling for
    something, reducing hundreds of measures to a few, or taking a model you
    fitted here and pointing it at a different corpus. None of it is needed to
    run an analysis, and the last part in particular is where studies go wrong.

---

# When you need more than the three questions

## Holding something constant

Sooner or later someone asks the obvious question: *isn't that just age?*

They are usually right to ask. Older people write differently from younger
people; men and women write differently on average; longer texts behave
differently from shorter ones. If your groups differ on any of those, a
language difference between them might be a language difference about *that*
instead of about your variable of interest. Controlling for it is how you find
out.

In Taters this is one answer to one question, applied to every analysis you
picked, so "controlling for age" cannot mean three different things in three
tables:

* **Group differences** become an **ANCOVA**. The *F* becomes a comparison of
  nested models, and the group means are reported *adjusted* — what each
  group's score would be if every group sat at the same average age — beside
  the raw means.
* **Correlations** become **partial correlations**: the relationship that
  remains once the controls are accounted for on both sides.
* **Prediction** fits each outcome three ways on one common sample — the
  controls alone, the language alone, and both — and reports what the language
  added *over* the controls. That last number is the one that answers "isn't
  that just age?" directly, rather than by argument.

A control does not have to be a number. Gender, condition, site, cohort — any
label works, dummy coded against its alphabetically first level, which the
report names for you because every coefficient and every adjusted mean is
relative to it.

One trap deserves naming, because it is easy to walk into and impossible to see
afterwards. A **column of numbers can still be a set of categories.** If your
gender variable is coded 1 and 2 and you hold it constant as a *measurement*,
you have asserted that the distance from one gender to the other is worth 0.4
of something — a sentence about nothing. Taters asks which of your numeric
controls are really labels rather than guessing, and treats columns of words as
categories either way.

And a caution about enthusiasm: every control you add costs you sample (any row
missing a control is dropped) and takes something out of the variance you are
trying to explain. Control for what a skeptical reader would ask about. Do not
control for everything you happened to collect.

---

## Analyzing components instead of measures

Every analysis can be run on **components** rather than on the raw measures,
and it is off unless you ask for it.

It is set **per analysis**, and within an analysis **per feature set**, because
one answer is usually wrong for something. Raw variables read better in a
correlation table, where every row is a measure you can name; a ridge over four
hundred collinear measures is exactly what components are for. And a hundred
dictionary categories are worth reducing while eight readability indices are
not — they differ so much in number and in kind that reducing both, or
neither, is the wrong answer for one of them.

So `pca` takes three kinds of answer:

| answer | what happens |
|---|---|
| `off` *(default)* | analyze the measures as they are |
| `all` | reduce every feature set |
| `["dictionary"]` | reduce those sets, leave the rest as raw measures |

You set it once for the run and then change it on any individual analysis from
the options screen — the same "answer once, override where it differs" that
every other setting uses. `pca_components` is `0` to let a rule decide or a
number you choose; `pca_retain` is the rule — `parallel` (parallel analysis,
the default: a component is kept while its eigenvalue beats what random data
of the same size would produce at that rank) or `kaiser` (every eigenvalue
above 1, which on a wide table is most of them). Either way, look at the
eigenvalues. `pca_rotation` is varimax, on by default, and is what makes a
component nameable rather than merely mathematical.

Each analysis writes its own `*_pca_loadings.csv` and `*_pca_eigenvalues.csv`
beside its results. When more than one feature set is reduced, each set gets
its own loadings table (`ridge_pca_loadings__dictionary.csv`,
`ridge_pca_loadings__cohesion.csv`), with columns named exactly as the
components are named in the results, because each set is reduced on its own
and a component never mixes sets — one shared table under a common
`Component_1…` header read as though it had. Read the loadings **before** any result that mentions a
component — the number is meaningless without the table saying what it is made
of. They are per-analysis on purpose: two analyses reducing different feature
sets have genuinely different components under the same names, so one shared
loadings table would be a trap.

Two things to be clear about. Reducing changes what your results are *about* —
from "which of these 400 measures relates to the outcome" to "which of these 12
dimensions does" — which is a decision about your study, and why it is a toggle
rather than a default. And the components are fitted over every row, including
rows a prediction model later holds out; because a PCA never looks at the
outcome this shares the shape of the features and nothing about what is being
predicted, so a cross-validated score from components is very slightly
optimistic rather than meaningfully so.

Missing data is handled by the same two rules the prediction steps use, in the
same order. Your row filters apply first, so the components are fitted on the
rows the analysis is actually about. Then any feature missing for more than half
of those rows is set aside and named in the report — a measure defined for a
tenth of your corpus cannot describe a direction the other nine-tenths lie
along. That half is the threshold you can change: `pca_max_missing` on a
correlation table or a group comparison, and `max_missing` on a prediction,
where the one answer governs both which features a model may be fitted on and
which a component may be built from. What is left is fitted on complete rows only; a row still
missing a feature gets no component scores and drops out, and the report says how
many did.

That last part is stricter than what a correlation table does with the same
table, and deliberately so. A correlation is one number per pair of columns and
can use whatever rows that pair happens to have. A PCA decomposes the whole
matrix at once, and a matrix stitched together from a different set of rows in
every cell can turn out not to be positive semi-definite — when that happens the
decomposition hands back components explaining *negative* variance, which is
nonsense that looks exactly like output. Filling the blanks with column means
instead, which this used to do, is worse than it sounds: it pulls every
incomplete row toward the center on that feature, which is precisely the spread
a PCA is there to measure.

---

## Finding structure instead of testing it

Sometimes you do not have an outcome yet. You have 400 correlated columns and
the sense that they are really measuring a dozen things.

That is what dimensionality reduction is for, and Taters implements it as
principal component analysis with varimax rotation — the same machinery behind
the **Meaning Extraction Method**, just pointed at any feature table rather
than a document-term matrix. PCA finds the directions along which your features
actually vary together; varimax rotation then makes those directions
*readable*, so each one loads strongly on a small cluster of features and
weakly on everything else, which is what lets you look at a loading table and
give a component a name.

The decisions that matter here are the classic factor-analysis ones and they
are worth making deliberately: how many components to keep (eigenvalue rules
are a starting point, not an answer), and whether to rotate (for interpretation,
almost always yes). The smallest number of components you can coherently
interpret is usually the best one — it is not unusual to see studies extract
hundreds of topics, and it is not unusual for 95% of those to be
uninterpretable grab-bags.

See the [text guide](analyzing-text.md#topic-modeling-the-meaning-extraction-method-mem)
for the MEM specifically, and the [statistics guide](stats.md) for the
mechanics. If you would rather read the MEM as four plain steps than as a
paper, [liwc.app/help/mem](https://www.liwc.app/help/mem) sets it out that
way.

* Chung, C. K., & Pennebaker, J. W. (2008). Revealing dimensions of thinking in
  open-ended self-descriptions: An automated meaning extraction method for
  natural language. *Journal of Research in Personality, 42*(1), 96–132.
  [https://doi.org/10.1016/j.jrp.2007.04.006](https://doi.org/10.1016/j.jrp.2007.04.006)

---

# Taking a model somewhere else

A fitted model is a file, and a file can be pointed at any corpus you like.
Whether the numbers that come back still *mean* anything is a separate
question, and it is the one this part is about.

## Taking a model to a different kind of text

Crossing platforms *is* done in this literature, and done well. Giorgi et al.
(2022) took the personality models trained on roughly 66,000 Facebook
participants — the Park et al. models above — and applied them to 6,064,267
Twitter users, aggregated to 2,041 counties. Across 13 outcomes the resulting
county-level estimates "replicated patterns that have been observed in
individual-level and geographic studies", including higher Republican vote
share in less agreeable counties. So a saved model genuinely can travel.

But note *how* they did it, because this is the part that matters for you. The
models were **not applied unchanged.** To adjust for the differences between
the Facebook source and the Twitter target, they applied a domain-adaptation
step (Target Side Domain Adaptation, as cited there), correcting both for
geographically-specific word usage and for the different word distributions of
the two platforms. Someone doing this carefully treated the platform shift as
a problem requiring an explicit correction — not as something that comes out
in the wash.

**Taters has no domain-adaptation step.** Applying a saved model to a
different kind of text here is the naive version of what that paper did
carefully, so the burden of showing the scores still mean something is yours.

And keep the size of the hop in view. Facebook to Twitter is a *short* one:
both are short, informal, public posts written to a loosely-known audience.
Twitter to political speeches, or to private diary entries, is a much bigger
shift — different register, different audience, different purpose, wildly
different length — and a model has no way to tell you it has left the
territory where it was validated. It will return numbers. The numbers will be
arithmetically correct. Whether they still measure the construct is an
empirical question about *your* corpus, and the honest thing is to treat a
cross-domain score as a hypothesis to be validated against something external
rather than as a measurement you can report.

**What it tells you.** Every headline number is **out-of-fold**: each text is
predicted by a model that never saw it. That is the whole discipline of this
section, and it is worth being precise about why. Overfitting is "the tendency
for statistical models to mistakenly fit sample-specific noise as if it were
signal", and an in-sample R² does not estimate how your *fitted equation* will
do on new data — it estimates the average performance of the model *form*
across hypothetical samples, which is "virtually always an overly optimistic
estimate" of the thing you actually care about. K-fold cross-validation is "a
minimally biased way of estimating the true generalization performance of any
model" (Yarkoni & Westfall, 2017).

So the in-sample R² sits in the next column, and the gap between the two is the
difference between a model that learned something and a model that memorized
your participants. A cross-validated R² can be *negative*, and when it is, it
is telling you something clear and useful: this model is worse than just
guessing the mean.

You also get the coefficients, one row per predictor and one column per
outcome, sorted so the features that did something are at the top of the file.

**What to do with the results.** Two things, and they pull in opposite
directions.

The first is to look at the coefficients and interpret them — and to be careful
here, because a ridge coefficient on one of 200 collinear features is not "the
effect of that feature". Use them to see the shape of the thing, and lean on
the correlations for claims about individual features.

The second is to *keep the model*. A fit is an instrument, not a result. The
saved model carries the predictor names, the training means and standard
deviations, the chosen penalty and the fold seed, so it can score a new corpus
standardized against the original sample — which is what makes two datasets'
scores comparable at all. That fit-once, apply-many discipline is how a finding
becomes a measure.

Which raises the honest warning, and it is one I have made in print before: being
able to **predict** something is conceptually quite different from
**understanding** it. A model that reports "these 200 features together predict
depression at R² = .12" is a real result and it is also, by itself, not an
insight about depression. The most actionable findings in this field have
usually come from transparent models that a domain expert could reason about,
not from the most powerful algorithm available. Taters gives you the predictive
machinery because you often need it; it puts the interpretable analyses first
because that is usually where the understanding comes from.

**Expect small numbers.** This is the single most useful thing to know before
you look at your first result. Eichstaedt et al. (2021) put five feature sets
through the same 10-fold cross-validated pipeline on 65,896 people's Facebook
posts and reported what each actually achieved: DICTION *r* = .23, LIWC2015
*r* = .28, General Inquirer *r* = .29, and 2,000 LDA topics *r* = .37. So a
well-built cross-validated language model of a personality-like outcome lands
somewhere around *r* = .2 to .4 — **not** .8. A model claiming *r* = .9 on new
data is far more likely to have a leak than a discovery.

If you are coming from experimental psychology those numbers look
disappointing. They are not, and there are two separate reasons why. An *r* of
.20 is a medium effect by the field's own calibrated benchmarks, and
practically useful; *r* ≥ .40 in psychology "is likely to be a gross
overestimate" (Funder & Ozer, 2019 — whose benchmarks assume, as they are
careful to say, that your estimates are reliable in the first place).
Separately: complex outcomes have many causes each contributing a little, which
makes small effects the ones "most likely to be real", and they "can have
substantial consequences, especially when considered at scale and over time"
(Götz et al., 2022).

---

## Scoring with a model that was built differently

A saved model was fitted on features measured with particular settings, and
those settings are part of the instrument. Score it against features measured
some other way and you get a number that looks exactly like a prediction and
is not one — the column names match, the values do not. In a real run that
moved a mean predicted age from 36.5 to 44.0 years and reported 904
predictions without a word of complaint.

So Taters records how every feature table was measured, in a small
`<name>_settings.json` beside it, and a saved model carries the same record
for the features it was fitted on. When you score, the two are compared:

- **They match** — it scores, silently. Nothing to say.
- **They differ, and both are known** — refused, naming every setting and both
  values. Not overridable: the model was fitted on other numbers, and there is
  nothing to interpret.
- **One side has no record** — refused, but waivable one table at a time
  (`unverified_ok=("cohesion",)`). "I cannot check this" is a different claim
  from "these disagree", and only the first is yours to wave through.

You may well want it both ways at once: your own cohesion features measured
how *you* want them, and a colleague's model applied to the same corpus even
though it was fitted differently. That works. Taters measures the feature a
second time using the model's own settings, into a private folder of its own,
and scores the model against that — leaving your table untouched and out of
it. The review screen says so before anything runs:

```text
Parts of speech                             you picked
Parts of speech — for age_blogs [ridge]     the model needs it
Score with models I already have            you picked
```

If your settings already happen to match the model's, nothing extra runs and
both share the one extraction. And a model that carries its own word lists —
which is how they are saved — can do this on a machine that has never seen
them. That portability is also why exporting a model warns you what is inside
it: a word list can be licensed, or can hold material you would rather not
publish.

Three kinds of feature table depend on the corpus itself rather than only on
settings, and each is handled so a model fitted on one can still meet a new
study:

- **A document-term matrix** has one column per vocabulary term, and the
  vocabulary came from *your* texts. The model carries that vocabulary, and
  the new study is scanned against it — so a word the new texts never use is
  a column of zeros, as it should be, rather than a missing predictor. Your
  own matrix, built from the new study's vocabulary, is left alone.
- **Topic-model themes** were fitted to your texts; refitting them on another
  study gives different themes, or a different number of them. The model
  carries the fitted theme model and *applies* it to the new texts, which is
  the only honest way to score the same themes twice.
- **Parts of speech** (and any other table whose columns are whatever
  occurred) may lack a tag no text in the new study happens to use. That
  absence is the measurement: the column is scored as zero.

A model fitted before the settings record changed shape is refused with
that reason — "recorded under different versions of the settings record;
re-fit the model" — rather than as a settings mismatch. The two records
hash their word lists differently and cannot be compared setting by setting,
and telling you a dictionary had changed when nobody touched it sent one
user looking for a change that never happened.

A model fitted with **controls** — `age` and `gender` held constant, say —
also needs those two columns to score anything, and no feature step produces
them: they are the spreadsheet's own. When a chosen model needs controls,
the run carries them from the spreadsheet you are scoring (the same
metadata step the statistics use), and the wizard checks the spreadsheet
actually has those columns before anything runs. A folder of documents has
no such columns, so a controlled model cannot score one.

Some differences cannot be fixed this way, and are refused rather than
guessed at. A model fitted on acoustic or transcript-embedding features needs
a per-file chain that cannot be re-run inside one pass. And a difference in
how the *text* was assembled — which columns were read, how several were
joined — changes every number while the measuring settings agree; a model
records that its text was prepared differently without recording how, so that
one needs your judgment rather than an automatic replay.

The thing not to expect from any of it: this checks that features were
*measured* the same way, not that they mean the same thing. Identical settings
over a `large-v3` transcript and a `base.en` one give different numbers and
this deliberately says nothing, because the transcript is the text, and a
model exists in order to meet new text.

---

### Several models at once

Tick as many saved models as you like on the "Score with models I already
have" row -- a ridge, two classifiers, your word vectors, a fine-tuned
predictor. Each is scored on its own, against the feature tables *it* was
fitted on, into `features/model_scores/<model name>.csv` with its own
unscored-row accounting; and everything is merged into
`features/model_scores.csv`, an outer join on `text_id` in which every score
column carries the model's name in front -- `openness_ridge__pred_openness`,
`condition_clf__p_condition_A` -- the way several dictionaries' categories
carry the dictionary's name, so two models predicting the same outcome
never collide. With a single model the table is exactly what it always was,
plain names and no subfolder. Every model's provenance gate runs before any
scoring, so a run with five models refuses up front, naming every model
with a problem, rather than failing on the fourth after scoring three; and
two models whose names read the same are refused too (rename one under
Settings → Manage Taters data → Manage saved models), because the name is the file name, the
column prefix and the private-table folder.

---

### Classifiers from Hugging Face

A model you did not train works the same way. Thousands of finished text
classifiers and regressors are published on the Hugging Face hub -- sentiment,
emotion, stance, toxicity, a rating predicted from text -- and if you have
used one elsewhere it is probably already in your Hugging Face cache. Under
Settings → Manage Taters data → Manage saved models, *Import a classifier from Hugging Face* lists
the text classifiers in that cache with their labels, or takes a checkpoint
folder, or a hub name to download on first use. It asks two things: what the
model predicts (the stem of its columns, so `sentiment` rather than the
model's file name) and what to call it. A folder is copied into your library
so the model travels with it; a hub name is loaded from the model cache.

From then on it is a saved model like any other: tick it on "Score with
models I already have", alone or with a ridge and a fine-tuned predictor, and
it writes `pred_sentiment`, `prob_sentiment` and one `p_sentiment_<class>`
column per class (one `pred_` column for a regression head), with the
checkpoint's own labels -- renamable per model in Settings. A multi-label
head (an emotion model where a text can be both *joy* and *surprise*) writes
one `p_<outcome>_<label>` column per label and, in `pred_`, every label whose
probability reaches the model's threshold (0.5 unless you change it under
"Change how the ticked model is applied"), joined with `|`. Long texts are
read in windows and averaged, as the fine-tuned predictor reads them. Text
classifiers, multi-label heads and regression heads import; a bare encoder is
something to adapt or fine-tune under "Train a model", and an audio or image
classifier is refused by name until Taters reads those.

---

## Where to go next

- [Statistics guide](stats.md) — the same analyses from the mechanical side:
  every output file, every setting, and the numerical details
- [Analyzing text](analyzing-text.md) — where the features come from
- [The Taters app](wizard.md) — how to get all of this without writing code
- [Pipelines](pipelines.md) — the file the app writes, if you would rather
  script it

---

## References

The analyses themselves are standard; what is specific to *language* data is
mostly a matter of scale (hundreds of correlated features), of text length, and
of the discipline of out-of-sample validation. These are the sources I would
point a newcomer at.

**Start here**

* Kern, M. L., Park, G., Eichstaedt, J. C., Schwartz, H. A., Sap, M., Smith,
  L. K., & Ungar, L. H. (2016). Gaining insights from social media language:
  Methodologies and challenges. *Psychological Methods, 21*(4), 507–525.
  [https://doi.org/10.1037/met0000091](https://doi.org/10.1037/met0000091)
  — written precisely because so little guidance existed for psychologists
  entering this area. Where to start, what language data can support, how big
  the effects will realistically be, and which traps to avoid.

* Kennedy, B., Ashokkumar, A., Boyd, R. L., & Dehghani, M. (2022). Text
  analysis for Psychology: Methods, principles, and practices. In M. Dehghani &
  R. L. Boyd (Eds.), *The handbook of language analysis in psychology*
  (pp. 3–62). The Guilford Press.

* Schwartz, H. A., & Ungar, L. H. (2015). Data-driven content analysis of
  social media: A systematic overview of automated methods. *The ANNALS of the
  American Academy of Political and Social Science, 659*(1), 78–94.
  [https://doi.org/10.1177/0002716215569197](https://doi.org/10.1177/0002716215569197)
  — a short, accessible map of the methods landscape, written for social
  scientists.

* Boyd, R. L., & Schwartz, H. A. (2021). Natural language analysis and the
  psychology of verbal behavior: The past, present, and future states of the
  field. *Journal of Language and Social Psychology, 40*(1), 21–41.
  [https://doi.org/10.1177/0261927X20967028](https://doi.org/10.1177/0261927X20967028)

**Text length, preparation, and the predict-versus-understand tradeoff**

* Boyd, R. L. (2017). Psychological text analysis in the digital humanities. In
  S. Hai-Jew (Ed.), *Data Analytics in Digital Humanities* (pp. 161–189).
  Springer International Publishing.
  [https://doi.org/10.1007/978-3-319-54499-1_7](https://doi.org/10.1007/978-3-319-54499-1_7)

**Language carries psychological signal — the founding demonstrations**

* Schwartz, H. A., Eichstaedt, J. C., Kern, M. L., Dziurzynski, L., Ramones,
  S. M., Agrawal, M., Shah, A., Kosinski, M., Stillwell, D., Seligman,
  M. E. P., & Ungar, L. H. (2013). Personality, gender, and age in the language
  of social media: The open-vocabulary approach. *PLOS ONE, 8*(9), e73791.
  [https://doi.org/10.1371/journal.pone.0073791](https://doi.org/10.1371/journal.pone.0073791)
  — 700 million words from 75,000 people who had also taken standard
  personality tests. Open access, and the best first paper to hand a reader.

* Eichstaedt, J. C., Schwartz, H. A., Kern, M. L., Park, G., Labarthe, D. R.,
  Merchant, R. M., Jha, S., Agrawal, M., Dziurzynski, L. A., Sap, M., Weeg, C.,
  Larson, E. E., Ungar, L. H., & Seligman, M. E. P. (2015). Psychological
  language on Twitter predicts county-level heart disease mortality.
  *Psychological Science, 26*(2), 159–169.
  [https://doi.org/10.1177/0956797614557867](https://doi.org/10.1177/0956797614557867)

* Brown, N. J. L., & Coyne, J. C. (2018). Does Twitter language reliably
  predict heart disease? A commentary on Eichstaedt et al. (2015a). *PeerJ, 6*,
  e5656. [https://doi.org/10.7717/peerj.5656](https://doi.org/10.7717/peerj.5656)
  — read this one *with* the paper above. Rerunning the original analysis and
  swapping in suicide as the outcome reversed the associations. It is a
  one-paper education in reading an aggregate-level predictive result
  skeptically, and it is about that specific county-level analysis rather than
  about predictive language modeling in general.

**Prediction, cross-validation, and what score to expect**

* Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation
  for nonorthogonal problems. *Technometrics, 12*(1), 55–67.
  [https://doi.org/10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634)
  — where the penalty comes from. (Note for reference managers: Hoerl and
  Kennard published *two* ridge papers back to back in the same issue. The
  companion, "Applications to nonorthogonal problems", is 12(1), 69–82.)

* Yarkoni, T., & Westfall, J. (2017). Choosing prediction over explanation in
  psychology: Lessons from machine learning. *Perspectives on Psychological
  Science, 12*(6), 1100–1122.
  [https://doi.org/10.1177/1745691617693393](https://doi.org/10.1177/1745691617693393)
  — why "how well does my model fit?" is the wrong question. About psychology
  broadly, not about text analysis specifically.

* Park, G., Schwartz, H. A., Eichstaedt, J. C., Kern, M. L., Kosinski, M.,
  Stillwell, D. J., Ungar, L. H., & Seligman, M. E. P. (2015). Automatic
  personality assessment through social media language. *Journal of Personality
  and Social Psychology, 108*(6), 934–952.
  [https://doi.org/10.1037/pspp0000020](https://doi.org/10.1037/pspp0000020)
  — the fit-once, then-score-other-people workflow, treated as a measurement
  instrument rather than a result.

* Kjell, O., Ganesan, A. V., Boyd, R. L., Oltmanns, J., Rivero, A., Feltman,
  S., Carr, M. A., Alves, J., Luft, B., Kotov, R., & Schwartz, H. A. (2026).
  Replicability and validity of a new artificial-intelligence assessment of
  posttraumatic stress disorder from patient language: A sequential evaluation
  with model preregistration. *Clinical Psychological Science*. Advance online
  publication. [https://doi.org/10.1177/21677026261439026](https://doi.org/10.1177/21677026261439026)
  — a preregistered model, applied unchanged to a prospectively collected
  sample. The standard to aim at if you intend to publish a language-based
  measure.

* Giorgi, S., Nguyen, K. L., Eichstaedt, J. C., Kern, M. L., Yaden, D. B.,
  Kosinski, M., Seligman, M. E. P., Ungar, L. H., Schwartz, H. A., & Park, G.
  (2022). Regional personality assessment through social media language.
  *Journal of Personality, 90*(3), 405–425.
  [https://doi.org/10.1111/jopy.12674](https://doi.org/10.1111/jopy.12674)
  — Facebook-trained models carried across to Twitter and aggregated to
  counties, with an explicit domain-adaptation step in between. The precedent
  for taking a model somewhere new, and for how much care that takes.

* Eichstaedt, J. C., Kern, M. L., Yaden, D. B., Schwartz, H. A., Giorgi, S.,
  Park, G., Hagan, C. A., Tobolsky, V. A., Smith, L. K., Buffone, A., Iwry, J.,
  Seligman, M. E. P., & Ungar, L. H. (2021). Closed- and open-vocabulary
  approaches to text analysis: A review, quantitative comparison, and
  recommendations. *Psychological Methods, 26*(4), 398–427.
  [https://doi.org/10.1037/met0000349](https://doi.org/10.1037/met0000349)
  — five feature sets through one cross-validated pipeline, so you can see what
  each actually achieves (Table 2, p. 418). The best answer to "what score
  should I be happy with?"

* Ganesan, A. V., Matero, M., Ravula, A. R., Vu, H., & Schwartz, H. A. (2021).
  Empirical evaluation of pre-trained transformers for human-level NLP: The
  role of sample size and dimensionality. In *Proceedings of the 2021
  Conference of the North American Chapter of the Association for Computational
  Linguistics: Human Language Technologies* (pp. 4515–4532). Association for
  Computational Linguistics.
  [https://doi.org/10.18653/v1/2021.naacl-main.357](https://doi.org/10.18653/v1/2021.naacl-main.357)

**Classification, worked through on a clinical outcome**

* Eichstaedt, J. C., Smith, R. J., Merchant, R. M., Ungar, L. H., Crutchley,
  P., Preoţiuc-Pietro, D., Asch, D. A., & Schwartz, H. A. (2018). Facebook
  language predicts depression in medical records. *Proceedings of the National
  Academy of Sciences, 115*(44), 11203–11208.
  [https://doi.org/10.1073/pnas.1802331115](https://doi.org/10.1073/pnas.1802331115)

**Reading small effects without despairing**

* Funder, D. C., & Ozer, D. J. (2019). Evaluating effect size in psychological
  research: Sense and nonsense. *Advances in Methods and Practices in
  Psychological Science, 2*(2), 156–168.
  [https://doi.org/10.1177/2515245919847202](https://doi.org/10.1177/2515245919847202)

* Götz, F. M., Gosling, S. D., & Rentfrow, P. J. (2022). Small effects: The
  indispensable foundation for a cumulative psychological science.
  *Perspectives on Psychological Science, 17*(1), 205–215.
  [https://doi.org/10.1177/1745691620984483](https://doi.org/10.1177/1745691620984483)
  (and, for the other side of it, Primbs et al., 2023,
  [https://doi.org/10.1177/17456916221100420](https://doi.org/10.1177/17456916221100420))

**Multiple comparisons**

* Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A
  practical and powerful approach to multiple testing. *Journal of the Royal
  Statistical Society: Series B (Methodological), 57*(1), 289–300.
  [https://doi.org/10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)

* Benjamini, Y., & Yekutieli, D. (2001). The control of the false discovery
  rate in multiple testing under dependency. *The Annals of Statistics, 29*(4),
  1165–1188. [https://doi.org/10.1214/aos/1013699998](https://doi.org/10.1214/aos/1013699998)

**Dimensionality reduction and the Meaning Extraction Method**

* Chung, C. K., & Pennebaker, J. W. (2008). Revealing dimensions of thinking in
  open-ended self-descriptions: An automated meaning extraction method for
  natural language. *Journal of Research in Personality, 42*(1), 96–132.
  [https://doi.org/10.1016/j.jrp.2007.04.006](https://doi.org/10.1016/j.jrp.2007.04.006)
