# Statistics

Feature extraction answers "what is in this language?". Statistics answer
"so what?" — do these groups differ, does this measure track that outcome,
can we predict it. Taters can run that last part for you, at the tail of the
same pipeline that produced the features, so the numbers in your results
table come from a run you can hand to a colleague and have them reproduce.


> **New to this?** Start with [Running Analyses](running-analyses.md), which covers why you would run each of these, what it tells you, and what to do with the results. This page is the mechanical companion to it: every output file, every setting, and the numerical details.

## Descriptive statistics, for every run

Before any test, the distributions. Every run that writes a feature table
also writes a table of descriptive statistics for it under
`stats_descriptives/` — `readability.csv` for the readability table, and so
on — whether or not you asked for any statistics. For every numeric measure:
how many texts have it (`n`) and how many do not (`missing`), the mean and
standard deviation (n − 1), the minimum, quartiles and maximum, skewness and
excess kurtosis (the bias-adjusted sample statistics SPSS and R's
`psych::describe(type = 2)` report, zero for a normal distribution), how
many values are exactly zero, and how many distinct values there are.
Identifier columns are left out; count columns such as `token_count` are
described like any other, because they are the first thing to look at when a
measure looks strange. A `README.md` beside the tables says what each column
is. The tables are redone only when the feature table is newer than they are.

**In Taters:** `python -m taters.stats.describe --feature_csvs
features/readability.csv --out_dir stats_descriptives`, or
`Taters().stats.describe_features`, or the step every composed pipeline
already has.

## Group differences, correlations, and the analysis table

Say you have 100,000 rows of text with a group label (Group A, B, C) and
some Big Five scores. Pick your features in the wizard as usual, and it
then asks one more optional question: run any statistics on the results? Say
yes, point it at the column holding your groups or your outcomes, and the
run ends with a `stats_results/` folder — tidy CSVs plus a plain-English
`report.md` narrating what was found.

Underneath, three things happen. First an **analysis table** is assembled:
every feature table you chose, joined to the metadata columns from your
spreadsheet, one row per text. The join is deliberately strict — only rows
present in every table survive — and every row lost along the way is
counted in `assemble_manifest.json` and in the report, because "we analyzed
99,881 of your 100,000 rows, and here is where the rest went" is the first
question a reviewer asks. Keeping that merged table is optional: it is the
file to open in R or hand to a colleague, but on a wide feature set it is the
largest thing the run produces, so there is a setting for it.

What counts as a feature table is anything with one row per text: dictionary
counts, readability, cohesion, lexical richness, parts of speech, archetype
similarity, sentence embeddings, topic scores, scores from a saved model, and
the **document-term matrix** — one column per frequent word or phrase, which
is the classic open-vocabulary design when you want to predict an outcome
from what people actually said (reduce it to components first, or let the
ridge penalty do the work). The n-gram *frequency list* is the one output
that does not join: it is a table of the corpus, not of each text, and the
wizard marks it so when you are picking features for statistics. A table
pulled in only as another step's input — the matrix behind a topic model —
runs but stays out of the analysis unless you tick it yourself.

**What counts as a feature.** Every step writes a few numbers beside its
measures that are not measures: the token count of a document-term matrix
or topic model, the word count beside a dictionary's categories, a
readability step's raw sentence, character and syllable counts. Each step
declares those, and the statistics keep them *in* the analysis table — there
to filter on and to look at — but *out* of the feature sets, so a prediction
model is not quietly learning from text length. The report's first section
lists exactly what was kept aside. One shared setting, `bookkeeping`, puts
them back in (`features`) for a design where length is a predictor.

You can also **filter rows** before anything runs: the usual one is a minimum
word count, because a three-word answer makes most language measures
meaningless. Word count is the first thing offered, and choosing it adds the
step that counts — you do not have to have picked something that measures
length, or to know what that step happens to call its count. You can filter
on the run's own measures, on your spreadsheet's own columns, or both; tick
as many variables as you like and then set each one's condition in turn. A
column of labels is filtered by ticking the values to keep (`gender in
[Female, Male]`); a column of numbers by an operator and a value. A
row with a blank in the filtered column is dropped too — an unknown word
count is not evidence of a long text — and each filter's toll is reported.

The wizard also vets the columns you pick for the statistics against the
whole spreadsheet before anything runs — a class too thin to classify, a
group of one, an outcome with words in it, an id that repeats, a control
level of two — and either asks again or offers the values worth keeping,
which become a filter of the same kind. The
[wizard guide](wizard.md) lists the checks.

Then the analyses themselves. **Group differences** runs a one-way ANOVA per
feature with Tukey-Kramer post-hoc tests, effect sizes (eta-squared, and
Cohen's *d* per pair), and per-group descriptives, all in one table you can
scan down. If your groups' variances should not be pooled, ask for Welch's
version instead and the post-hocs switch to Games-Howell with it — Tukey's
statistic divides by a pooled error term, so keeping it after choosing Welch
would be incoherent. **Correlations** puts features in the rows and your
outcomes in the columns, with *r*, *p*, an FDR-adjusted *p*, and the
pairwise *N* for every cell, so a coefficient computed on 40 of 100,000
rows cannot masquerade as one computed on all of them.

### Correcting for multiple comparisons

Testing 160 cohesion measures at *p* < .05 without correction hands you eight
"findings" by chance alone. But *which* correction to apply is a
methodological decision, not something a tool should quietly make for you, so
Taters asks — and records the answer in the report, where a methods section
can be written from it. There are two independent families to correct, and
you choose in both.

**Across features.** Every analysis that produces a *p* per feature offers:

| Choice | Controls | When |
|---|---|---|
| `fdr_bh` (default) | False discovery rate | The usual exploratory case: you accept a known proportion of false positives among your hits. |
| `fdr_by` | False discovery rate under *any* dependence | Language features are heavily inter-correlated in ways nobody has characterized; BH assumes positive dependence, BY assumes nothing and is stricter for it. |
| `holm` | Family-wise error | You want the classical guarantee, and Holm gives it with more power than Bonferroni, always. |
| `bonferroni` | Family-wise error | Simplest to explain, strictest, and what some reviewers expect. |
| `none` | Nothing | A single planned comparison, or you intend to correct yourself downstream. |

Asking for `none` writes no adjusted column at all rather than one that
repeats the raw *p* under an "adjusted" heading, and the report says plainly
that nothing was corrected.

**Between groups (post-hoc).** The pairwise tests carry their own correction,
and it is chosen separately because it answers a different question. The
default (`auto`) pairs Tukey-Kramer with a pooled ANOVA and Games-Howell with
Welch's, which is the coherent pairing; you can force either one, or ask for
Bonferroni-corrected pairwise *t*-tests, or uncorrected ones. If you take the
uncorrected route the output says so in its `method` column — six pairs at
.05 is a ~26% chance of at least one false positive, so it should be a
choice you made on purpose. Whatever you pick, the confidence intervals are
corrected to match the tests, so "excludes zero" and "*p* < alpha" never
disagree.

Missing data is never quietly zeroed either: group differences drop a row only
from the features it lacks, correlations use each pair's own complete rows,
and every table shows the *N* behind every number.

One rendering note, because reviewers ask: a *p* of `0` in these tables means
the tail underflowed double precision (about *p* < 1e-300), not that the
probability is zero. The columns stay numeric so R and pandas read them as
numbers; write it up the way you always would, as *p* < .001.

* Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B (Methodological), 57*(1), 289–300. [https://doi.org/10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)

* Benjamini, Y., & Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. *The Annals of Statistics, 29*(4), 1165–1188.

  
* Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics, 6*(2), 65–70.

  
* Games, P. A., & Howell, J. F. (1976). Pairwise multiple comparison procedures with unequal N’s and/or variances: A Monte Carlo study. *Journal of Educational Statistics, 1*(2), 113–125. [https://doi.org/10.3102/10769986001002113](https://doi.org/10.3102/10769986001002113)

* Welch, B. L. (1951). On the comparison of several mean values: An alternative approach. *Biometrika, 38*(3/4), 330–336. [https://doi.org/10.2307/2332579](https://doi.org/10.2307/2332579)

* Rubin, M. (2021). When to adjust alpha during multiple testing: A consideration of disjunction, conjunction, and individual testing. *Synthese, 199*(3), 10969–11000. [https://doi.org/10.1007/s11229-021-03276-4](https://doi.org/10.1007/s11229-021-03276-4)



If you asked for each text column to be measured separately — a survey with
two open-ended questions, say — then each column is *analyzed* separately
too, and every results table gains a `source_col` saying which question it
describes. That is not a limitation but a correctness requirement: a
participant contributes one row per question, so pooling them would count
one person's two answers as two independent observations and inflate every
degree of freedom in sight.

One other choice is worth making deliberately: whether to analyze all your
features **together** or **one table at a time**. Together is the default
and usually right for group differences. One-at-a-time is how you answer
"which of these feature sets actually carries the signal?" — the same
analysis repeated per table, results side by side in one file.

A caveat about combining rows. If you asked the wizard to combine rows (one
row per participant rather than per response), a numeric outcome cannot ride
along untouched — a participant with twelve responses has twelve scores. It
becomes the participant's average, named `openness_mean` (with an
`openness_n` counting what went into it), and that is the name the analyses
use. The wizard handles the renaming for you; if you are writing the YAML
by hand, that is the trap to know about.

A note on column names. Every measure Taters ships picks column names that
no other measure uses, and there is a build test that refuses any two that
could agree — so `flesch_reading_ease` means the same thing in every results
table you will ever produce, whatever else was in the run. Where a clash is
genuinely unavoidable, it gets resolved rather than ignored: if two tables
both carry a column called `word_count`, or one of your own spreadsheet
columns has the same name as a measure, both copies are renamed
`<table>__<column>` — `readability__word_count` and `cohesion__word_count` —
and the run's record lists every rename it made. Both copies, not just the
second one, so the answer does not depend on which file was read first.

## Prediction (cross-validated ridge)

The third question is the ambitious one: can the language *predict* the
outcome? Taters fits ridge regression, which is the right default here for a
structural reason — language features are many and heavily collinear (three
readability indices are three views of sentence length), and that is exactly
where ordinary least squares is least stable. Ridge trades a little bias for
a large drop in variance and never has to invert a singular matrix.

What ridge cannot do is choose its own penalty, so Taters searches a grid of
them by k-fold cross-validation and reports performance at the chosen one.
Every headline number is **out-of-fold**: each row is predicted by a model
that never saw it. The in-sample R² sits in the next column, and the gap
between the two is the whole story of whether a model learned something or
memorized the training set. Standardizing the predictors is on by default,
which makes coefficients comparable to each other (each is per standard
deviation); turn it off and they stay in the features' own units, where a
coefficient on `char_count` and one on `ttr` are not remotely comparable.

This is also where "one table at a time" earns its keep: fit per feature set
and `ridge_cv_metrics.csv` becomes a ranking — one row per feature set per
outcome, sortable by `cv_r2`, answering "does cohesion beat readability for
this outcome?" directly rather than by eye.

A fit is an instrument, not a result. The model file records the predictor
names, the training means and standard deviations, the chosen penalty and
the grid it came from, the coefficients, and the fold seed — so it can score
a *different* dataset later, standardized against the original training
sample, which is what makes the two sets of scores comparable at all. Import
one into your library and it appears on the **feature checklist** as "Score
with models I already have" — alongside the other measures, because what
comes back is a column per text, and the next thing anyone wants to do with
it is analyze it like any other feature.

One entry covers every kind of saved model — a ridge, a classifier, a MEM
topic model, word vectors, a fine-tuned transformer — because "score this
dataset with what I already have" is one thing to want, and each file says
which kind it is. The library listing tells you which: `age_blogs [ridge] ·
adds pred_age_blogs · needs cohesion`. Tick as many as you like: each is
scored on its own and the results are merged into one `model_scores` table,
with every column prefixed by its model's name whenever there is more than
one (`age_blogs__pred_age`), so two models predicting the same outcome can
sit side by side in the same analysis. A model that is missing a predictor
your new data lacks refuses by name rather than filling in a guess, and names
the feature step you need to add — and it does so before any model is
scored, so a run with five models fails up front or not at all. The details
are under [Several models at once](running-analyses.md#several-models-at-once).

Name your models when you import them. Every outcome gets a model file of
its own -- predict five personality traits and you get five files, each named
for its feature set and its outcome (`ridge__all__openness.json`, or
`classifier__all__condition.json`), each importable and nameable on its own.
Every such file writes a column called `pred_<outcome>`, so two models of the
same outcome scored in one run collide — and one scored against a dataset
that already has an `age` column is worse than a collision, because both
readings are plausible and nothing reports it. The import screen asks for a name and for
the output column names, and both travel inside the model file.

* Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics, 12*(1), 55–67. [https://doi.org/10.2307/1267351](https://doi.org/10.2307/1267351)

* Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning: Data mining, inference, and prediction (2nd edition)*. Springer-Verlag. [https://www.springer.com/gp/book/9780387848570](https://www.springer.com/gp/book/9780387848570)



### Comparing the feature tables

When the tables are analyzed together, the prediction steps do not stop at
one model with everything in it. They fit each table alone, every
combination of tables, and all of them together — for four tables that is
fifteen models per outcome — and the report ranks them in one table, so
"is the matrix worth having once the dictionary is in?" is a comparison of
two rows. Combinations are named by their members (`dictionary+readability`)
in `ridge_cv_metrics.csv`, in the coefficient table and in `models/`, and
the full combination keeps the name `all`. Beside the set's name the metrics
carry `n_feature_sets`, how many tables it is made of, and `n_features`, how
many feature columns they brought, so a score can be read against how much
went into it (`n_predictors` is what the fitted model actually used,
controls included). The setting is `set_combos`:
`subsets` (the default, every combination, for up to five tables — six
would be sixty-three models, so above five each table and all together are
fitted and the report says so), `each_and_all`, or `none` for the single
combined model. With a reduction on, each table is reduced on its own and a
combination is fitted on its members' components; its model carries every
reduction it needs, so it can still be applied to a new corpus. Word clouds
are drawn for the single tables and for `all`, not for every combination.

## When a predictor is missing more often than it is present

This one is worth reading before you trust any prediction result, because
the failure looks exactly like success.

Some measures are undefined for some texts, not merely absent. The cohesion
step writes `adjacent_overlap_2_*_para` columns, which compare paragraphs two
apart — so they are blank for any text with fewer than three paragraphs, and
most texts have one or two. Listwise deletion, the textbook default, then
removes the whole *row* for want of one column. In a real run 60 of 166
predictors were blank for 885 of 938 texts, every one of those rows was
dropped, a ridge was fitted on **52 rows with 165 predictors**, and it
reported an R² like any other.

Taters drops the column instead, which is the right way round: a predictor
available for 6% of your corpus cannot tell you anything about the other
94%, whereas the rows it would destroy carry every other feature intact.
The same run then fits on 904 rows with 105 predictors. The threshold is
`max_missing` (default `0.5` — a predictor missing for more than half your
texts is not a predictor), what was set aside is always printed, and every
metrics table carries `n_used` beside `n_available` so the question "how much
of my data went into this?" has an answer in the same row as the answer you
came for. Set `max_missing=1.0` to keep every column and accept the row loss;
the report then says what it cost.

Two warnings appear in `report.md` under **About the sample**: when a model
used less than 80% of the rows that had the outcome, and when it had at least
as many predictors as rows. Neither stops the run — they are your call — but
neither is silent any more.

There is a middle case, and it bit a real run. A predictor blank in *some*
rows — fewer than half — is kept, and every row lacking it is unscorable:
here, and in any study the model is later applied to. A lexical measure
that is undefined under 42 tokens was kept on a study of long texts and then
left half of a shorter second study unscored. So the report has a third
section, **Predictors with gaps**, naming each such predictor and how often
it was blank, while you can still leave it out of the feature set or raise
the minimum text length.

And when a model *is* scored, whatever went unscored is accounted for beside
the scores: `<scores>_unscored.csv` says how many rows have a blank
prediction, which predictor each lacked and how often, and how many rows
never reached the scorer because a feature table did not have them. The
file exists exactly when something went unscored, so its presence is
itself the signal.

## How the folds are made

Both prediction steps share two settings. `n_folds` is the number of
cross-validation folds (five by default). `stratify`, on by default, makes
the folds alike: the classifier deals each class round the folds so every
fold has the same mix of classes; the ridge orders rows by the outcome and
deals them the same way, so every fold gets its share of low, middle and
high values. Off, the folds are a plain seeded shuffle — which can put most
of the high scores, or all of a small class, in one fold, and the
fold-to-fold scores then say more about the deal than about the model. The
report says which was used; the saved model records it.

## Statistics on a spreadsheet you already have

Everything above assumes Taters measured the language first. Often it did not:
the numbers already exist. LIWC output from the desktop program, questionnaire
scales, ratings a team coded by hand, measures computed in R. **Wrangle/Analyze data → Analyze data**
on the main menu is the same statistics stage with the extraction taken out.

Point it at the spreadsheet, tick which columns are the predictors, and answer
the questions you would have answered anyway: what the outcomes are, which
column separates the groups, what to hold constant, how to correct the
p-values. Only columns that read as numbers are offered as predictors, and a
column ticked as one is not offered again as an outcome, a group, a control or
the row's identifier — a column on both sides of a regression predicts itself
perfectly and means nothing.

What comes out is what comes out of any other run: `analysis_table.csv`, the
result tables, the figures, a model file per outcome, and the plain-English
report. The pipeline is saved as YAML like every other flow, so the same
analysis re-runs on next year's data with one command. Word clouds are off,
since a cloud of column names is not a finding; the row is still on the
options screen if you have joined a document-term matrix and want them.

**Several rows per participant.** If your file has one row per wave, per
session, or per trial, say so when it asks what one row of results should
describe: choose to combine rows and name the column that identifies the
person. The predictors are then averaged within each participant and arrive as
`<column>_mean`, exactly as they would in a grouped extraction run. The counts
that averaging produces are kept aside rather than analyzed — how many rows
somebody contributed is a fact about your data collection, not a predictor.

## Holding something constant

Any of the three analyses can be run with control variables, and "controlling
for age" means the same sample and the same adjustment in all of them, because
it is one answer to one question rather than three separate settings.

- **Group differences** become an **ANCOVA**: the F-test is a comparison of
  nested models, and the per-group means are reported *adjusted* to the
  control means (`adj_mean_<level>`) beside the raw ones. With no controls the
  F is identical to `scipy.stats.f_oneway` and the post-hoc contrasts to
  `scipy.stats.tukey_hsd`, so nothing changes for anyone who wants neither.
- **Correlations** become **partial correlations**, with `df = n − 2 − k`.
- **Prediction** fits each outcome three ways on one common complete-case
  sample — controls alone, language alone, both — and reports
  `delta_r2_over_controls`. That single sample matters: listwise deletion over
  different column sets would give the controls-only model more rows than the
  combined one, and the difference between their scores, which is the whole
  point of running both, would then be part sample and part language.

Categorical controls are dummy coded against their alphabetically first level,
and the report names that reference level, because every coefficient and every
adjusted mean is relative to it and no reader can work that out from the
numbers. A blank cell stays missing rather than becoming "not this level" — a
respondent whose gender is unrecorded is not thereby male.

A column of numbers can still be a set of categories, and this is the case that
bites: a gender coded 1 and 2, held constant as a *measurement*, asserts that
the distance from one gender to the other is worth 0.4 of something. The wizard
asks which of your numeric controls are really labels rather than guessing, and
columns holding words are treated as categorical either way.

## Predicting a category rather than a number

If the thing you want to predict is a class — a diagnosis, a condition,
`male`/`female` — that is classification, not regression, and handing one to
the other is a category error rather than a near miss. Each step refuses the
other's outcomes by name and points at its sibling. The wizard offers it beside
the ridge and asks for the label column separately (`stats_class_cols`), so one
run can predict a score and a diagnosis without either step being handed the
other's column.

Taters fits L2-penalized logistic regression, for the same reason ridge is the
regression default: language features are many and collinear, and the penalty
is what keeps the fit stable when two predictors say almost the same thing.
The penalty is chosen by out-of-fold **log loss** rather than accuracy —
accuracy is a step function of the probabilities, so selecting on it means
choosing between ties. Folds are stratified by class, because a fold holding
none of a small class has no recall and no curve to measure, which quietly
turns five-fold validation into four-fold.

What comes back is deliberately more than one number:

| file | what it answers |
|---|---|
| `classifier_cv_metrics.csv` | accuracy **and** the accuracy of always guessing the commonest class, AUC, macro/weighted F1 |
| `classifier_per_class.csv` | precision, recall, F1 and support per class — where the model is good and where it is useless |
| `classifier_confusion.csv` | what the mistakes actually were |
| `classifier_folds.csv` | each fold's own score |
| `classifier_predictions__<set>.csv` | the saved model's prediction and the held-out one, per row, with the fold |

Accuracy is never reported without that baseline beside it, because on an
outcome where 90% of people are in one class, a model that always answers
"that one" is 90% accurate and worthless. Lead with the AUC: it asks whether
the model *ranks* cases correctly, independent of where you put the threshold,
and is unmoved by class imbalance.

## Reading the coefficient table

`ridge_coefficients.csv` is **wide**: one row per predictor, one column per
outcome, ordered so the predictors with the largest coefficients are at the
top of the file. Comparing what predicted age against what predicted
extraversion is then reading across a row, not reading 165 rows and then
another 165 and holding them side by side.

Feature set and model stay as *columns* rather than becoming more column
blocks, so fitting five feature sets separately adds five times the rows —
which sort and filter — where five times the columns would not. A blank cell
means the predictor was not in that outcome's model at all (dropped as
constant, or set aside as too sparse), which is a different thing from a
coefficient of zero.

## Dimensionality reduction (PCA with varimax rotation)

Most of what Taters produces is *wide*: hundreds of dictionary categories,
cohesion indices, embedding dimensions, acoustic summaries. Wide is good for
capturing everything and bad for almost everything you do next — models
overfit, multiple-comparison corrections eat your power, and nobody can
interpret a regression with four hundred predictors. The statistical tools
in Taters exist for that moment: when you have too many correlated features
and want a small number of meaningful dimensions instead.

Principal component analysis is the old, honest workhorse here: it finds
the directions along which your features actually vary together, so that a
few components can stand in for many columns. Varimax rotation then makes
those components *readable* — each one loads strongly on a small cluster of
features and weakly on the rest, which is what lets you look at a loading
table and give a component a name. This is the same machinery behind the
Meaning Extraction Method in the [text guide](analyzing-text.md), just
pointed at any feature table rather than a document-term matrix: dictionary
scores, cohesion indices, acoustic summaries, embeddings, or any mix you
have gathered into one CSV.

The decisions that matter are the classic ones from the factor-analysis
literature, and they are worth making deliberately rather than by default:
how many components to keep (eigenvalue rules are a starting point, not an
answer — look at the eigenvalues and the interpretability of the result),
and whether to rotate (for interpretation, almost always yes). The papers
below are the standard guides to those choices:

* Fabrigar, L. R., Wegener, D. T., MacCallum, R. C., & Strahan, E. J. (1999). Evaluating the use of exploratory factor analysis in psychological research. *Psychological Methods, 4*(3), 272–299. [https://doi.org/10.1037/1082-989X.4.3.272](https://doi.org/10.1037/1082-989X.4.3.272)

  
* Preacher, K. J., & MacCallum, R. C. (2003). Repairing Tom Swift’s electric factor analysis machine. *Understanding Statistics, 2*(1), 13–43. [https://doi.org/10.1207/S15328031US0201_02](https://doi.org/10.1207/S15328031US0201_02)

  
* Kaiser, H. F. (1958). The varimax criterion for analytic rotation in factor analysis. *Psychometrika, 23*(3), 187–200. [https://doi.org/10.1007/BF02289233](https://doi.org/10.1007/BF02289233)

  
* Horn, J. L. (1965). A rationale and test for the number of factors in factor analysis. *Psychometrika, 30*(2), 179–185. [https://doi.org/10.1007/BF02289447](https://doi.org/10.1007/BF02289447)

  
* Glorfeld, L. W. (1995). An improvement on Horn’s parallel analysis methodology for selecting the correct number of factors to retain. *Educational and Psychological Measurement, 55*(3), 377–393. [https://doi.org/10.1177/0013164495055003002](https://doi.org/10.1177/0013164495055003002)


**How many components.** Left to decide for itself (`pca_components: 0`),
Taters uses **parallel analysis** by default: it draws fifty random data sets
of the same size as yours, takes the 95th percentile of their eigenvalues at
each rank, and keeps a component only while its own eigenvalue beats that —
so a component has to explain more than chance would. The older **Kaiser
rule** (keep every eigenvalue above 1) is there as `pca_retain: kaiser`; on a
wide table it keeps most of the components, because chance alone lifts the
leading eigenvalues of random data well past 1. Parallel analysis is
memory-light: the random data are never held whole, only folded chunk by
chunk into one small matrix, so it runs on an old laptop as readily as the
fit itself. The model file records which rule decided, with the eigenvalues
beside their chance thresholds up to the first that fell short, so the count
can be read and argued with. The MEM topic model asks the same question under
the name `mem_k_selection`, and has two extra answers to it — see
[How many topics?](analyzing-text.md#how-many-topics).

Two properties of the Taters implementation are worth knowing because they
matter scientifically. First, it is **exact and deterministic** — a full
eigendecomposition, not a randomized approximation, with results that
reproduce in R (`psych::principal`) to a couple of decimal places — and it
streams, so a table with millions of rows fits in ordinary memory. Second,
**a fit is a reusable instrument**: it saves a model file carrying the
feature names, the training means and standard deviations, and the
projection, so you can score a *new* dataset on the *same* components later
— standardized against the original sample, which is what makes scores
comparable across datasets. That fit-once, apply-many discipline is the
same one the MEM topic model uses, because they share this engine.

A prediction model fitted on components carries its reduction: the saved
ridge or classifier names the *raw* features as its inputs, rebuilds the
components from them when it scores — standardized against the fitting
sample, the same discipline as above — and so can be applied to a new study
like any other model. Before this, a model whose predictors were
`Component_1` to `Component_5` named columns no step produces, and could not
be applied at all.

Every analysis in this stage can also reduce *its own* features before it
runs, without a separate step: `pca="all"` reduces everything it was given,
`pca=["dictionaries", "cohesion"]` reduces only the sets you name, and `"off"`
(the default) leaves your measures alone. `pca_components` and `pca_rotation`
are the two decisions above; `pca_max_missing` is the threshold for setting a
mostly-empty feature aside (on a prediction the same answer is `max_missing`,
because there it also governs what the model may be fitted on). The loadings
and eigenvalues land beside that analysis's own results, named after it — two
analyses reducing different sets have genuinely different components, so one
shared table would be a trap. The
[analysis guide](running-analyses.md) covers when this is a good idea and what
it costs.

**In Taters:** `python -m taters.stats.pca fit` on any feature CSV;
`... apply` with the saved model on a new one. The rest of this stage is
reachable the same way — `python -m taters.stats.group_differences`,
`... correlations`, `... assemble` — or through `Taters().stats`, or as
ordinary steps in a pipeline file. The wizard just writes that file for
you.

## Word clouds

Every result above is also drawn, as a word cloud, under
`stats_results/figures/wordclouds/<analysis>/<feature set>/` — one folder per
analysis and one per feature set inside it, so a ridge run per table gives
`ridge-regression/dictionary/`, `ridge-regression/topic_model_mem/` and so on
— and the report shows the pictures in a section of its own. They are read the same way everywhere: the
size and shade of a word follow the size of its statistic, blue is a positive
value and red a negative one. A cloud is a way to read a table quickly, not a
result of its own — the CSV it was drawn from is the record.

What gets drawn:

* **Ridge regression** — two clouds per outcome per feature set: the features
  predicting a higher score and those predicting a lower one, sized by their
  standardized coefficient. The coefficients come from one model: language
  fitted beside every control when there were controls, language alone
  otherwise. The controls themselves are not drawn; `age` as the biggest
  word in a cloud about language is true and beside the point.
* **Classification** — the same pair per class: what pushes a text toward the
  class and what pushes it away.
* **Correlations** — features correlated positively and negatively with each
  outcome, sized by *r*, and only those under *p* < .05 (the adjusted *p*
  when the table has one). When nothing passes, the report says so instead
  of showing an empty picture.
* **Group differences** — per pair of groups, what is higher in the first and
  what is higher in the second, sized by Cohen's *d*, filtered the same way.
* **Components** — one cloud per principal component with both signs in it,
  the thirty features loading most strongly, blue loading positively and red
  negatively, in a `components/` folder inside the set they reduce.
* **Themes** — when a feature set is the topic model's themes, the ridge or
  correlation cloud shows `Theme_5` as a word, which means nothing on its
  own. So every theme that appears in that set's clouds is drawn again in a
  `themes/` folder beside them, its legend naming where it mattered most
  (`TIPI_Open: β +0.32`). Every theme, whether or not a model used it, is
  also drawn beside the features under
  `features/figures/wordclouds/topic_model_mem/`.
* **Most frequent terms** — the corpus as a whole, from the frequency list
  (`features/figures/wordclouds/ngram_frequencies/`), and, when the
  statistics have a group column and the document-term matrix is one of the
  feature tables, one cloud per group with the terms that group used most.

The clouds are on by default; the shared `wordclouds` setting turns every one
of them off and `wordcloud_words` caps the words per cloud (80). A cloud is
redrawn only when the table it came from is newer than it is, so a re-run of
an unchanged pipeline redraws nothing.

**In Taters:** `python -m taters.figures.wordclouds stats --stats_dir
stats_results` redraws the statistics clouds from the tables, with
`--max_p`, `--max_words` and `--component_words` to taste; `... themes
--loadings_csv features/topic_model_mem_loadings.csv` and `... frequencies
--freq_csv features/ngram_frequencies.csv` do the same for the text stage.
Or `Taters().figures`, or the steps a pipeline file already has.

