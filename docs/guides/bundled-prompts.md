# Built-in prompts

Taters ships with a small library of **prompts** for the "Generate responses
with an LLM" step. Like the built-in dictionaries, you do not have to download
or import them: the first time Taters needs your library it copies them into
`~/.taters/library/prompts/`, and from then on they appear wherever the step
asks you which prompts to run, and under **Settings and tools → Manage Taters
data → Manage language model prompts**.

They are ordinary `.csv` files (one `role,content` row per turn), so you can
open one, see exactly what the model is told, copy it, change a definition,
and import your own version alongside the original. Upgrading Taters keeps
them current the same way it does the dictionaries: a release that adds or
corrects a prompt brings it to your library, but a built-in you deleted stays
deleted and a copy you edited is never overwritten.

!!! warning "These are not validated measures (yet)"

    The theory-grounded prompts below were written for Taters. Each one is
    built on a published theory and, where there is one, its standard
    measure, and the definitions follow those sources closely. None of them
    has been validated against self-report the way a few published prompts
    have (Peters & Matz, 2024; Wright et al., 2026; Hur et al., 2024). Before
    you treat a score as a measurement, check its convergent and
    discriminant validity on your own data, and report which prompt you
    used. The run's report quotes every prompt verbatim, which takes care of
    the second part.

!!! note "Cite what you use"

    Every prompt here stands on somebody's theory, and usually on years of
    measurement work. If a score ends up in a paper, cite the sources listed
    with it alongside Taters itself.

## Running them well

All of the theory-grounded prompts ask the model for a JSON object, so turn
on **`llm_parse_json`** in the step's settings: each key in the reply becomes
a column, named after the prompt (the `Moral foundations (MFQ-2)` prompt's
`care` score lands in `Moral_foundations_MFQ_2__care`). Where a prompt asks
for an object inside the object, like the Big Five's
`{"extraversion": {"evidence": ..., "score": ...}}`, each inner value gets a
column of its own: `Big_Five_BFI_2__extraversion_score` and
`Big_Five_BFI_2__extraversion_evidence`. Lists (the agency and communion
themes) are kept as JSON text in one cell.

The rest follows the practice in Rathje et al. (2024) and Wright et al.
(2026), and most of it is what Taters does anyway:

* **Temperature 0**, which is the default. Same model, same settings, same
  reply.
* **One text per conversation.** Every text is asked about from scratch; the
  only earlier turns the model sees are the prompt's own worked examples.
* **Check the parse.** A reply that does not parse is a row with blank
  columns and `parse_ok` 0, and the report says how many replies each key
  appeared in. Taters does not re-ask the model when a reply fails to parse
  or a score is out of range, so look at those rows before you analyze.
* **Scores about people need many texts.** A prompt that rates the author
  (the Big Five, values) is most trustworthy averaged over many texts from
  the same person, and better still over repeated runs and several models:
  Wright et al. (2026) found the average across models agreed with
  self-report best of all. Taters' feature averaging handles the first part;
  running the same pipeline with a second model handles the third.

## Theory-grounded prompts

### Big Five (BFI-2)

*Rates the author. 1–5 per domain, with a short note on the evidence for
each.* Columns: `extraversion_score`, `agreeableness_score`,
`conscientiousness_score`, `negative_emotionality_score` and
`open_mindedness_score`, each with its `_evidence` note.

The five-factor model, using the domain definitions of the BFI-2:
Extraversion, Agreeableness, Conscientiousness, Negative Emotionality and
Open-Mindedness. The model is told to rate typical, enduring tendencies
rather than the mood of the moment, to give 3 when the text says nothing
about a domain, and never to lean on assumptions about age, gender or group
membership. This is a holistic rating of each domain. The best-validated
approach so far is Wright et al.'s item-level method instead: one inventory
item per call, answered as the author would answer it, then scored with the
inventory's own key. Taters does not do that one (yet).

* Soto, C. J., & John, O. P. (2017). The next Big Five Inventory (BFI-2): Developing and assessing a hierarchical model with 15 facets to enhance bandwidth, fidelity, and predictive power. *Journal of Personality and Social Psychology, 113*(1), 117–143. [https://doi.org/10.1037/pspp0000096](https://doi.org/10.1037/pspp0000096)
* Peters, H., & Matz, S. C. (2024). Large language models can infer psychological dispositions of social media users. *PNAS Nexus, 3*(6), pgae231. [https://doi.org/10.1093/pnasnexus/pgae231](https://doi.org/10.1093/pnasnexus/pgae231)
* Wright, A. G. C., Ringwald, W. R., Vize, C. E., & Eichstaedt, J. C. (2026). Assessing personality using zero-shot generative AI scoring of brief open-ended text. *Nature Human Behaviour, 10*(3), 541–555. [https://doi.org/10.1038/s41562-025-02389-x](https://doi.org/10.1038/s41562-025-02389-x)

### Moral foundations (MFQ-2)

*Rates each text. 0–3 per foundation, with three worked examples.* Columns:
`care`, `equality`, `proportionality`, `loyalty`, `authority`, `purity`, and
a `rationale`.

Moral Foundations Theory as revised for the MFQ-2, which splits Fairness into
Equality and Proportionality for six foundations in all. Following how moral
language is annotated by hand (as in the Moral Foundations Reddit Corpus), a
foundation counts only when the author makes a moral evaluation that rests on
it: approval, condemnation, obligation or outrage. A topic that merely comes
up scores 0, which is the line keyword-based moral dictionaries cannot draw.

* Graham, J., Haidt, J., Koleva, S., Motyl, M., Iyer, R., Wojcik, S. P., & Ditto, P. H. (2013). Moral foundations theory: The pragmatic validity of moral pluralism. *Advances in Experimental Social Psychology, 47*, 55–130. [https://doi.org/10.1016/B978-0-12-407236-7.00002-4](https://doi.org/10.1016/B978-0-12-407236-7.00002-4)
* Atari, M., Haidt, J., Graham, J., Koleva, S., Stevens, S. T., & Dehghani, M. (2023). Morality beyond the WEIRD: How the nomological network of morality varies across cultures. *Journal of Personality and Social Psychology, 125*(5), 1157–1188. [https://doi.org/10.1037/pspp0000470](https://doi.org/10.1037/pspp0000470)
* Trager, J., et al. (2022). *The Moral Foundations Reddit Corpus* (arXiv:2208.05545). arXiv. [https://doi.org/10.48550/arXiv.2208.05545](https://doi.org/10.48550/arXiv.2208.05545)

### Schwartz basic values

*Rates the author; works on single texts but means most averaged over a
person's texts. 0–3 per value.* Columns: `self_direction`, `stimulation`,
`hedonism`, `achievement`, `power`, `security`, `conformity`, `tradition`,
`benevolence`, `universalism`, and a `rationale`.

Schwartz's ten motivationally distinct values, arranged in a circle where
neighbors are compatible and opposites conflict (self-direction against
conformity, power against universalism). Values show in what a person
pursues, praises, defends, or criticizes others for lacking, so the model is
told to rate what the author values rather than what they describe, and not
to give opposing values both high scores without clear evidence for each. If
you follow standard Schwartz scoring, center each person's ten scores on
their own mean (ipsatize) before comparing people.

* Schwartz, S. H. (1992). Universals in the content and structure of values: Theoretical advances and empirical tests in 20 countries. *Advances in Experimental Social Psychology, 25*, 1–65. [https://doi.org/10.1016/S0065-2601(08)60281-6](https://doi.org/10.1016/S0065-2601(08)60281-6)
* Schwartz, S. H., et al. (2012). Refining the theory of basic individual values. *Journal of Personality and Social Psychology, 103*(4), 663–688. [https://doi.org/10.1037/a0029393](https://doi.org/10.1037/a0029393)

### Agency and communion

*Rates each text; written for personal narratives. 0–3 for each, plus the
themes found, with two worked examples.* Columns: `agency`, `communion`,
`agency_themes` and `communion_themes` (JSON lists), and a `rationale`.

Bakan's duality of agency (self-assertion, mastery, separation) and communion
(connection, intimacy, union), coded with the themes of McAdams et al.'s
(1996) system for autobiographical scenes: four agentic themes (self-mastery,
status/victory, achievement/responsibility, empowerment) and four communal
ones (love/friendship, dialogue, caring/help, unity/togetherness). The two are
independent, so a story can be high on both. The same two dimensions run
through the dual perspective model of social cognition. If you want the
word-counting version instead, the built-in
[Big Two dictionary](bundled-dictionaries.md#big-two-agency-and-communion)
measures the same pair.

* Bakan, D. (1966). *The duality of human existence: Isolation and communion in Western man*. Rand McNally.
* McAdams, D. P., Hoffman, B. J., Mansfield, E. D., & Day, R. (1996). Themes of agency and communion in significant autobiographical scenes. *Journal of Personality, 64*(2), 339–377. [https://doi.org/10.1111/j.1467-6494.1996.tb00514.x](https://doi.org/10.1111/j.1467-6494.1996.tb00514.x)
* Abele, A. E., & Wojciszke, B. (2014). Communal and agentic content in social cognition: A dual perspective model. *Advances in Experimental Social Psychology, 50*, 195–255. [https://doi.org/10.1016/B978-0-12-800284-1.00004-7](https://doi.org/10.1016/B978-0-12-800284-1.00004-7)

### Depressive symptoms (PHQ-9)

*Rates each text (or all of one person's texts joined into one). 0–3 per
symptom, with whether the evidence was stated or inferred, and a short
quote.* Columns: for each of `anhedonia`, `depressed_mood`, `sleep`,
`fatigue`, `appetite`, `worthlessness_guilt`, `concentration`, `psychomotor`
and `suicidal_ideation`, a `_score`, an `_evidence` (explicit, inferred or
none) and an `_excerpt`.

The nine PHQ-9 items, which map onto the DSM criteria for a major depressive
episode, scored one by one on the PHQ-9's own frequency anchors. The model
scores what the author states first, then infers the rest cautiously, and
says which is which, following Ganesan et al.'s analysis of how GPT models
read depression in text. The prompt tells the model not to add up a total:
add the nine `_score` columns yourself for the 0–27 total, and use the usual
5 / 10 / 15 / 20 cutoffs for the severity bands.

This is a research scoring aid, never a diagnosis. Route any
`suicidal_ideation_score` above 0 to a person under your IRB's safety
protocol. And do not read a 0 there as evidence of absence: Ganesan et al.
found GPT-4 reluctant to infer suicidality unless it was stated outright.

* Kroenke, K., Spitzer, R. L., & Williams, J. B. W. (2001). The PHQ-9: Validity of a brief depression severity measure. *Journal of General Internal Medicine, 16*(9), 606–613. [https://doi.org/10.1046/j.1525-1497.2001.016009606.x](https://doi.org/10.1046/j.1525-1497.2001.016009606.x)
* Ganesan, A. V., Varadarajan, V., et al. (2024). *Explaining GPTs' schema of depression: A machine behavior analysis* (arXiv:2411.13800). arXiv. [https://doi.org/10.48550/arXiv.2411.13800](https://doi.org/10.48550/arXiv.2411.13800)

### Core affect

*Rates each text. Positive 0–10 and negative 0–10, rated separately, and
arousal 1–9.* Columns: `positive`, `negative`, `arousal`, and a `rationale`.

Russell's core affect, where feelings vary along valence and arousal, with
one change that matters a great deal: valence is rated as two separate
scales, positive and negative, following the bivariate model of evaluative
space, so mixed feelings score high on both instead of averaging out to a
"neutral" that is anything but. Arousal uses the anchors of the
Self-Assessment Manikin. The model rates the feeling the *author* expresses,
not whether the topic is good or bad, which is where generic sentiment
prompts most often go wrong: a flat report of a tragedy is low on negative
feeling, and a cheerful account of a hard day is high on positive. For a
single valence score, subtract negative from positive, as Hur et al. (2024)
did.

* Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology, 39*(6), 1161–1178. [https://doi.org/10.1037/h0077714](https://doi.org/10.1037/h0077714)
* Russell, J. A. (2003). Core affect and the psychological construction of emotion. *Psychological Review, 110*(1), 145–172. [https://doi.org/10.1037/0033-295X.110.1.145](https://doi.org/10.1037/0033-295X.110.1.145)
* Cacioppo, J. T., & Berntson, G. G. (1994). Relationship between attitudes and evaluative space: A critical review, with emphasis on the separability of positive and negative substrates. *Psychological Bulletin, 115*(3), 401–423. [https://doi.org/10.1037/0033-2909.115.3.401](https://doi.org/10.1037/0033-2909.115.3.401)
* Bradley, M. M., & Lang, P. J. (1994). Measuring emotion: The Self-Assessment Manikin and the semantic differential. *Journal of Behavior Therapy and Experimental Psychiatry, 25*(1), 49–59. [https://doi.org/10.1016/0005-7916(94)90063-9](https://doi.org/10.1016/0005-7916(94)90063-9)
* Hur, J. K., Heffner, J., Feng, G. W., Joormann, J., & Rutledge, R. B. (2024). Language sentiment predicts changes in depressive symptoms. *Proceedings of the National Academy of Sciences, 121*(39), e2321321121. [https://doi.org/10.1073/pnas.2321321121](https://doi.org/10.1073/pnas.2321321121)

### Discrete emotions (appraisal)

*Rates each text. 0–3 per emotion; more than one can be present. Three worked
examples.* Columns: `anger`, `fear`, `sadness`, `disgust`, `joy`, `surprise`,
and a `rationale`.

Ekman's six basic emotions, but each one is defined by its core relational
theme in Lazarus's sense: the way the author sees their situation that gives
rise to it, rather than the words they use for it. That is what lets the
model tell anger (a wrong, and someone to blame) from sadness (a loss that
cannot be undone) when the vocabulary overlaps. Ratings are from the writer's
perspective, not the reader's. If your research needs the self-conscious
emotions (pride, shame, guilt), copy the prompt and add them with Lazarus's
themes.

* Ekman, P. (1992). An argument for basic emotions. *Cognition and Emotion, 6*(3–4), 169–200. [https://doi.org/10.1080/02699939208411068](https://doi.org/10.1080/02699939208411068)
* Lazarus, R. S. (1991). *Emotion and adaptation*. Oxford University Press.
* Rathje, S., Mirea, D.-M., Sucholutsky, I., Marjieh, R., Robertson, C. E., & Van Bavel, J. J. (2024). GPT is an effective tool for multilingual psychological text analysis. *Proceedings of the National Academy of Sciences, 121*(34), e2308950121. [https://doi.org/10.1073/pnas.2308950121](https://doi.org/10.1073/pnas.2308950121)

## Simple starting points

Three plainer prompts ship as well. They are there to show the format and to
get a first run going, not as measures.

### Sentiment 1-7

*Rates each text from 1 (very negative) to 7 (very positive), and asks for
the number only.* With `llm_parse_json` on, the bare number becomes the
`Sentiment_1_7__value` column; off, it is the reply. For a measure of
feeling, [Core affect](#core-affect) is the better choice.

### Emotion with reason

*The dominant emotion (joy, anger, sadness, fear, or none), its intensity
from 1 to 5, and a one-sentence reason, as JSON, with two worked examples.*
[Discrete emotions (appraisal)](#discrete-emotions-appraisal) is the
theory-grounded version.

### One-sentence summary

*A plain one-sentence summary of each text, in the writer's own terms.* Free
text, nothing to parse.
