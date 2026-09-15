# Built-in dictionaries and archetypes

Taters ships with a small collection of ready-made **content-coding dictionaries** and
**archetype dictionaries**. You do not have to download or import them: the
first time Taters needs your library it copies them into
`~/.taters/library/`, and from then on they appear wherever a flow asks you
to pick one —

* **Extract features → Dictionary counts (LIWC-style)**, which counts how
  often each category's words appear in your texts;
* **Extract features → Archetype similarity**, which scores each text by how
  close it sits to a set of seed phrases;
* **Train a model → Word vectors: train on these texts**, where a
  content-coding dictionary can be handed in as the **concept dictionary**
  whose categories get expanded through the vector space your own corpus
  produced;
* **Settings and tools → Manage Taters data → Manage dictionaries**, which asks which of the two
  libraries you mean and then lets you rename, export, remove, or add to it.

They are ordinary files, so you can open one, copy it, edit a category, and
import your own version alongside the original.

Upgrading Taters keeps them current: a release that adds a dictionary or
corrects one brings it to your library, however long that library has been in
use. It will not undo anything you decided, though — a built-in you deleted
stays deleted, and a copy you edited stays yours and is never overwritten. If
you want a modified built-in to keep receiving updates, save your version
under a different name and leave the original alone.

!!! note "Cite what you use"

    Every dictionary here is somebody's research and hard work, most of it published and
    validated over several years. If a measure ends up in a paper, a talk or a
    report, *please* cite the source below alongside Taters itself. The wording is
    the authors' own where they supplied it.

Note that this is *not* built in a dynamic fashion. I will try to keep this updated
whenever I add a new dictionary to those built into the application. However,
if you use a built-in dictionary and cannot find the citation for it, please
drop me a line and I'll make sure to get it added here.

*Also*, I've been collecting dictionaries for ages now. If you find any typos/errors
below (or if you have a dictionary that you want to include!), just let me know.

## Content-coding dictionaries

These are LIWC-format dictionaries (`.dicx`), read by the same parser that
reads any dictionary you write yourself. Three of them are very large —
Urban Dictionary Supplements (141,021 terms), Stereotype Content (14,449)
and General Inquirer IV (8,640) — and will take noticeably longer to run
than the rest.

### Absolutist

*1 category, 19 terms.* Measures absolutist thinking (e.g., always, never) in texts.

* Al-Mosaiwi, M., & Johnstone, T. (2018). In an absolute state: Elevated use of absolutist words is a marker specific to anxiety, depression, and suicidal ideation. *Clinical Psychological Science, 6*(4), 529–542. [https://doi.org/10.1177/2167702617747074](https://doi.org/10.1177/2167702617747074)
* Al-Mosaiwi, M., & Johnstone, T. (2018). Linguistic markers of moderate and absolute natural language. *Personality and Individual Differences, 134*, 119–124. [https://doi.org/10.1016/j.paid.2018.06.004](https://doi.org/10.1016/j.paid.2018.06.004)

### Age Stereotypes

*10 categories, 210 terms.* Reflects eight broadly defined stereotypes identified in past research as descriptive of older adults.

* Remedios, J. D., Chasteen, A. L., & Packer, D. J. (2010). Sunny side up: The reliance on positive age stereotypes in descriptions of future older selves. *Self and Identity, 9*(3), 257–275. [https://doi.org/10.1080/15298860903054175](https://doi.org/10.1080/15298860903054175)

### Agitation-Dejection

*2 categories, 48 terms.* Based on studies linking promotion versus prevention focus with the emotions "agitation" and "dejection."

* Johnsen, J. A. K., Vambheim, S. M., Wynn, R., & Wangberg, S. C. (2014). Language of motivation and emotion in an internet support group for smoking cessation: Explorative use of automated content analysis to measure regulatory focus. *Psychology Research and Behavior Management, 7*, 19–29. [https://doi.org/10.2147/PRBM.S54947](https://doi.org/10.2147/PRBM.S54947)

### Behavioral Activation

*8 categories, 1,059 terms.* Captures linguistic indicators of planning and participation in enjoyable activities.

* Burkhardt, H. A., Alexopoulos, G. S., Pullmann, M. D., Hull, T. D., Areán, P. A., & Cohen, T. (2021). Behavioral activation and depression symptomatology: Longitudinal assessment of linguistic indicators in text-based therapy sessions. *Journal of Medical Internet Research, 23*(7), e28244. [https://doi.org/10.2196/28244](https://doi.org/10.2196/28244)

### Big Two - Agency and Communion

*2 categories, 333 terms.* Measures the degree to which a person is thinking in terms of agency and communion.

* Pietraszkiewicz, A., Formanowicz, M., Sendén, M. G., Boyd, R. L., Sikström, S., & Sczesny, S. (2019). The Big Two dictionaries: Capturing agency and communion in natural language. *European Journal of Social Psychology, 49*(5), 871–887. [https://doi.org/10.1002/ejsp.2561](https://doi.org/10.1002/ejsp.2561)

### Body Type

*16 categories, 498 terms.* A content analysis dictionary for automating the scoring of Fisher and Cleveland's (1958) body type framework in English-language texts.

* Wilson, A. (2006). Development and application of a content analysis dictionary for body boundary research. *Literary and Linguistic Computing, 21*(1), 105–110. [https://doi.org/10.1093/llc/fqi014](https://doi.org/10.1093/llc/fqi014)
* Fisher, S., & Cleveland, S. E. (1958). *Body image and personality*. Van Nostrand.

### Dehumanization

*5 categories, 363 terms.* Measures several types of (de)humanization, such as mechanistic and animalistic dehumanization.

* Platten, S., Haji, R., & Boyd, R. L. (2020). *Humanizing and dehumanizing themes of Muslims surrounding 9/11: Computerized language analysis* [Poster]. 81st Canadian Psychological Association Annual National Convention, Montréal, Quebec, Canada.

### Empath

*194 categories, 7,642 terms.* The general content-coding dictionary from the Empath package.

* Fast, E., Chen, B., & Bernstein, M. S. (2016). Empath: Understanding topic signals in large-scale text. In *Proceedings of the 2016 CHI Conference on Human Factors in Computing Systems* (pp. 4647–4657). [https://doi.org/10.1145/2858036.2858535](https://doi.org/10.1145/2858036.2858535)

### Empathic Concern

*1 category, 9,355 terms.* An automatically created empathy dictionary extracted from document-level ratings.

* Sedoc, J., Buechel, S., Nachmany, Y., Buffone, A., & Ungar, L. (2020). Learning word ratings for empathy and distress from document-level user responses. In *Proceedings of the 12th Language Resources and Evaluation Conference* (pp. 1664–1673). [https://www.aclweb.org/anthology/2020.lrec-1.206](https://www.aclweb.org/anthology/2020.lrec-1.206)

### English Prime

*1 category, 45 terms.* Captures violations of the English Prime system, a theoretical marker of cognitive inflexibility.

* Boyd, R. L. (2012). *To be or not to be: An empirical test of English Prime as theory* [Thesis, North Dakota State University]. [https://library.ndsu.edu/ir/handle/10365/26556](https://library.ndsu.edu/ir/handle/10365/26556)

### Forest Values

*4 categories, 608 terms.* Reflects four distinct ways in which people value forests and forest ecosystems.

* Bengston, D. N., & Xu, Z. (1995). *Changing national forest values: A content analysis*. [https://doi.org/10.2737/NC-RP-323](https://doi.org/10.2737/NC-RP-323)

### General Inquirer IV

*182 categories, 8,640 terms.* The original mainstream text analysis dictionary, still widely used today. Many of its categories are of questionable validity.

* Stone, P. J., Bales, R. F., Namenwirth, J. Z., & Ogilvie, D. M. (1962). The General Inquirer: A computer system for content analysis and retrieval based on the sentence as a unit of information. *Behavioral Science, 7*(4), 484–498. [https://doi.org/10.1002/bs.3830070412](https://doi.org/10.1002/bs.3830070412)
* Stone, P. J., Dunphy, D. C., Smith, M. S., & Ogilvie, D. M. (1966). *The General Inquirer: A computer approach to content analysis*. M.I.T. Press.
* Boyd, R. L., & Schwartz, H. A. (2021). Natural language analysis and the psychology of verbal behavior: The past, present, and future states of the field. *Journal of Language and Social Psychology, 40*(1), 21–41. [https://doi.org/10.1177/0261927X20967028](https://doi.org/10.1177/0261927X20967028)

### Grievance

*22 categories, 2,479 terms.* A psycholinguistic dictionary for understanding language use in the context of grievance-fueled violence threat assessment.

* van der Vegt, I., Mozes, M., Kleinberg, B., & Gill, P. (2021). The Grievance Dictionary: Understanding threatening language use. *Behavior Research Methods*. [https://doi.org/10.3758/s13428-021-01536-2](https://doi.org/10.3758/s13428-021-01536-2)

### Home Perceptions

*6 categories, 76 terms.* Counts words describing clutter, a sense of the home as unfinished, restfulness, and nature.

* Saxbe, D. E., & Repetti, R. (2010). No place like home: Home tours correlate with daily patterns of mood and cortisol. *Personality and Social Psychology Bulletin, 36*(1), 71–81. [https://doi.org/10.1177/0146167209352864](https://doi.org/10.1177/0146167209352864)

### Honor

*28 categories, 1,056 terms.* Designed to diagnose "honor talk" in any text you are interested in analyzing.

* Gelfand, M. J., Severance, L., Lee, T., Bruss, C. B., Lun, J., Abdel-Latif, A.-H., Al-Moghazy, A. A., & Moustafa Ahmed, S. (2015). Culture and getting to yes: The linguistic signature of creative agreements in the United States and Egypt. *Journal of Organizational Behavior, 36*(7), 967–989. [https://doi.org/10.1002/job.2026](https://doi.org/10.1002/job.2026)

### Invective

*1 category, 100 terms.* Detects invective language in narrative.

* Panter, A. T. (2017). *Invective language in course evaluations* [Dictionary]. The University of North Carolina at Chapel Hill.

### Linguistic Category Model

*3 categories, 7,477 terms.* A computerized implementation of the Linguistic Category Model.

* Seih, Y.-T., Beier, S., & Pennebaker, J. W. (2017). Development and examination of the Linguistic Category Model in a computerized text analysis method. *Journal of Language and Social Psychology, 36*(3), 343–355. [https://doi.org/10.1177/0261927X16657855](https://doi.org/10.1177/0261927X16657855)

### Mindfulness

*1 category, 63 terms.* Mindfulness language describing the mindfulness state and the more encompassing "mindfulness journey."

* Collins, S. E., Chawla, N., Hsu, S. H., Grow, J., Otto, J. M., & Marlatt, G. A. (2009). Language-based measures of mindfulness: Initial validity and clinical utility. *Psychology of Addictive Behaviors, 23*(4), 743–749. [https://doi.org/10.1037/a0017579](https://doi.org/10.1037/a0017579)

### Moral Foundations

*11 categories, 324 terms.* Gives the proportions of virtue and vice words for each moral foundation.

* Graham, J., Haidt, J., & Nosek, B. A. (2009). Liberals and conservatives rely on different sets of moral foundations. *Journal of Personality and Social Psychology, 96*(5), 1029–1046. [https://doi.org/10.1037/a0015141](https://doi.org/10.1037/a0015141)

### Moral Foundations 2.0

*10 categories, 2,040 terms.* An updated version of the Moral Foundations Dictionary, recommended over the original by its creators.

* Frimer, J. A. (2020). Do liberals and conservatives use different moral languages? Two replications and six extensions of Graham, Haidt, and Nosek's (2009) moral text analysis. *Journal of Research in Personality, 84*, 103906. [https://doi.org/10.1016/j.jrp.2019.103906](https://doi.org/10.1016/j.jrp.2019.103906)

### Moral Foundations - eMFD

*10 categories, 3,270 terms.* The extended Moral Foundations Dictionary, built from text annotations generated by a large sample of human coders rather than from theory alone.

* Hopp, F. R., Fisher, J. T., Cornell, D., Huskey, R., & Weber, R. (2021). The extended Moral Foundations Dictionary (eMFD): Development and applications of a crowd-sourced approach to extracting moral intuitions from text. *Behavior Research Methods, 53*(1), 232–246. [https://doi.org/10.3758/s13428-020-01433-0](https://doi.org/10.3758/s13428-020-01433-0)

* Hopp, F. R., Fisher, J. T., & Weber, R. (2020). A graph-learning approach for detecting moral conflict in movie scripts. *Media and Communication, 8*(3), 164–179. [https://doi.org/10.17645/mac.v8i3.3155](https://doi.org/10.17645/mac.v8i3.3155)

* van Vliet, L. (2021). Moral expressions in 280 characters or less: An analysis of politician tweets following the 2016 Brexit referendum vote. *Frontiers in Big Data*. [https://doi.org/10.3389/fdata.2021.699653](https://doi.org/10.3389/fdata.2021.699653)

### Morality as Cooperation

*14 categories, 3,026 terms.* Codes for constructs from Morality-as-Cooperation theory, which originates in ethnographic accounts of morality.

* Alfano, M., Cheong, M., & Curry, O. S. (2024). Moral universals: A machine-reading analysis of 256 societies. *Heliyon*, e25940. [https://doi.org/10.1016/j.heliyon.2024.e25940](https://doi.org/10.1016/j.heliyon.2024.e25940)

### Pain

*5 categories, 181 terms.* Measures the language of pain disclosure, built with attention to previously validated pain scales.

* Wright, R. C., Junghaenel, D. U., Rivas, R., Hristidis, V., & Robbins, M. L. (2020). A new approach to capturing pain disclosure in daily life in-person and online. *Journal of Health Psychology*. Advance online publication. [https://doi.org/10.1177/1359105320918322](https://doi.org/10.1177/1359105320918322)

### Personal Values

*14 categories, 1,068 terms.* Measures the 10 Schwartz values and 4 higher-order value dimensions.

* Ponizovskiy, V., Ardag, M., Grigoryan, L., Boyd, R., Dobewall, H., & Holtz, P. (2020). Development and validation of the Personal Values Dictionary: A theory-driven tool for investigating references to basic human values in text. *European Journal of Personality, 34*(5), 885–902. [https://doi.org/10.1002/per.2294](https://doi.org/10.1002/per.2294)

### Privacy

*9 categories, 312 terms.* Words people use when talking about privacy, organized into eight categories measuring different dimensions of privacy.

* Gill, A. J., Vasalou, A., Papoutsi, C., & Joinson, A. N. (2011). Privacy dictionary: A linguistic taxonomy of privacy for content analysis. In *Proceedings of the SIGCHI Conference on Human Factors in Computing Systems* (pp. 3227–3236). Association for Computing Machinery. [http://doi.org/10.1145/1978942.1979421](http://doi.org/10.1145/1978942.1979421)

* Vasalou, A., Gill, A. J., Mazanderani, F., Papoutsi, C., & Joinson, A. (2011). Privacy dictionary: A new resource for the automated content analysis of privacy. *Journal of the American Society for Information Science and Technology, 62*(11), 2095–2105. [https://doi.org/10.1002/asi.21610](https://doi.org/10.1002/asi.21610)

### Prosocial Words

*1 category, 127 terms.* Measures the density of prosocial words in anything a person says.

* Frimer, J. A., Schaefer, N. K., & Oakes, H. (2014). Moral actor, selfish agent. *Journal of Personality and Social Psychology, 106*(5), 790–802. [https://doi.org/10.1037/a0036040](https://doi.org/10.1037/a0036040)

* Frimer, J. A., Aquino, K., Gebauer, J. E., Zhu, L. (Lei), & Oakes, H. (2015). A decline in prosocial language helps explain public disapproval of the US Congress. *Proceedings of the National Academy of Sciences, 112*(21), 6591–6594. [https://doi.org/10.1073/pnas.1500355112](https://doi.org/10.1073/pnas.1500355112)

### Qualia

*19 categories, 405 terms.* Words referring to the five senses, broken down by type of qualia (for vision: colors, luminance, actions, and shapes).

* Developed by Molly Ireland and colleagues.

### Regressive Imagery

*51 categories, 3,149 terms.* Captures primary (primordial) and secondary (conceptual) process thinking in natural language.

* Martindale, C. (1975). The grammar of altered states of consciousness: A semiotic reinterpretation of aspects of psychoanalytic theory. *Psychoanalysis and Contemporary Thought, 4*, 331–354.

* Martindale, C. (1990). *The clockwork muse: The predictability of artistic change*. Basic Books.

* Martindale, C. (2007). Creativity, primordial cognition, and personality. *Personality and Individual Differences, 43*(7), 1777–1785. [https://doi.org/10.1016/j.paid.2007.05.014](https://doi.org/10.1016/j.paid.2007.05.014)

* West, A. N., & Martindale, C. (1988). Primary process content in paranoid schizophrenic speech. *The Journal of Genetic Psychology, 149*(4), 547–553. [https://doi.org/10.1080/00221325.1988.10532180](https://doi.org/10.1080/00221325.1988.10532180)

### Self-Care

*1 category, 35 terms.* Measures the degree to which self-care words are used (e.g., diet, yoga).

* Wang, X., Parameswaran, S., Bagul, D., & Kishore, R. (2017). Does online social support work in stigmatized chronic diseases? A study of the impacts of different facets of informational and emotional support on self-care behavior in an HIV online forum. *ICIS 2017 Proceedings*. [https://aisel.aisnet.org/icis2017/General/Presentations/22](https://aisel.aisnet.org/icis2017/General/Presentations/22)

* Wang, X., Parameswaran, S., Bagul, D. M., & Kishore, R. (2018). Can online social support be detrimental in stigmatized chronic diseases? A quadratic model of the effects of informational and emotional support on self-care behavior of HIV patients. *Journal of the American Medical Informatics Association, 25*(8), 931–944. [https://doi.org/10.1093/jamia/ocy012](https://doi.org/10.1093/jamia/ocy012)

### Self-Determination Self-Talk

*2 categories, 35 terms.* Built for self-talk data; measures autonomy-supportive versus controlling language within self-talk.

* Oliver, E. J., Markland, D., Hardy, J., & Petherick, C. M. (2008). The effects of autonomy-supportive versus controlling environments on self-talk. *Motivation and Emotion, 32*(3), 200–212. [https://doi.org/10.1007/s11031-008-9097-x](https://doi.org/10.1007/s11031-008-9097-x)

### Situational 8 DIAMONDS

*8 categories, 430 terms.* Captures language corresponding to each of the DIAMONDS dimensions of situations.

* Serfass, D. G., & Sherman, R. A. (2015). Situations in 140 characters: Assessing real-world situations on Twitter. *PLoS ONE, 10*(11). [https://doi.org/10.1371/journal.pone.0143051](https://doi.org/10.1371/journal.pone.0143051)

### Stereotype Content

*43 categories, 14,449 terms.* A stereotype content dictionary built by a semi-automated method to capture the Stereotype Content Model in text.

* Nicolas, G., Bai, X., & Fiske, S. T. (2021). Comprehensive stereotype content dictionaries using a semi-automated method. *European Journal of Social Psychology, 51*(1), 178–196. [https://doi.org/10.1002/ejsp.2724](https://doi.org/10.1002/ejsp.2724)

### Urban Dictionary Supplements

*43 categories, 141,021 terms.* An automatically generated extension to the LIWC dictionary covering terms defined in Urban Dictionary. Much the largest dictionary here; expect it to take a while.

* Bahgat, M., Wilson, S., & Magdy, W. (2022). LIWC-UD: Classifying online slang terms into LIWC categories. In *14th ACM Web Science Conference 2022* (pp. 422–432). [https://doi.org/10.1145/3501247.3531572](https://doi.org/10.1145/3501247.3531572)

### Well-being

*11 categories, 64 terms.* Words that may indicate the presence of purpose or meaning.

* Ratner, K., Burrow, A. L., Burd, K. A., & Hill, P. L. (2019). On the conflation of purpose and meaning in life: A qualitative study of high school and college student conceptions. *Applied Developmental Science*, 1–21. [https://doi.org/10.1080/10888691.2019.1659140](https://doi.org/10.1080/10888691.2019.1659140)

### Whirlall

*1 category, 18 terms.* Measures words related to "whirling" and "twirling" in Rorschach responses.

* Thomas, C. B., & Duszynski, K. R. (1985). Are words of the Rorschach predictors of disease and death? The case of "whirling." *Psychosomatic Medicine, 47*(2), 201–211.

## Archetype dictionaries

Archetype dictionaries are CSVs of **seed sentences** rather than word lists.
Each construct is defined by example statements, and a sentence-transformers
model scores how close each of your texts sits to them. See
[Archetypes](analyzing-text.md#archetypes-theory-driven-embedding-based) for
what the scores mean and how to write your own.

### Resilience

Facets of psychological resilience — optimism, sense of social support, and
related constructs — each defined by prototype statements.

* Mahwish, S., Boyd, R. L., Varadarajan, V., Kotov, R., Luft, B. J., Schwartz, H. A., & Clouston, S. A. P. (2026). Measuring resilience using language modeling: A computational approach to observing resilience. *Journal of Traumatic Stress, 39*(3). [https://doi.org/10.1002/jts.70046](https://doi.org/10.1002/jts.70046)

### Suicidality

Risk constructs drawn from the major theories of suicide — acquired
capability, perceived burdensomeness, thwarted belongingness, escape, and
others — each defined by prototype statements.

* Boyd, R. L., Vallejo, I., & Lanning, K. (2026). Toward an integrative science of suicidality: Understanding suicide risk factors through real-world natural language. *Journal of Psychopathology and Clinical Science*. [https://doi.org/10.1037/abn0001157](https://doi.org/10.1037/abn0001157)

## Using a dictionary as a word-vector concept dictionary

Content-coding dictionaries and word-vector concept dictionaries are interchangeable
Anything listed above under **Content-coding dictionaries** can be
handed to **Train a model → Word vectors** as a concept dictionary, where its
categories seed a search through the vector space your own corpus produced.
Archetype dictionaries cannot: they are seed *sentences* for an embedding
model, not term lists, and are only offered where archetypes are.
