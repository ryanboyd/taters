# Built-in dictionaries, norms and archetypes

Taters ships with a collection of ready-made **content-coding dictionaries**,
**word-norm tables** and **archetype dictionaries**. You do not have to download or import them: the
first time Taters needs your library it copies them into
`~/.taters/library/`, and from then on they appear wherever a flow asks you
to pick one —

* **Extract features → Dictionary counts (LIWC-style)**, which counts how
  often each category's words appear in your texts;
* **Extract features → Archetype similarity**, which scores each text by how
  close it sits to a set of seed phrases;
* **a saved word-vector model's apply settings** (Settings → Manage Taters
  data → Manage saved models), where a content-coding dictionary can be set
  as the **concept dictionary** whose categories get expanded through the
  vector space your own corpus produced when the model scores texts;
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


### American Indian Stereotype

*6 categories, 233 terms.* Words associated with stereotypes of American Indians, built for work on whether perspective-taking reduces bias.

* Sherman, A., Cupo, L., & Mithlo, N. M. (2020). Perspective-taking increases emotionality and empathy but does not reduce harmful biases against American Indians: Converging evidence from the museum and lab. PLOS ONE, 15(2), e0228784. [https://doi.org/10.1371/journal.pone.0228784](https://doi.org/10.1371/journal.pone.0228784)


### Arguing Lexicon

*17 categories, 5,049 terms.* Patterns that signal somebody is arguing rather than simply stating — assessments, authority, conditionals, inconsistency and the rest.

* Somasundaran, S., Ruppenhofer, J., & Wiebe, J. (2007). Detecting Arguing and Sentiment in Meetings. In Proceedings of the 8th SIGdial Workshop on Discourse and Dialogue (pp. 26–34). Antwerp: Association for Computational Linguistics.
* Expanded / modified from the original RegEx/Macro version using EXREX (https://github.com/asciimoo/exrex)


### Attention Focus

*4 categories, 68 terms.* Where attention is pointed: long-term against short-term, internal against external.

* Gregson, A. H. C. (2018). Under what conditions does CEO attention influence corporate social performance? The moderating effect of span of control, executive committee gender diversity and industry dynamism. (Master’s Thesis). Universiteit van Amsterdam, Amsterdam Business School. Retrieved from [http://www.scriptiesonline.uba.uva.nl/document/667864](http://www.scriptiesonline.uba.uva.nl/document/667864)


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


### Brand Personality

*6 categories, 833 terms.* The classic brand personality traits — sincerity, excitement, competence, sophistication, ruggedness.

* Opoku, R. A., Hultman, M., & Saheli-Sangari, E. (2008). Positioning in Market Space: The Evaluation of Swedish Universities’ Online Brand Personalities. Journal of Marketing for Higher Education, 18(1), 124–144. [https://doi.org/10.1080/08841240802100386](https://doi.org/10.1080/08841240802100386)


### Concreteness (Suffixes)

*1 category, 20 terms.* Mergenthaler's abstractness measure, which spots abstraction from word endings rather than from a word list.

* Mergenthaler, E. (1996). Emotion-abstraction patterns in verbatim protocols: A new way of describing psychotherapeutic processes. Journal of Consulting and Clinical Psychology, 64(6), 1306–1315.


### Connectives

*1 category, 100 terms.* Connective words and phrases as a single category. The cohesion step has its own, finer-grained connective lists if you want them split up.

* Ko, W.-J., Durrett, G., & Li, J. J. (2019). Domain Agnostic Real-Valued Specificity Prediction. Proceedings of the AAAI Conference on Artificial Intelligence, 33, 6610–6617. [https://doi.org/10.1609/aaai.v33i01.33016610](https://doi.org/10.1609/aaai.v33i01.33016610)


### Controversial Terms

*3 categories, 462 terms.* Terms that turned up around controversial news stories.

* Mejova, Y., Zhang, A. X., Diakopoulos, N., & Castillo, C. (2014). Controversy and sentiment in online news. ArXiv:1409.8152 [Cs]. [http://arxiv.org/abs/1409.8152](http://arxiv.org/abs/1409.8152)


### Corporate Social Responsibility

*4 categories, 955 terms.* CSR language across environment, community, employees and human rights, built from company prospectuses.

* Pencle, N., & Mălăescu, I. (2016). What’s in the words? Development and validation of a multidimensional dictionary for CSR and application using prospectuses. Journal of Emerging Technologies in Accounting, 13(2), 109–127. [https://doi.org/10.2308/jeta-51615](https://doi.org/10.2308/jeta-51615)


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


### Gendered Terms

*2 categories, 82 terms.* Masculine and feminine coded words, from the work on gendered wording in job adverts.

* Gaucher, D., Friesen, J., & Kay, A. C. (2011). Evidence that gendered wording in job advertisements exists and sustains gender inequality. Journal of Personality and Social Psychology, 101(1), 109–128. [https://doi.org/10.1037/a0022530](https://doi.org/10.1037/a0022530)


### General Inquirer IV

*182 categories, 8,640 terms.* The original mainstream text analysis dictionary, still widely used today. Many of its categories are of questionable validity.

* Stone, P. J., Bales, R. F., Namenwirth, J. Z., & Ogilvie, D. M. (1962). The General Inquirer: A computer system for content analysis and retrieval based on the sentence as a unit of information. *Behavioral Science, 7*(4), 484–498. [https://doi.org/10.1002/bs.3830070412](https://doi.org/10.1002/bs.3830070412)
* Stone, P. J., Dunphy, D. C., Smith, M. S., & Ogilvie, D. M. (1966). *The General Inquirer: A computer approach to content analysis*. M.I.T. Press.
* Boyd, R. L., & Schwartz, H. A. (2021). Natural language analysis and the psychology of verbal behavior: The past, present, and future states of the field. *Journal of Language and Social Psychology, 40*(1), 21–41. [https://doi.org/10.1177/0261927X20967028](https://doi.org/10.1177/0261927X20967028)


### Global Citizen

*1 category, 129 terms.* Markers of global citizenship — identifying with humanity rather than with one nation.

* Reysen, S., Pierce, L., Mazambani, G., Mohebpour, I., Puryear, C., Snider, J. S., … Blake, M. E. (2014). Construction and initial validation of a dictionary for global citizen linguistic markers. International Journal of Cyber Behavior, Psychology and Learning, 4(4), 1–15. [https://doi.org/10.4018/ijcbpl.2014100101](https://doi.org/10.4018/ijcbpl.2014100101)


### Grant Evaluation

*7 categories, 358 terms.* The language reviewers use in grant critiques, built from NIH R01 summary statements.

* Kaatz, A., Magua, W., Zimmerman, D. R., & Carnes, M. (2015). A quantitative linguistic analysis of National Institutes of Health R01 application critiques from investigators at one institution. Academic Medicine : Journal of the Association of American Medical Colleges, 90(1), 69–75. [https://doi.org/10.1097/ACM.0000000000000442](https://doi.org/10.1097/ACM.0000000000000442)


### Grievance

*22 categories, 2,479 terms.* A psycholinguistic dictionary for understanding language use in the context of grievance-fueled violence threat assessment.

* van der Vegt, I., Mozes, M., Kleinberg, B., & Gill, P. (2021). The Grievance Dictionary: Understanding threatening language use. *Behavior Research Methods*. [https://doi.org/10.3758/s13428-021-01536-2](https://doi.org/10.3758/s13428-021-01536-2)


### Hedges

*1 category, 99 terms.* Hedging words — the ones people reach for when they are softening a claim.

* Danescu-Niculescu-Mizil, C., Sudhof, M., Jurafsky, D., Leskovec, J., & Potts, C. (2013). A computational approach to politeness with application to social factors. Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 250–259. Retrieved from [https://www.aclweb.org/anthology/P13-1025](https://www.aclweb.org/anthology/P13-1025)


### Hogenraad Anticipation

*10 categories, 845 terms.* Hogenraad's lexicon for anticipation and dread, used in his work on a computer-readable fear index.

* Hogenraad, R. (2019). Fear in the West: A sentiment analysis using a computer-readable “Fear Index.” Quality & Quantity, 53(3), 1239–1261. [https://doi.org/10.1007/s11135-018-0813-7](https://doi.org/10.1007/s11135-018-0813-7)


### Hogenraad Metamorphosis

*9 categories, 627 terms.* Hogenraad's transfiguration lexicon, for language about transformation and change of state.

* [https://archive.org/details/aeTRANSFIGdlc](https://archive.org/details/aeTRANSFIGdlc)


### Home Perceptions

*6 categories, 76 terms.* Counts words describing clutter, a sense of the home as unfinished, restfulness, and nature.

* Saxbe, D. E., & Repetti, R. (2010). No place like home: Home tours correlate with daily patterns of mood and cortisol. *Personality and Social Psychology Bulletin, 36*(1), 71–81. [https://doi.org/10.1177/0146167209352864](https://doi.org/10.1177/0146167209352864)


### Honor

*28 categories, 1,056 terms.* Designed to diagnose "honor talk" in any text you are interested in analyzing.

* Gelfand, M. J., Severance, L., Lee, T., Bruss, C. B., Lun, J., Abdel-Latif, A.-H., Al-Moghazy, A. A., & Moustafa Ahmed, S. (2015). Culture and getting to yes: The linguistic signature of creative agreements in the United States and Egypt. *Journal of Organizational Behavior, 36*(7), 967–989. [https://doi.org/10.1002/job.2026](https://doi.org/10.1002/job.2026)


### Interpersonal Psychological Theory

*3 categories, 34 terms.* The interpersonal theory of suicide — thwarted belongingness, perceived burdensomeness, acquired capability.

* Cantu, S., & Dykeman, C. (2019). Linguistic and personological features of suicidality in bulimia: A study of memoirs. PsyArXiv. [https://doi.org/10.31234/osf.io/c23wn](https://doi.org/10.31234/osf.io/c23wn)


### Invective

*1 category, 100 terms.* Detects invective language in narrative.

* Panter, A. T. (2017). *Invective language in course evaluations* [Dictionary]. The University of North Carolina at Chapel Hill.


### Linguistic Category Model

*3 categories, 7,477 terms.* A computerized implementation of the Linguistic Category Model.

* Seih, Y.-T., Beier, S., & Pennebaker, J. W. (2017). Development and examination of the Linguistic Category Model in a computerized text analysis method. *Journal of Language and Social Psychology, 36*(3), 343–355. [https://doi.org/10.1177/0261927X16657855](https://doi.org/10.1177/0261927X16657855)


### LUSI Action-Inaction

*2 categories, 1,032 terms.* Language of acting against language of holding back.

* [https://www.depts.ttu.edu/psy/lusi/resources.php](https://www.depts.ttu.edu/psy/lusi/resources.php)


### LUSI Disengagement

*5 categories, 157 terms.* Moral disengagement language, in the modified LUSI version.

* [https://www.depts.ttu.edu/psy/lusi/resources.php](https://www.depts.ttu.edu/psy/lusi/resources.php)


### LUSI Netspeak

*12 categories, 1,043 terms.* Internet speech — abbreviations, emoticons, the usual online shorthand.

* [https://www.depts.ttu.edu/psy/lusi/resources.php](https://www.depts.ttu.edu/psy/lusi/resources.php)


### LUSI Risky Sex

*1 category, 224 terms.* Language about sexual risk-taking.

* [https://www.depts.ttu.edu/psy/lusi/resources.php](https://www.depts.ttu.edu/psy/lusi/resources.php)


### Mind Perception

*5 categories, 327 terms.* Mental-state language, splitting mind from body and self from other.

* Schweitzer, S., & Waytz, A. (2020). Language as a window into mind perception: How mental state language differentiates body and mind, human and nonhuman, and the self from others. Journal of Experimental Psychology: General. Advance online publication. [https://doi.org/10.1037/xge0001013](https://doi.org/10.1037/xge0001013)


### Mindfulness

*1 category, 63 terms.* Mindfulness language describing the mindfulness state and the more encompassing "mindfulness journey."

* Collins, S. E., Chawla, N., Hsu, S. H., Grow, J., Otto, J. M., & Marlatt, G. A. (2009). Language-based measures of mindfulness: Initial validity and clinical utility. *Psychology of Addictive Behaviors, 23*(4), 743–749. [https://doi.org/10.1037/a0017579](https://doi.org/10.1037/a0017579)


### Moral Foundations

*11 categories, 324 terms.* Gives the proportions of virtue and vice words for each moral foundation.

* Graham, J., Haidt, J., & Nosek, B. A. (2009). Liberals and conservatives rely on different sets of moral foundations. *Journal of Personality and Social Psychology, 96*(5), 1029–1046. [https://doi.org/10.1037/a0015141](https://doi.org/10.1037/a0015141)


### Moral Foundations - eMFD

*10 categories, 3,270 terms.* The extended Moral Foundations Dictionary, built from text annotations generated by a large sample of human coders rather than from theory alone.

* Hopp, F. R., Fisher, J. T., Cornell, D., Huskey, R., & Weber, R. (2021). The extended Moral Foundations Dictionary (eMFD): Development and applications of a crowd-sourced approach to extracting moral intuitions from text. *Behavior Research Methods, 53*(1), 232–246. [https://doi.org/10.3758/s13428-020-01433-0](https://doi.org/10.3758/s13428-020-01433-0)

* Hopp, F. R., Fisher, J. T., & Weber, R. (2020). A graph-learning approach for detecting moral conflict in movie scripts. *Media and Communication, 8*(3), 164–179. [https://doi.org/10.17645/mac.v8i3.3155](https://doi.org/10.17645/mac.v8i3.3155)

* van Vliet, L. (2021). Moral expressions in 280 characters or less: An analysis of politician tweets following the 2016 Brexit referendum vote. *Frontiers in Big Data*. [https://doi.org/10.3389/fdata.2021.699653](https://doi.org/10.3389/fdata.2021.699653)


### Moral Foundations 2.0

*10 categories, 2,040 terms.* An updated version of the Moral Foundations Dictionary, recommended over the original by its creators.

* Frimer, J. A. (2020). Do liberals and conservatives use different moral languages? Two replications and six extensions of Graham, Haidt, and Nosek's (2009) moral text analysis. *Journal of Research in Personality, 84*, 103906. [https://doi.org/10.1016/j.jrp.2019.103906](https://doi.org/10.1016/j.jrp.2019.103906)


### Moral Justification

*12 categories, 225 terms.* How people justify moral violations — deontological, consequentialist and emotive reasoning.

* Wheeler, M. A., & Laham, S. M. (2016). What we talk about when we talk about morality: Deontological, consequentialist, and emotive language use in justifications across foundation-specific moral violations. Personality and Social Psychology Bulletin, 42(9), 1206–1216. [https://doi.org/10.1177/0146167216653374](https://doi.org/10.1177/0146167216653374)


### Morality as Cooperation

*14 categories, 3,026 terms.* Codes for constructs from Morality-as-Cooperation theory, which originates in ethnographic accounts of morality.

* Alfano, M., Cheong, M., & Curry, O. S. (2024). Moral universals: A machine-reading analysis of 256 societies. *Heliyon*, e25940. [https://doi.org/10.1016/j.heliyon.2024.e25940](https://doi.org/10.1016/j.heliyon.2024.e25940)


### Motivated Social Cognition

*4 categories, 127 terms.* Categories from the motivated social cognition account of political ideology.

* Neiman, J. L., Gonzalez, F. K., Wilkinson, K., Smith, K. B., & Hibbing, J. R. (2016). Speaking different languages or reading from the same script? Word usage of Democratic and Republican politicians. Political Communication, 33(2), 346–349. [https://doi.org/10.1080/10584609.2016.1161349](https://doi.org/10.1080/10584609.2016.1161349)


### Opinion Lexicon

*2 categories, 6,769 terms.* Hu and Liu's positive and negative opinion words, from the customer-review mining literature.

* Hu, M., & Liu, B. (2004). Mining and Summarizing Customer Reviews. In Proceedings of the Tenth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 168–177). New York, NY, USA: ACM. [https://doi.org/10.1145/1014052.1014073](https://doi.org/10.1145/1014052.1014073)


### Pain

*5 categories, 181 terms.* Measures the language of pain disclosure, built with attention to previously validated pain scales.

* Wright, R. C., Junghaenel, D. U., Rivas, R., Hristidis, V., & Robbins, M. L. (2020). A new approach to capturing pain disclosure in daily life in-person and online. *Journal of Health Psychology*. Advance online publication. [https://doi.org/10.1177/1359105320918322](https://doi.org/10.1177/1359105320918322)


### PANAS-X

*16 categories, 60 terms.* The PANAS-X affect scales as word categories.

* Watson, D., & Clark, L. A. (1994). The PANAS-X: Manual for the positive and negative affect schedule-expanded form. Iowa City, IA: The University of Iowa.


### Parenting Styles Model

*4 categories, 150 terms.* Categories from the parenting-styles account of political ideology — the strict-father and nurturant-parent models.

* Neiman, J. L., Gonzalez, F. K., Wilkinson, K., Smith, K. B., & Hibbing, J. R. (2016). Speaking different languages or reading from the same script? Word usage of Democratic and Republican politicians. Political Communication, 33(2), 346–349. [https://doi.org/10.1080/10584609.2016.1161349](https://doi.org/10.1080/10584609.2016.1161349)


### Personal Values

*14 categories, 1,068 terms.* Measures the 10 Schwartz values and 4 higher-order value dimensions.

* Ponizovskiy, V., Ardag, M., Grigoryan, L., Boyd, R., Dobewall, H., & Holtz, P. (2020). Development and validation of the Personal Values Dictionary: A theory-driven tool for investigating references to basic human values in text. *European Journal of Personality, 34*(5), 885–902. [https://doi.org/10.1002/per.2294](https://doi.org/10.1002/per.2294)


### Physiological Sensation

*1 category, 86 terms.* Bodily sensation words, from work on social anxiety with and without depression.

* Shaffer, V. N., Kim, D., & Yoon, K. L. (2021). Physiological sensation word usage in social anxiety disorder with and without comorbid depression. Journal of Behavior Therapy and Experimental Psychiatry, 71, 101638. [https://doi.org/10.1016/j.jbtep.2021.101638](https://doi.org/10.1016/j.jbtep.2021.101638)


### Policy Position (Laver-Garry)

*26 categories, 417 terms.* Laver and Garry's policy-position categories, built from UK party manifestos. Aimed squarely at British political text.

* Laver, M., Benoit, K., & Garry, J. (2003). Extracting Policy Positions from Political Texts Using Words as Data. American Political Science Review, 97(2), 311–331. [https://doi.org/10.1017/S0003055403000698](https://doi.org/10.1017/S0003055403000698)


### Privacy

*9 categories, 312 terms.* Words people use when talking about privacy, organized into eight categories measuring different dimensions of privacy.

* Gill, A. J., Vasalou, A., Papoutsi, C., & Joinson, A. N. (2011). Privacy dictionary: A linguistic taxonomy of privacy for content analysis. In *Proceedings of the SIGCHI Conference on Human Factors in Computing Systems* (pp. 3227–3236). Association for Computing Machinery. [http://doi.org/10.1145/1978942.1979421](http://doi.org/10.1145/1978942.1979421)

* Vasalou, A., Gill, A. J., Mazanderani, F., Papoutsi, C., & Joinson, A. (2011). Privacy dictionary: A new resource for the automated content analysis of privacy. *Journal of the American Society for Information Science and Technology, 62*(11), 2095–2105. [https://doi.org/10.1002/asi.21610](https://doi.org/10.1002/asi.21610)


### Prorefugee

*4 categories, 113 terms.* Solidarity language about refugees — death, threat, harm and support.

* Smith, L. G. E., McGarty, C., & Thomas, E. F. (2018). After Aylan Kurdi: How Tweeting About Death, Threat, and Harm Predict Increased Expressions of Solidarity With Refugees Over Time. Psychological Science, 29(4), 623–634. [https://doi.org/10.1177/0956797617741107](https://doi.org/10.1177/0956797617741107)


### Prosocial Words

*1 category, 127 terms.* Measures the density of prosocial words in anything a person says.

* Frimer, J. A., Schaefer, N. K., & Oakes, H. (2014). Moral actor, selfish agent. *Journal of Personality and Social Psychology, 106*(5), 790–802. [https://doi.org/10.1037/a0036040](https://doi.org/10.1037/a0036040)

* Frimer, J. A., Aquino, K., Gebauer, J. E., Zhu, L. (Lei), & Oakes, H. (2015). A decline in prosocial language helps explain public disapproval of the US Congress. *Proceedings of the National Academy of Sciences, 112*(21), 6591–6594. [https://doi.org/10.1073/pnas.1500355112](https://doi.org/10.1073/pnas.1500355112)


### Psychodynamic Conflict

*1 category, 14 terms.* Conflict language as used in the psychoanalytic literature.

* Dent, L., & Christian, C. (2019). The shifting prevalence of conflict in psychoanalytic literature: A brief report of a corpus-based text analysis. Psychoanalytic Psychology, 36(2), 184–188. [https://doi.org/10.1037/pap0000220](https://doi.org/10.1037/pap0000220)


### Qualia

*19 categories, 405 terms.* Words referring to the five senses, broken down by type of qualia (for vision: colors, luminance, actions, and shapes).

* Developed by Molly Ireland and colleagues.


### Regressive Imagery

*51 categories, 3,149 terms.* Captures primary (primordial) and secondary (conceptual) process thinking in natural language.

* Martindale, C. (1975). The grammar of altered states of consciousness: A semiotic reinterpretation of aspects of psychoanalytic theory. *Psychoanalysis and Contemporary Thought, 4*, 331–354.

* Martindale, C. (1990). *The clockwork muse: The predictability of artistic change*. Basic Books.

* Martindale, C. (2007). Creativity, primordial cognition, and personality. *Personality and Individual Differences, 43*(7), 1777–1785. [https://doi.org/10.1016/j.paid.2007.05.014](https://doi.org/10.1016/j.paid.2007.05.014)

* West, A. N., & Martindale, C. (1988). Primary process content in paranoid schizophrenic speech. *The Journal of Genetic Psychology, 149*(4), 547–553. [https://doi.org/10.1080/00221325.1988.10532180](https://doi.org/10.1080/00221325.1988.10532180)


### Regulatory Mode

*2 categories, 68 terms.* Locomotion against assessment — getting on with it against weighing it up.

* Kanze, D., Conley, M. A., & Higgins, E. T. (2019). The motivation of mission statements: How regulatory mode influences workplace discrimination. Organizational Behavior and Human Decision Processes. [https://doi.org/10.1016/j.obhdp.2019.04.002](https://doi.org/10.1016/j.obhdp.2019.04.002)


### Schwartz Values

*10 categories, 1,313 terms.* The ten basic human values of the Schwartz circumplex.

* Jones, K. L., Noorbaloochi, S., Jost, J. T., Bonneau, R., Nagler, J., & Tucker, J. A. (2018). Liberal and Conservative Values: What We Can Learn From Congressional Tweets. Political Psychology, 39(2), 423–443. [https://doi.org/10.1111/pops.12415](https://doi.org/10.1111/pops.12415)


### Security Lexicon

*1 category, 226 terms.* Security and threat language.

* Baele, S. J., & Sterck, O. C. (2015). Diagnosing the Securitisation of Immigration at the EU Level: A New Method for Stronger Empirical Claims. Political Studies, 63(5), 1120–1139. [https://doi.org/10.1111/1467-9248.12147](https://doi.org/10.1111/1467-9248.12147)


### Self-Care

*1 category, 35 terms.* Measures the degree to which self-care words are used (e.g., diet, yoga).

* Wang, X., Parameswaran, S., Bagul, D., & Kishore, R. (2017). Does online social support work in stigmatized chronic diseases? A study of the impacts of different facets of informational and emotional support on self-care behavior in an HIV online forum. *ICIS 2017 Proceedings*. [https://aisel.aisnet.org/icis2017/General/Presentations/22](https://aisel.aisnet.org/icis2017/General/Presentations/22)

* Wang, X., Parameswaran, S., Bagul, D. M., & Kishore, R. (2018). Can online social support be detrimental in stigmatized chronic diseases? A quadratic model of the effects of informational and emotional support on self-care behavior of HIV patients. *Journal of the American Medical Informatics Association, 25*(8), 931–944. [https://doi.org/10.1093/jamia/ocy012](https://doi.org/10.1093/jamia/ocy012)


### Self-Determination Self-Talk

*2 categories, 35 terms.* Built for self-talk data; measures autonomy-supportive versus controlling language within self-talk.

* Oliver, E. J., Markland, D., Hardy, J., & Petherick, C. M. (2008). The effects of autonomy-supportive versus controlling environments on self-talk. *Motivation and Emotion, 32*(3), 200–212. [https://doi.org/10.1007/s11031-008-9097-x](https://doi.org/10.1007/s11031-008-9097-x)


### Self-Transcendent Emotion

*6 categories, 351 terms.* Awe, elevation, gratitude and admiration — the emotions that point outward from the self.

* Ji, Q., & Raney, A. A. (2020). Developing and validating the self-transcendent emotion dictionary for text analysis. PLOS ONE, 15(9), e0239050. [https://doi.org/10.1371/journal.pone.0239050](https://doi.org/10.1371/journal.pone.0239050)


### Situational 8 DIAMONDS

*8 categories, 430 terms.* Captures language corresponding to each of the DIAMONDS dimensions of situations.

* Serfass, D. G., & Sherman, R. A. (2015). Situations in 140 characters: Assessing real-world situations on Twitter. *PLoS ONE, 10*(11). [https://doi.org/10.1371/journal.pone.0143051](https://doi.org/10.1371/journal.pone.0143051)


### Social Ties

*1 category, 102 terms.* References to social relationships and connection.

* Pressman, S. D., & Cohen, S. (2007). Use of social words in autobiographies and longevity. Psychosomatic Medicine, 69(3), 262–269.


### Stress

*1 category, 270 terms.* Stress language, built from social media posts.

* Wang, W., Hernandez, I., Newman, D. A., He, J., & Bian, J. (2016). Twitter analysis: Studying US weekly trends in work stress and emotion. Applied Psychology, 65(2), 355–378. [https://doi.org/10.1111/apps.12065](https://doi.org/10.1111/apps.12065)


### Urban Dictionary Supplements

*43 categories, 141,021 terms.* An automatically generated extension to the LIWC dictionary covering terms defined in Urban Dictionary. Much the largest dictionary here; expect it to take a while.

* Bahgat, M., Wilson, S., & Magdy, W. (2022). LIWC-UD: Classifying online slang terms into LIWC categories. In *14th ACM Web Science Conference 2022* (pp. 422–432). [https://doi.org/10.1145/3501247.3531572](https://doi.org/10.1145/3501247.3531572)


### Value Lexicon (Bardi)

*10 categories, 30 terms.* Bardi's value categories, again on the Schwartz model.

* Bardi, A., Calogero, R. M., & Mullen, B. (2008). A new archival approach to the study of values and value--Behavior relations: Validation of the value lexicon. Journal of Applied Psychology, 93(3), 483–497. [https://doi.org/10.1037/0021-9010.93.3.483](https://doi.org/10.1037/0021-9010.93.3.483)


### Value Lexicon (Wilson)

*50 categories, 1,267 terms.* Wilson's much larger value lexicon — fifty categories rather than ten.

* Wilson, S. R., Shen, Y., & Mihalcea, R. (2018). Building and Validating Hierarchical Lexicons with a Case Study on Personal Values. In S. Staab, O. Koltsova, & D. I. Ignatov (Eds.), Social Informatics (pp. 455–470). Springer International Publishing.


### Well-being

*11 categories, 64 terms.* Words that may indicate the presence of purpose or meaning.

* Ratner, K., Burrow, A. L., Burd, K. A., & Hill, P. L. (2019). On the conflation of purpose and meaning in life: A qualitative study of high school and college student conceptions. *Applied Developmental Science*, 1–21. [https://doi.org/10.1080/10888691.2019.1659140](https://doi.org/10.1080/10888691.2019.1659140)


### Whirlall

*1 category, 18 terms.* Measures words related to "whirling" and "twirling" in Rorschach responses.

* Thomas, C. B., & Duszynski, K. R. (1985). Are words of the Rorschach predictors of disease and death? The case of "whirling." *Psychosomatic Medicine, 47*(2), 201–211.

## Word norms

These are rating tables rather than word lists: each one attaches a number to
a word — how concrete it is, how pleasant, how early in life you learned it —
and a text's score is the **average rating of the words in it that had one**.
That is a different question from the one a content-coding dictionary answers,
which is why they live in their own library and have their own step (**Extract
features → Word norms**) rather than being mixed in above.

Every rating also gets a `_Coverage` column saying what share of the text was
rated at all. Read it: a mean taken over three words and a mean taken over
three hundred look identical otherwise. A text with no rated words in it gets
an empty cell rather than a zero, because it has no score rather than a score
of nothing.

One difference from the versions of these that shipped with BUTTER: the
`_intercept` row is gone. That was a constant from a fitted linear model and
has no meaning here, so files carrying one are refused rather than mis-scored.
The lexica that needed it to work are not included.

### Abusive Words

*1 rating, 7,049 terms.* How abusive a word is, from a lexicon induced with a feature-based classifier.

* Wiegand, M., Ruppenhofer, J., Schmidt, A., & Greenberg, C. (2018). Inducing a lexicon of abusive words: A feature-based approach. Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), 1046–1056. [https://doi.org/10.18653/v1/N18-1095](https://doi.org/10.18653/v1/N18-1095)
* Notes: This version of the dictionary is the "extended" version from the research listed above. This dictionary version is not sensitive to parts of speech.


### Affective Norms (Warriner)

*3 ratings, 13,904 terms.* Valence, arousal and dominance for nearly 14,000 English lemmas. The standard affective norms.

* Warriner, A. B., Kuperman, V., & Brysbaert, M. (2013). Norms of valence, arousal, and dominance for 13,915 English lemmas. Behavior Research Methods, 45(4), 1191–1207. [https://doi.org/10.3758/s13428-012-0314-x](https://doi.org/10.3758/s13428-012-0314-x)


### Concreteness (Brysbaert)

*1 rating, 37,058 terms.* How concrete or abstract a word is, on a 1–5 scale, for 40,000 lemmas. Probably the single most-used norm set in psycholinguistics.

* Brysbaert, M., Warriner, A. B., & Kuperman, V. (2014). Concreteness ratings for 40 thousand generally known English word lemmas. Behavior Research Methods, 46(3), 904–911. [https://doi.org/10.3758/s13428-013-0403-5](https://doi.org/10.3758/s13428-013-0403-5)


### DIC-LSA

*5 ratings, 17,350 terms.* Norms bootstrapped out to a larger vocabulary using word-similarity indexes.

* Bestgen, Y., & Vincze, N. (2012). Checking and bootstrapping lexical norms by means of word similarity indexes. Behavior Research Methods, 44(4), 998–1006. [https://doi.org/10.3758/s13428-012-0195-z](https://doi.org/10.3758/s13428-012-0195-z)


### Embodiment

*2 ratings, 687 terms.* How strongly a word is grounded in bodily experience.

* [https://psyc.ucalgary.ca/languageprocessing/node/22](https://psyc.ucalgary.ca/languageprocessing/node/22)


### Gender Norms (Lewis)

*8 ratings, 2,327 terms.* How gendered a word is, estimated from children's books.

* [PREPRINT] Lewis, M., Borkenhagen, M. C., Converse, E., Lupyan, G., & Seidenberg, M. S. (2020, March 30). What might books be teaching young children about gender?. [https://doi.org/10.31234/osf.io/ntgfe](https://doi.org/10.31234/osf.io/ntgfe)


### Humor

*10 ratings, 4,997 terms.* How funny a word is on its own, rated for nearly 5,000 words. Yes, really.

* Engelthaler, T., & Hills, T. T. (2018). Humor norms for 4,997 English words. Behavior Research Methods, 50(3), 1116–1124. [https://doi.org/10.3758/s13428-017-0930-6](https://doi.org/10.3758/s13428-017-0930-6)


### LabMT

*7 ratings, 10,222 terms.* The happiness ratings behind hedonometrics, from the Twitter and Google Books word lists.

* Dodds, P. S., Harris, K. D., Kloumann, I. M., Bliss, C. A., & Danforth, C. M. (2011). Temporal Patterns of Happiness and Information in a Global Social Network: Hedonometrics and Twitter. PLOS ONE, 6(12), e26752. [https://doi.org/10.1371/journal.pone.0026752](https://doi.org/10.1371/journal.pone.0026752)


### Lancaster Sensorimotor

*33 ratings, 39,707 terms.* How strongly a word is experienced through each of
six senses and five body parts, for 40,000 words. The largest table here by a
distance.

Taken from the authors' own workbook, values unchanged. Six of their columns
are **not** included: `N_Known_*`, `List_N_*` and `Mean_Age_*` describe the
norming study rather than the word -- `Mean_Age_Perceptual`, for instance, is
the average age of the participants who happened to rate that item list.
Averaged over a text those produce a number that means nothing, and a
prediction model would happily fit it.

* Lynott, D., Connell, L., Brysbaert, M., Brand, J., & Carney, J. (2020). The Lancaster Sensorimotor Norms: Multidimensional measures of perceptual and action strength for 40,000 English words. *Behavior Research Methods, 52*(3), 1271–1291. [https://doi.org/10.3758/s13428-019-01316-z](https://doi.org/10.3758/s13428-019-01316-z)

What each column is, in the authors' words:

* Auditory_M: Auditory strength: mean rating (0–5) of how strongly the concept is experienced by hearing
* Gustatory_M: Gustatory strength: mean rating (0–5) of how strongly the concept is experienced by tasting
* Haptic_M: Haptic strength: mean rating (0–5) of how strongly the concept is experienced by feeling through touch
* Interoceptive_M: Interoceptive strength: mean rating (0–5) of how strongly the concept is experienced by sensations inside the body
* Olfactory_M: Olfactory strength: mean rating (0–5) of how strongly the concept is experienced by smelling
* Visual_M: Visual strength: mean rating (0–5) of how strongly the concept is experienced by seeing
* Foot_leg_M: Foot action strength: mean rating (0–5) of how strongly the concept is experienced by performaing an action with the foot / leg
* Hand_arm_M: Hand action strength: mean rating (0–5) of how strongly the concept is experienced by performaing an action with the hand / arm
* Head_M: Head action strength: mean rating (0–5) of how strongly the concept is experienced by performaing an action with the head excluding mouth
* Mouth_M: Mouth action strength: mean rating (0–5) of how strongly the concept is experienced by performaing an action with the mouth / throat
* Torso_M: Torso action strength: mean rating (0–5) of how strongly the concept is experienced by performaing an action with the torso
* Auditory_MSD: Mean of the Standard Deviations of auditory strength ratings
* Gustatory_MSD: Mean of the Standard Deviations of gustatory strength ratings
* Haptic_MSD: Mean of the Standard Deviations of haptic strength ratings
* Interoceptive_MSD: Mean of the Standard Deviations of interoceptive strength ratings
* Olfactory_MSD: Mean of the Standard Deviations of olfactory strength ratings
* Visual_MSD: Mean of the Standard Deviations of visual strength ratings
* Foot_leg_MSD: Mean of the Standard Deviations of foot action strength ratings
* Hand_arm_MSD: Mean of the Standard Deviations of hand action strength ratings
* Head_MSD: Mean of the Standard Deviations of head action strength ratings
* Mouth_MSD: Mean of the Standard Deviations of mouth action strength ratings
* Torso_MSD: Mean of the Standard Deviations of torso action strength ratings
* MaxStrength_Perceptual: Perceptual strength in the dominant modality (i.e., highest strength rating across six perceptual modalities)
* Minkowski3_Perceptual: Aggregated perceptual strength in all modalities where the influence of weaker modalities is attenuated, calculated as Minkowski distance (with exponent 3) of the 6-dimension vector of perceptual strength from the origin.
* Exclusivity_Perceptual: Modality exclusivity of the concept; the extent to which a concept is experienced though a single perceptual modality (0–1, typically expressed as %), calculated as the range of perceptual strength values divided by their sum
* MaxStrength_Action: Action strength in the dominant effector (i.e., highest strength rating across five action effectors)
* Minkowski3_Action: Aggregated action strength in all effectors where the influence of weaker effectors is attenuated, calculated as Minkowski distance (with exponent 3) of the 5-dimension vector of action strength from the origin.
* Exclusivity_Action: Effector exclusivity of the concept; the extent to which a concept is experienced though a single action effector (0–1, typically expressed as %), calculated as the range of action strength values divided by their sum
* MaxStrength_Sensorimotor: Sensorimotor strength in the dominant dimension (i.e., highest strength rating across 11 sensorimotor dimensions)
* Minkowski3_Sensorimotor: Aggregated sensorimotor strength in all dimensions where the influence of weaker dimensions is attenuated, calculated as Minkowski distance (with exponent 3) of the 11-dimension vector of sensorimotor strength from the origin.
* Exclusivity_Sensorimotor: Sensorimotor exclusivity of the concept; the extent to which a concept is experienced though a single sensorimotor dimension (0–1, typically expressed as %), calculated as the range of sensorimotor strength values divided by their sum
* Pct_Known_Perceptual: Percentage of participants (0–1) in perceptual strength norming who knew the concept well enough to provide valid ratings
* Pct_Known_Action: Percentage of participants (0–1) in action strength norming who knew the concept well enough to provide valid ratings

### Mental-Physical Verb

*1 rating, 250 terms.* Whether a verb describes a mental or a physical act — a measure of mental state attribution.

* Orr, R. I., & Gilead, M. (2021). Supplemental materials for development and validation of the mental-physical verb norms (MPVN): A text analysis measure of mental state attribution. [https://osf.io/5cez7/](https://osf.io/5cez7/)


### Modality

*6 ratings, 821 terms.* Which sense a word belongs to, for object properties and nouns.

* Lynott, D., & Connell, L. (2009). Modality exclusivity norms for 423 object properties. Behavior Research Methods, 41(2), 558–564. [https://doi.org/10.3758/BRM.41.2.558](https://doi.org/10.3758/BRM.41.2.558)
* Lynott, D., & Connell, L. (2013). Modality exclusivity norms for 400 nouns: The relationship between perceptual experience and surface word form. Behavior Research Methods, 45(2), 516–526. [https://doi.org/10.3758/s13428-012-0267-0](https://doi.org/10.3758/s13428-012-0267-0)


### Psycholinguistic Features

*4 ratings, 85,861 terms.* Familiarity, age of acquisition, concreteness and imagery, inferred out to a very large vocabulary with word embeddings.

* Paetzold, G., & Specia, L. (2016). Inferring Psycholinguistic Properties of Words. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (pp. 435–440). San Diego, California: Association for Computational Linguistics. [https://doi.org/10.18653/v1/N16-1050](https://doi.org/10.18653/v1/N16-1050)


### Stereotype Content

*43 ratings, 14,449 terms.* The stereotype content dimensions — sociability, morality, ability, agency, status and more — built semi-automatically from WordNet and word embeddings.

Two kinds of column, and they read differently. The `_Freq` ones are membership: every word in the table is marked 1 or 0 for each dimension, so the score is the **share of the rated words that belong to that dimension**. The `_Direction` ones are −1, 0 or +1 and are blank for words outside that dimension, so the score is the **average direction of the words that were in it** — a text can be full of morality words that are all negative. The three valence columns are proportions that sum to 1.

Taken straight from the authors' own release (`Full Dictionaries.csv` on OSF), values unchanged.

* Nicolas, G., Bai, X., & Fiske, S. T. (2021). Comprehensive stereotype content dictionaries using a semi-automated method. *European Journal of Social Psychology, 51*(1), 178–196. [https://doi.org/10.1002/ejsp.2724](https://doi.org/10.1002/ejsp.2724)
* Nicolas, G., Bai, X., & Fiske, S. T. (2019). Automated dictionary creation for analyzing text: An illustration from stereotype content [Preprint]. [https://doi.org/10.31234/osf.io/afm8k](https://doi.org/10.31234/osf.io/afm8k)


### Tabooness

*42 ratings, 460 terms.* How taboo a word is, with separate norms from male and female raters.

* Janschewitz, K. (2008). Taboo, emotionally valenced, and emotionally neutral word norms. Behavior Research Methods, 40(4), 1065–1074. [https://doi.org/10.3758/BRM.40.4.1065](https://doi.org/10.3758/BRM.40.4.1065)
* Scores texts using the entire set of participant norms, plus the male- and female-derived norms separately. Within the variable names, "M" stands for the Mean from the norms, and "MSD" stands for the Mean of the norm's Standard Deviation.

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
set as a concept dictionary on a saved word-vector model (Settings → Manage
saved models → the model's apply settings), where its categories seed a
search through the vector space your own corpus produced. Training the model
and scoring texts against concepts are separate acts: the dictionaries belong
to the second, and can be changed without retraining.
Archetype dictionaries cannot: they are seed *sentences* for an embedding
model, not term lists, and are only offered where archetypes are.
