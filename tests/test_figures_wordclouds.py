"""
Tests for `taters.figures.wordclouds`: which words go into which cloud.

The choosing is pure -- functions over lists of rows, the shape the CSVs
have -- so most tests hand in a few hand-written rows and check the specs
that come back: which model was used, which sign went where, what the
threshold kept, what the picture is called. The integration tests at the end
fit a real ridge on a two-set table and check the files and the report
section, and skip without Pillow.
"""

from __future__ import annotations

import pytest

from taters.figures import wordclouds as wc

pytest.importorskip("numpy")


def _allow_all(_fs, _pred):
    return True


# ---------------------------------------------------------------------------
# Ridge and classifier
# ---------------------------------------------------------------------------

def _ridge_rows():
    """Language alone (coefficients halved), controls alone, language with
    both controls (the full model, the true coefficients) and language with
    one control (doubled) -- so the picture shows which model was picked."""
    rows = []
    for model, controls in (("language", ""), ("controls", "age|gender"),
                            ("controls+language", "age|gender"),
                            ("controls+language", "age")):
        for pred, coef in (("happy", "0.4"), ("sad", "-0.3"), ("age", "0.9"),
                           ("gender=female", "0.2")):
            scale = {("language", ""): 0.5, ("controls+language", "age"): 2.0
                     }.get((model, controls), 1.0)
            rows.append({"feature_set": "dictionary", "model": model,
                         "controls": controls, "predictor": pred,
                         "openness": str(float(coef) * scale)})
    return rows


def test_ridge_clouds_come_from_the_full_model_only():
    """
    Fitters write every control combination. The cloud shows language fitted
    beside every control -- the language's own contribution -- and not the
    language-alone model or a partial control set, so the same predictor is
    never counted from three models at once.
    """
    specs = wc.clouds_for_ridge(_ridge_rows(), _allow_all)
    pos = next(s for s in specs if s.name.endswith("__positive"))
    weights = dict(pos.words)
    assert weights["happy"] == 0.4          # the age|gender model, not the doubled one
    assert len([w for w in pos.words if w[0] == "happy"]) == 1


def test_ridge_clouds_split_by_sign_and_leave_controls_out():
    """Two pictures per outcome: what predicts more and what predicts less.
    `age` is a control, not language, and is not drawn however large its
    coefficient."""
    sets = {"dictionary": ["happy", "sad"]}
    allowed = wc._feature_filter(sets, {})
    specs = wc.clouds_for_ridge(_ridge_rows(), allowed)
    assert [s.name for s in specs] == ["openness__positive", "openness__negative"]
    assert {s.folder for s in specs} == {"ridge-regression/dictionary"}
    assert dict(specs[0].words) == {"happy": 0.4}
    assert dict(specs[1].words) == {"sad": -0.3}
    assert "openness" in specs[0].title and specs[0].analysis == "Ridge regression"


def test_a_component_counts_as_a_feature_of_its_set():
    """After a reduction the predictors are `dictionary_Component_1`, which no
    set lists as a column; the eigenvalues table says whose it is."""
    allowed = wc._feature_filter({"dictionary": ["happy"]},
                                 {"dictionary": ["dictionary_Component_1"]})
    assert allowed("dictionary", "dictionary_Component_1")
    assert not allowed("dictionary", "age")
    assert allowed("all", "happy")


def test_an_empty_side_becomes_a_sentence_not_a_missing_picture():
    rows = [{"feature_set": "s", "model": "language", "controls": "",
             "predictor": "happy", "openness": "0.5"}]
    specs = wc.clouds_for_ridge(rows, _allow_all)
    neg = next(s for s in specs if s.name.endswith("__negative"))
    assert not neg.words and "no feature had a negative coefficient" in neg.note


def test_classifier_clouds_are_per_class_and_per_direction():
    rows = [{"feature_set": "s", "outcome": "condition", "model": "language",
             "controls": "", "class": klass, "predictor": pred, "coef": coef}
            for klass in ("A", "B")
            for pred, coef in (("happy", "0.4"), ("sad", "-0.2"))]
    specs = wc.clouds_for_classifier(rows, _allow_all)
    names = [s.name for s in specs]
    assert "condition__A__positive" in names and "condition__B__negative" in names
    toward_a = next(s for s in specs if s.name == "condition__A__positive")
    assert toward_a.folder == "classification/s"
    assert dict(toward_a.words) == {"happy": 0.4}
    assert "toward" in toward_a.title and "A" in toward_a.title


def test_the_split_column_names_its_half_in_title_and_file():
    """A split analysis writes one block per level of the split column,
    which comes first in the table; each half gets its own clouds."""
    rows = [{"site": site, "feature_set": "s", "model": "language",
             "controls": "", "predictor": "happy", "openness": "0.5"}
            for site in ("north", "south")]
    specs = wc.clouds_for_ridge(rows, _allow_all)
    pos = [s for s in specs if s.words]
    assert {s.name for s in pos} == {"openness__site-north__positive",
                                     "openness__site-south__positive"}
    assert "[site = north]" in pos[0].title


# ---------------------------------------------------------------------------
# Correlations, group differences, components
# ---------------------------------------------------------------------------

def test_correlation_clouds_keep_only_what_passes_the_adjusted_threshold():
    """The adjusted p decides when the table has one; a strong r with a weak
    adjusted p is not drawn, and the legend says the threshold."""
    rows = [{"feature_set": "s", "feature": f, "openness_r": r,
             "openness_p": p, "openness_p_adj": padj, "openness_n": "30"}
            for f, r, p, padj in (("happy", "0.6", "0.001", "0.01"),
                                  ("sad", "-0.5", "0.002", "0.02"),
                                  ("meh", "0.7", "0.03", "0.20"))]
    specs = wc.clouds_for_correlations(rows, max_p=0.05, method="pearson")
    pos = next(s for s in specs if s.name.endswith("__positive"))
    neg = next(s for s in specs if s.name.endswith("__negative"))
    assert dict(pos.words) == {"happy": 0.6}
    assert dict(neg.words) == {"sad": -0.5}
    assert "(adjusted)" in pos.legend and "0.05" in pos.legend


def test_correlation_clouds_fall_back_to_the_raw_p_when_nothing_was_adjusted():
    rows = [{"feature_set": "s", "feature": "happy", "openness_r": "0.6",
             "openness_p": "0.2", "openness_n": "30"}]
    specs = wc.clouds_for_correlations(rows, max_p=0.05)
    assert all(not s.words for s in specs)
    assert "(adjusted)" not in specs[0].legend
    assert "no feature was positively correlated" in specs[0].note


def test_group_difference_clouds_say_which_group_is_higher():
    rows = [{"feature_set": "s", "feature": f, "group_1": "A", "group_2": "B",
             "n_1": "10", "n_2": "10", "mean_diff": "1", "ci_low": "0",
             "ci_high": "2", "d": d, "p_adj": p, "method": "tukey_hsd"}
            for f, d, p in (("happy", "0.8", "0.01"), ("sad", "-0.6", "0.02"),
                            ("meh", "0.9", "0.30"))]
    specs = wc.clouds_for_group_differences(rows, max_p=0.05)
    assert [s.name for s in specs] == ["A_vs_B__higher_in_A", "A_vs_B__higher_in_B"]
    assert specs[0].folder == "group-differences/s"
    assert dict(specs[0].words) == {"happy": 0.8}
    assert dict(specs[1].words) == {"sad": -0.6}
    assert "higher in B" in specs[1].title and "Red" in specs[1].legend


def test_component_clouds_take_the_strongest_loadings_of_both_signs():
    rows = [{"feature_set": "dictionary", "feature": f,
             "dictionary_Component_1": c1, "dictionary_Component_2": c2}
            for f, c1, c2 in (("happy", "0.9", "0.1"), ("sad", "-0.8", "0.2"),
                              ("meh", "0.05", "0.7"), ("dictionary.WC", "0.3", ""))]
    specs = wc.clouds_for_components(rows, analysis="ridge", component_words=2)
    assert [s.name for s in specs] == ["dictionary_Component_1",
                                       "dictionary_Component_2"]
    assert {s.folder for s in specs} == {"ridge-regression/dictionary/components"}
    assert specs[0].analysis == "Ridge regression" and specs[0].kind == "component"
    assert dict(specs[0].words) == {"happy": 0.9, "sad": -0.8}
    # the collision prefix comes off the label
    labels = {w for w, _ in wc.clouds_for_components(
        rows, analysis="ridge", component_words=10)[0].words}
    assert "WC" in labels and "dictionary.WC" not in labels


def test_frequency_by_group_sums_the_matrix_columns_per_level():
    """The analysis table joins the matrix and the group column, so a
    group's cloud is a column sum over its rows -- nothing re-tokenized."""
    table = [{"text_id": "a", "condition": "X", "hello": "3", "world": "1"},
             {"text_id": "b", "condition": "X", "hello": "2", "world": ""},
             {"text_id": "c", "condition": "Y", "hello": "", "world": "4"}]
    sets = {"doc_term_matrix_count": ["hello", "world"], "readability": ["fk"]}
    specs = wc.clouds_for_frequencies_by_group(table, sets, group_col="condition",
                                               max_words=10)
    assert [s.name for s in specs] == ["condition-X", "condition-Y"]
    assert specs[0].folder == "most-frequent-terms-by-group/doc_term_matrix_count"
    assert specs[0].words == [("hello", 5.0), ("world", 1.0)]
    assert specs[1].words == [("world", 4.0)]
    assert not wc.clouds_for_frequencies_by_group(table, sets, group_col="",
                                                  max_words=10)


# ---------------------------------------------------------------------------
# Themes and frequencies
# ---------------------------------------------------------------------------

def test_theme_clouds_keep_the_tag_on_a_tagged_term():
    rows = [{"term": "felt", "pos": "VBD", "Theme_1": "0.7", "Theme_2": "0.0"},
            {"term": "happy", "pos": "JJ", "Theme_1": "-0.4", "Theme_2": "0.6"}]
    specs = wc.clouds_for_themes(rows, top_words=5, shares={"Theme_1": 12.5})
    assert [s.name for s in specs] == ["Theme_1", "Theme_2"]
    assert dict(specs[0].words) == {"felt (VBD)": 0.7, "happy (JJ)": -0.4}
    assert "12.5%" in specs[0].title
    assert dict(specs[1].words) == {"happy (JJ)": 0.6}


def test_theme_clouds_respect_the_loading_floor_and_top_n():
    rows = [{"term": f"w{i}", "Theme_1": str(0.9 - 0.1 * i)} for i in range(9)]
    specs = wc.clouds_for_themes(rows, top_words=3, min_abs_loading=0.5)
    assert [w for w, _ in specs[0].words] == ["w0", "w1", "w2"]
    specs = wc.clouds_for_themes(rows, top_words=30, min_abs_loading=0.85)
    assert [w for w, _ in specs[0].words] == ["w0"]


def test_redundant_phrases_are_skipped_unless_they_bring_a_new_word():
    """"of the" beside "of" and "the" says nothing new; "going to be" brings
    "going", which the cloud does not have, and stays."""
    phrases = [("the", 100.0), ("of", 80.0), ("of the", 40.0), ("to", 70.0),
               ("going to be", 4.0), ("be", 60.0)]
    kept = [t for t, _ in wc.skip_redundant(phrases)]
    assert "of the" not in kept
    assert "going to be" in kept
    assert kept[:4] == ["the", "of", "to", "be"]


def test_the_corpus_cloud_reads_the_frequency_list():
    rows = [{"rank": "1", "ngram": "the", "phrase_length": "1", "frequency": "100"},
            {"rank": "2", "ngram": "of", "phrase_length": "1", "frequency": "50"},
            {"rank": "3", "ngram": "of the", "phrase_length": "2", "frequency": "40"},
            {"rank": "4", "ngram": "potato", "phrase_length": "1", "frequency": "30"}]
    (spec,) = wc.clouds_for_frequencies(rows, top_words=10)
    assert spec.words == [("the", 100.0), ("of", 50.0), ("potato", 30.0)]
    assert spec.name == "top_words" and "3 most frequent" in spec.title


# ---------------------------------------------------------------------------
# End to end, with a real fit and real pictures
# ---------------------------------------------------------------------------

def _fitted(tmp_path, pca="all"):
    pytest.importorskip("scipy")
    from test_stats_pca import _two_sets
    from taters.stats.ridge import fit_ridge_csv
    from taters.stats.correlations import analyze_correlations

    table, _dic, _read = _two_sets(tmp_path)
    out = tmp_path / "stats"
    fit_ridge_csv(table_csv=table, outcome_cols=["outcome"], pca=pca,
                  feature_sets="per_table", out_dir=out, verbose=False)
    analyze_correlations(table_csv=table, outcome_cols=["outcome"],
                         feature_sets="per_table", out_dir=out, verbose=False)
    # in an actual run the analysis table and its sidecar sit beside the
    # results, so we put them there too
    (out / "analysis_table.csv").write_bytes(table.read_bytes())
    (out / "analysis_table_sets.json").write_bytes(
        (tmp_path / "analysis_table_sets.json").read_bytes())
    return out


def test_stats_wordclouds_draws_the_pictures_and_writes_the_section(tmp_path):
    pytest.importorskip("PIL")
    out = _fitted(tmp_path)
    folder = wc.stats_wordclouds(out, verbose=False)
    pngs = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*.png"))
    assert "ridge-regression/dictionary/outcome__positive.png" in pngs
    assert any(p.startswith("ridge-regression/dictionary/components/") for p in pngs)
    assert any(p.startswith("correlations/") for p in pngs)
    section = (out / "_sections" / "60-word-clouds.md").read_text(encoding="utf-8")
    assert section.startswith("## Word clouds")
    assert "![" in section and "figures/wordclouds/ridge-regression/dictionary/" in section
    assert "#### dictionary" in section and "#### readability" in section
    assert "What each component of dictionary is made of" in section


def test_a_second_run_rewrites_nothing(tmp_path):
    """The resume contract, for pictures: unchanged tables, unchanged files
    -- the harness asserts a re-run touches nothing."""
    pytest.importorskip("PIL")
    out = _fitted(tmp_path)
    folder = wc.stats_wordclouds(out, verbose=False)
    before = {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()}
    wc.stats_wordclouds(out, verbose=False)
    after = {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()}
    assert before == after
    assert folder.is_dir()


def test_turned_off_it_writes_nothing_at_all(tmp_path):
    out = _fitted(tmp_path)
    wc.stats_wordclouds(out, enabled=False, verbose=False)
    assert not (out / "figures").exists()
    assert not (out / "_sections" / "60-word-clouds.md").exists()


def test_the_report_carries_the_section_and_its_images(tmp_path):
    pytest.importorskip("PIL")
    from taters.stats.report import write_stats_report

    out = _fitted(tmp_path)
    wc.stats_wordclouds(out, verbose=False)
    report = write_stats_report(stats_dir=out, verbose=False).read_text(encoding="utf-8")
    assert report.index("## Word clouds") > report.index("## ")
    assert "figures/wordclouds/ridge-regression/dictionary/outcome__positive.png" in report


def test_theme_and_frequency_clouds_from_their_tables(tmp_path):
    pytest.importorskip("PIL")
    from csvhelpers import write_rows

    loadings = tmp_path / "topic_model_mem_loadings.csv"
    write_rows(loadings, ["term", "Theme_1", "Theme_2"],
               [["potato", "0.8", "-0.1"], ["gravy", "-0.5", "0.7"]])
    write_rows(tmp_path / "topic_model_mem_eigenvalues.csv",
               ["theme", "eigenvalue", "pct_variance"],
               [["Theme_1", "2", "40"], ["Theme_2", "1", "20"]])
    folder = wc.theme_wordclouds(loadings, verbose=False)
    assert folder == tmp_path / "figures" / "wordclouds" / "topic_model_mem"
    assert (folder / "Theme_1.png").is_file() and (folder / "Theme_2.png").is_file()
    index = (folder / "index.md").read_text(encoding="utf-8")
    assert "40.0% of variance" in index and "![" in index

    freq = tmp_path / "ngram_frequencies.csv"
    write_rows(freq, ["rank", "ngram", "phrase_length", "frequency"],
               [["1", "potato", "1", "12"], ["2", "gravy", "1", "7"]])
    folder = wc.frequency_wordclouds(freq, verbose=False)
    assert (folder / "top_words.png").is_file()


def test_a_set_made_of_themes_gets_each_theme_drawn_beside_its_clouds():
    """
    A ridge over topic-model themes shows "Theme_5" as a word, which says
    nothing until the theme's own words are in view (a real report: "I don't
    see a folder of word clouds that shows the words loading onto each MEM
    theme"). Every theme a set's clouds name gets its own picture in a
    `themes/` folder beside them, with where it appeared in the legend.
    """
    loadings = [{"term": "potato", "Theme_1": "0.8", "Theme_2": "-0.1"},
                {"term": "gravy", "Theme_1": "-0.5", "Theme_2": "0.7"},
                {"term": "butter", "Theme_1": "0.0", "Theme_2": "0.4"}]
    mentions = {"Theme_1": [("grit: β", -0.10), ("openness: β", 0.32)],
                "Theme_9": [("openness: β", 0.05)]}      # no such column, so skip it
    specs = wc.clouds_for_theme_predictors(
        loadings, mentions, analysis="Ridge regression",
        folder="ridge-regression", set_name="topic_model_mem", top_words=5)
    assert [s.name for s in specs] == ["Theme_1"]
    (spec,) = specs
    assert spec.folder == "ridge-regression/topic_model_mem/themes"
    assert spec.kind == "theme"
    assert dict(spec.words) == {"potato": 0.8, "gravy": -0.5}
    assert "openness: β +0.32; grit: β -0.10" in spec.legend, "strongest first"


def test_theme_pictures_land_in_the_run_and_the_loadings_are_found_by_name(tmp_path):
    """
    End to end on a two-set table where one set's features are "themes": a
    loadings table for the set sits under the pipeline folder, is found by
    name (the sidecar's recorded path may not resolve -- a Windows path read
    from WSL), and each theme the ridge names is drawn under the set's
    `themes/` folder and listed in the report section.
    """
    pytest.importorskip("PIL")
    from csvhelpers import write_rows

    out = _fitted(tmp_path, pca="off")      # no pca, so the features ARE the "themes"
    dic_cols = [f"dic_{i}" for i in range(10)]
    write_rows(tmp_path / "dictionary_loadings.csv", ["term"] + dic_cols,
               [[f"term{j}"] + [f"{(j - 5) / 10:.2f}" if (i + j) % 3 else "0"
                                for i in range(10)] for j in range(12)])
    folder = wc.stats_wordclouds(out, verbose=False)
    themes = sorted(q.name for q in (folder / "ridge-regression" / "dictionary"
                                     / "themes").glob("*.png"))
    assert themes and all(name.startswith("dic_") for name in themes)
    section = (out / "_sections" / "60-word-clouds.md").read_text(encoding="utf-8")
    assert "What the themes that appear above are made of" in section
    assert "ridge-regression/dictionary/themes/" in section


def test_a_combination_of_tables_gets_no_cloud_of_its_own():
    """Its members have theirs; a third picture of the same words would say
    nothing new and double the folder."""
    rows = [{"feature_set": fs, "model": "language", "controls": "",
             "predictor": "happy", "openness": "0.5"}
            for fs in ("dict", "mem", "dict+mem", "all")]
    specs = wc.clouds_for_ridge(rows, _allow_all)
    assert {s.set_name for s in specs} == {"dict", "mem", "all"}


def test_neighbor_clouds_are_one_per_probe_sized_by_similarity():
    """From a word-vector model's neighbors table: the probe the model
    does not know gets a sentence, not an empty picture."""
    rows = [{"probe": "cat", "rank": "1", "word": "dog", "similarity": "0.9"},
            {"probe": "cat", "rank": "2", "word": "fish", "similarity": "0.4"},
            {"probe": "cat", "rank": "3", "word": "car", "similarity": "-0.2"},
            {"probe": "zebra", "rank": "", "word": "", "similarity": ""}]
    specs = wc.clouds_for_neighbors(rows, top_words=5)
    assert [s.name for s in specs] == ["cat", "zebra"]
    assert specs[0].words == [("dog", 0.9), ("fish", 0.4)], "negative similarity is not closeness"
    assert "cat" in specs[0].title and specs[0].analysis == "Nearest neighbors"
    assert specs[1].words == [] and "not in the model's vocabulary" in specs[1].note
    assert [s.name for s in wc.clouds_for_neighbors(rows, top_words=1)] == ["cat", "zebra"]
    assert wc.clouds_for_neighbors(rows, top_words=1)[0].words == [("dog", 0.9)]


def test_neighbor_pictures_and_index_land_beside_the_model(tmp_path):
    pytest.importorskip("PIL")
    from csvhelpers import write_rows

    table = write_rows(tmp_path / "features" / "models" / "wv_neighbors.csv",
                       ["probe", "rank", "word", "similarity"],
                       [["cat", "1", "dog", "0.9"], ["cat", "2", "fish", "0.4"],
                        ["zebra", "", "", ""]])
    folder = wc.neighbor_wordclouds(table, verbose=False)
    assert folder == tmp_path / "features" / wc.FIGURES_DIR / "word_vectors"
    assert (folder / "cat.png").is_file() and not (folder / "zebra.png").exists()
    index = (folder / "index.md").read_text(encoding="utf-8")
    assert "![Words closest to “cat” in the model](cat.png)" in index
    assert "“zebra” is not in the model's vocabulary." in index
    off = wc.neighbor_wordclouds(table, tmp_path / "off", enabled=False, verbose=False)
    assert not off.exists()
