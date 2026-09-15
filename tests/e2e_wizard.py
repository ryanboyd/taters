"""
Drive the real product end to end: wizard -> preset -> run, on synthetic data.

    python tests/e2e_wizard.py                       # every text flow
    python tests/e2e_wizard.py --media clip.mkv      # plus a media flow
    python tests/e2e_wizard.py --flows A F --keep    # some flows, keep outputs

Not a pytest file (no ``test_`` prefix): it takes minutes, it needs the
tokenizer data downloaded, and its value is precisely that it is *not* a unit
test -- it answers the wizard's questions the way a person would, composes
the pipeline the wizard composes, runs it with the real analyzers, re-runs it
from the command line, imports the model it fitted and scores a second study
with it. On 2026-09-05 this found five gaps that three thousand unit tests
had missed, all in the "score a second study with a saved model" direction.
Kept in the repo, like ``mutcheck.py``, because temp folders do not survive
between sessions.

Every flow works in a fresh scratch home (``TATERS_HOME``) so the library is
clean, and a fresh project folder, because finished outputs short-circuit and
a stale run hides regressions. The ``__main__`` guard is load-bearing: the
analyzers use multiprocessing *spawn*, which re-imports this file in every
worker and, unguarded, re-ran the whole harness inside each one.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

WORDS = ("the study found that participants who reported higher stress also "
         "slept less and felt more tired during the day because their "
         "routines were disrupted although some said exercise helped them "
         "recover quickly and this pattern held across the sample").split()
WORDS_TWO = ("the survey showed that students who enjoyed music also read "
             "more and felt calm because evenings were quiet although "
             "homework kept them busy and this held across the term").split()


def _text(rng, n, words):
    return " ".join(rng.choice(words) for _ in range(n)).capitalize() + "."


def make_study(path, *, n=160, cols=("text",), words=WORDS, seed=7):
    """Three conditions whose texts differ in length, an outcome tied to
    length, and two covariates -- enough for every analysis to find
    something."""
    rng = random.Random(seed)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pid", "condition", "openness", "age", "gender", *cols])
        for i in range(n):
            cond = "ABC"[i % 3]
            k = 25 + 10 * "ABC".index(cond) + rng.randrange(15)
            w.writerow([f"p{i}", cond, f"{2 + 0.05 * k + rng.gauss(0, .6):.3f}",
                        rng.randrange(18, 70), rng.choice(["f", "m"]),
                        *[_text(rng, k, words) for _ in cols]])
    return path


class Harness:
    def __init__(self, work: Path, *, keep: bool):
        self.work = work
        self.proj = work / "project"
        self.proj.mkdir(parents=True, exist_ok=True)
        self.keep = keep
        self.failures: list = []
        os.environ["TATERS_HOME"] = str(work / "home")
        os.chdir(self.proj)
        sys.path.insert(0, str(HERE))
        from taters.ui.browse import _TYPE
        self.browse_type = _TYPE
        self.study = make_study(self.proj / "study.csv")
        self.study2 = make_study(self.proj / "study2.csv", n=60,
                                 words=WORDS_TWO, seed=8)
        self.study_two = make_study(self.proj / "study_two_texts.csv",
                                    cols=("text", "text2"))
        essays = self.proj / "essays"
        essays.mkdir(exist_ok=True)
        rng = random.Random(3)
        for i in range(8):
            (essays / f"essay{i}.txt").write_text(
                _text(rng, 60, WORDS) + "\n\n" + _text(rng, 40, WORDS),
                encoding="utf-8")
        self.essays = essays

    # -------------------------------------------------------------- output
    def ok(self, name, detail=""):
        print(f"ok {name}{': ' + detail if detail else ''}", flush=True)

    def bad(self, name, detail):
        self.failures.append(name)
        print(f"!! {name}: {detail}", flush=True)

    def browse(self, path):
        return [self.browse_type, str(path)]

    # -------------------------------------------------------------- a flow
    def flow(self, name, answers, **kw):
        """Answer the wizard, run what it composed, report."""
        from taters.pipelines.run_pipeline import run_preset
        from taters.ui import wizard as wiz
        from taters.ui.prompts import ScriptedPrompter

        p = ScriptedPrompter(list(answers))
        try:
            res = wiz.run_wizard(p, cwd=self.proj, **kw)
        except Exception as e:
            self.bad(name, f"wizard failed: {type(e).__name__}: {e}")
            for k, q in p.asked:
                print("    asked:", k, q)
            traceback.print_exc(limit=2)
            return None
        preset = res.preset
        folder = self.proj / preset["meta"]["id"]
        # a media pipeline has per-file steps, so it needs to know which folder
        # they fan out over; the wizard's result carries that, same as the app's
        # own runner does.
        man = run_preset(preset, workers=2, work_dir=folder,
                         root_dir=res.root_dir, file_type=res.file_type,
                         out_manifest=folder / "run_manifest.json",
                         verbose=False)
        errs = man.get("errors") or []
        if errs:
            self.bad(name, f"{len(preset['steps'])} steps -> {folder.name}")
            for e in errs:
                print("    ", e)
        else:
            self.ok(name, f"{len(preset['steps'])} steps -> {folder.name}")
        rep = folder / "stats_results" / "report.md"
        if rep.exists():
            heads = [ln for ln in rep.read_text(encoding="utf-8").splitlines()
                     if ln.startswith("## ")]
            print("     report:", " | ".join(heads)[:300])
        return folder, preset

    # -------------------------------------------------------------- flows
    def flow_a(self):
        wc = "\x00word_count"
        return self.flow("A spreadsheet, every text feature, four analyses", [
            "csv", *self.browse(self.study), ["text"], True, ["pid"],
            ["readability", "lexical_richness", "cohesion", "parts_of_speech",
             "ngram_frequencies", "doc_term_matrix", "topic_model_mem"],
            "row",
            ["stats_group_differences", "stats_correlations",
             "stats_ridge_fit", "stats_classify_fit"],
            "condition", ["openness"], ["condition"],
            True, ["age", "gender"],        # gender holds words, so labels already
            ["readability", "lexical_richness", "cohesion", "parts_of_speech",
             "doc_term_matrix", "topic_model_mem"], "together",
            "fdr_bh",
            True, ["generated"], [wc], ">=", "20",
            ":done", "Flow A", "save",
        ], analyses=True)

    def flow_b(self):
        return self.flow("B grouped rows + correlations", [
            "csv", *self.browse(self.study), ["text"], False,
            ["readability"], "group", ["condition"],
            ["stats_correlations"], ["openness"], False, "fdr_bh", False,
            ":done", "Flow B", "save",
        ])

    def flow_c(self):
        return self.flow("C two text columns measured separately", [
            "csv", *self.browse(self.study_two), ["text", "text2"],
            "separate", True, ["pid"],
            ["readability", "lexical_richness"], "row",
            ["stats_correlations", "stats_ridge_fit"], ["openness"], False,
            ["readability", "lexical_richness"], "together", "fdr_bh", False,
            ":done", "Flow C", "save",
        ])

    def flow_d(self):
        return self.flow("D folder of documents", [
            "txt_dir", *self.browse(self.essays),
            ["readability", "lexical_richness", "ngram_frequencies"], False,
            "Flow D", "save",
        ])

    def flow_e(self, runs):
        """Every saved pipeline re-runs from the command line and rewrites
        nothing."""
        for name, r in runs.items():
            if not r:
                continue
            folder, preset = r
            yaml_path = folder / f"{preset['meta']['id']}.yaml"
            # we watch tables, pictures and report fragments alike: a re-run
            # that redraws a word cloud is just as much a broken resume as one
            # that recomputes a table.
            kept = ("*.csv", "*.png", "*.md")
            before = {q: q.stat().st_mtime for g in kept for q in folder.rglob(g)}
            cp = subprocess.run(
                [sys.executable, "-m", "taters.pipelines.run_pipeline",
                 "--preset-file", str(yaml_path), "--quiet"],
                cwd=folder, capture_output=True, text=True, timeout=1800)
            after = {q: q.stat().st_mtime for g in kept for q in folder.rglob(g)}
            rewritten = [str(q.relative_to(folder)) for q in before
                         if after.get(q) != before[q]]
            label = f"E re-run {name}"
            if cp.returncode == 0 and not rewritten:
                self.ok(label, "rc=0, nothing rewritten")
            else:
                self.bad(label, f"rc={cp.returncode} rewritten={rewritten[:5]}")
                print(cp.stderr[-800:])

    def flow_g(self, a):
        """Flow A drew word clouds for every result and every theme, and the
        report shows them."""
        folder, _preset = a
        stats = folder / "stats_results" / "figures" / "wordclouds"
        feats = folder / "features" / "figures" / "wordclouds"
        stats_pngs = sorted(q.relative_to(stats).as_posix() for q in stats.rglob("*.png"))
        feat_pngs = sorted(q.relative_to(feats).as_posix() for q in feats.rglob("*.png"))
        want = ["ridge-regression/", "classification/",
                "most-frequent-terms-by-group/"]
        # we only expect component clouds if a reduction ran (flow A leaves pca off).
        if any((folder / "stats_results").glob("*_pca_loadings*.csv")):
            want.append("components/")
        missing = [w for w in want if not any(q.startswith(w) for q in stats_pngs)]
        themes = [q for q in feat_pngs if q.startswith("topic_model_mem/")]
        corpus = [q for q in feat_pngs if q == "ngram_frequencies/top_words.png"]
        report = (folder / "stats_results" / "report.md").read_text(encoding="utf-8")
        shown = "## Word clouds" in report and "figures/wordclouds/" in report
        described = sorted(q.name for q in (folder / "stats_descriptives").glob("*.csv"))
        if not described:
            missing.append("stats_descriptives/")
        if not missing and themes and corpus and shown:
            self.ok("G word clouds", f"{len(stats_pngs)} result clouds, "
                    f"{len(themes)} theme clouds, the corpus cloud, all in the "
                    f"report; {len(described)} tables described")
        else:
            self.bad("G word clouds", f"missing={missing} themes={len(themes)} "
                     f"corpus={bool(corpus)} in_report={shown}; "
                     f"stats={stats_pngs[:8]} features={feat_pngs[:8]}")

    def flow_f(self, a):
        """The model fitted in A, imported into the library, scores a second
        study whose vocabulary only partly overlaps."""
        from taters.helpers.library import import_into, kind_by_id

        from taters.helpers.library import kind_dir
        from taters.helpers.model_spec import rename_model

        model = a[0] / "stats_results" / "models" / "ridge__all__openness.json"
        classifier = a[0] / "stats_results" / "models" / "classifier__all__condition.json"
        if not model.exists() or not classifier.exists():
            self.bad("F", "A did not write both a ridge and a classifier")
            return None
        # two models, the way somebody's actual library would look: the ridge
        # and the classifier that A fitted, named the way the import screen
        # invites you to name them.
        ridge = import_into(kind_by_id("models"), model)
        rename_model(ridge, name="openness_ridge")
        clf = import_into(kind_by_id("models"), classifier)
        rename_model(clf, name="condition_clf")
        # and a third model nobody here trained: a (tiny, untrained) classifier
        # in the shape of one downloaded from the Hugging Face hub, imported the
        # way the Settings screen does it
        from tiny_encoder import build_classifier

        from taters.text.hf_classifier import import_hf_classifier

        tiny = build_classifier(self.proj / "tiny_clf", labels=("food", "work"))
        hf_manifest = import_hf_classifier(source=tiny, out_dir=self.proj / "hf_import",
                                           name="topic_clf", outcome="topic")
        hub = import_into(kind_by_id("models"), hf_manifest)
        models_dir = kind_dir(kind_by_id("models"))
        f = self.flow("F score a second study with three saved models", [
            "csv", *self.browse(self.study2), ["text"], True, ["pid"],
            ["readability", "lexical_richness", "cohesion", "parts_of_speech",
             "doc_term_matrix", "topic_model_mem", "score_with_model"], "row",
            # the wizard asks which models: we tick two, hit enter on the third
            f"\x00space:{(models_dir / ridge.name).resolve()}",
            f"\x00space:{(models_dir / clf.name).resolve()}",
            str((models_dir / hub.name).resolve()),
            [], ":done", "Flow F", "save",
        ])
        if f:
            scores = f[0] / "features" / "model_scores.csv"
            rows = list(csv.DictReader(scores.open(encoding="utf-8-sig"))) \
                if scores.exists() else []
            scored = sum(1 for r in rows if r.get("openness_ridge__pred_openness"))
            print(f"     model_scores: {len(rows)} rows, {scored} scored; "
                  f"columns: {list(rows[0]) if rows else None}")
            account = f[0] / "features" / "model_scores_unscored.csv"
            if account.exists():
                print("     unscored accounting:",
                      "; ".join(f"{r['model']}: {r['reason']} {r['detail']}={r['rows']}"
                                for r in csv.DictReader(
                                    account.open(encoding="utf-8-sig")))[:300])
            own = sorted(q.name for q in (f[0] / "features" / "model_scores").glob("*.csv")
                         if not q.stem.endswith("_unscored"))
            if not rows or not {"openness_ridge__pred_openness",
                                "condition_clf__pred_condition",
                                "topic_clf__pred_topic"} <= set(rows[0]) \
                    or own != ["condition_clf.csv", "openness_ridge.csv", "topic_clf.csv"]:
                self.bad("F columns", f"{list(rows[0]) if rows else None}; own files {own}")
            else:
                self.ok("F columns", f"{len(rows[0])} merged columns, own files {own}")
        return f

    def flow_h(self, a):
        """Module command lines, end to end."""
        table = a[0] / "stats_results" / "analysis_table.csv"
        out = self.proj / "cli_out"
        out.mkdir(exist_ok=True)
        model = a[0] / "stats_results" / "models" / "ridge__all__openness.json"
        cmds = [
            [sys.executable, "-m", "taters.text.analyze_readability",
             "--csv-path", str(self.study), "--text-col", "text",
             "--id-col", "pid", "--gathered-csv", str(out / "g.csv"),
             "--out-features-csv", str(out / "read.csv"),
             "--overwrite-existing"],
            [sys.executable, "-m", "taters.stats.correlations",
             "--table-csv", str(table), "--outcome-col", "openness",
             "--out-dir", str(out / "corr"), "--pca", "all"],
            [sys.executable, "-m", "taters.stats.ridge", "apply",
             "--model-json", str(model), "--input-csv", str(table),
             "--out", str(out / "scored.csv")],
            [sys.executable, "-m", "taters.stats.group_differences",
             "--table-csv", str(table), "--group-col", "condition",
             "--posthoc", "none", "--out-dir", str(out / "gd")],
        ]
        for c in cmds:
            cp = subprocess.run(c, capture_output=True, text=True, timeout=1800)
            label = f"H {c[2]} {c[3] if not c[3].startswith('--') else ''}".strip()
            if cp.returncode == 0:
                self.ok(label)
            else:
                self.bad(label, f"rc={cp.returncode}")
                print(cp.stderr[-600:])

    def flow_r(self, a):
        """"Run analyses" over a spreadsheet of numbers somebody already has.

        The realistic version of that: the analysis table flow A just built.
        It is exactly what a researcher arrives with -- an id, some
        conditions, an outcome, and a pile of measured columns -- and this
        flow treats those columns as the predictors without extracting
        anything."""
        import csv as _csv
        import json

        table = a[0] / "stats_results" / "analysis_table.csv"
        if not table.exists():
            self.bad("R", "A wrote no analysis table")
            return None
        with table.open(encoding="utf-8-sig", newline="") as fh:
            sample = list(_csv.DictReader(fh))[:40]
        header = list(sample[0]) if sample else []
        # only columns that really hold numbers: the picker offers those and
        # nothing else, the same test it makes. `pid` looks like a column and
        # is not a measure
        known = {"text_id", "pid", "condition", "openness", "age", "gender"}

        def numeric(col):
            values = [r.get(col) for r in sample]
            try:
                return all(float(v) == float(v) for v in values if (v or "").strip())
            except ValueError:
                return False

        predictors = [c for c in header if c not in known and numeric(c)][:4]
        if len(predictors) < 2:
            self.bad("R", f"only {len(predictors)} measured column(s) to analyze")
            return None

        r = self.flow("R run analyses on a spreadsheet of numbers", [
            "csv", *self.browse(table),
            predictors,                 # the columns that ARE the measures
            True, ["text_id"],          # the identifier
            "row",                      # one row of results per spreadsheet row
            ["stats_group_differences", "stats_correlations", "stats_ridge_fit"],
            "condition",                # the groups to compare
            ["openness"],               # the outcome to predict
            False,                      # no controls
            "fdr_bh", False,            # correction, no row filters
            ":done", "Flow R", "save",
        ], analyses=True, preselected=["spreadsheet_columns"],
            var_defaults={"wordclouds": False})
        if not r:
            return None
        stats = r[0] / "stats_results"
        sets = json.loads((stats / "analysis_table_sets.json").read_text(encoding="utf-8"))
        features = sets.get("sets", {}).get("spreadsheet_columns", [])
        metrics = stats / "ridge_cv_metrics.csv"
        rows = list(_csv.DictReader(metrics.open(encoding="utf-8-sig"))) \
            if metrics.exists() else []
        if sorted(features) == sorted(predictors) and rows \
                and (stats / "report.md").is_file():
            self.ok("R outputs", f"{len(features)} predictors, "
                                 f"r2 {rows[0].get('cv_r2')}, report written")
        else:
            self.bad("R outputs", f"features {features}; {len(rows)} metric row(s)")
        return r

    def flow_v(self):
        """Word vectors, three ways round: trained from the Train task's
        wizard path; imported and applied to a second study; and carried
        inside a ridge that scores that study through the replay, which has
        to find the matrix in the library because the ridge carries only the
        manifest."""
        import importlib.util

        if importlib.util.find_spec("gensim") is None:
            print("skip V: gensim is not installed", flush=True)
            return None
        from taters.helpers.library import import_into, kind_by_id, kind_dir

        v = self.flow("V train word vectors (Train a model)", [
            "csv", *self.browse(self.study), ["text"], True, ["pid"],
            "row", False, "Flow V", "save",
        ], analyses=False, preselected=["word_vectors_train"], text_only=True)
        if not v:
            return None
        folder = v[0]
        feats = folder / "features"
        rows = list(csv.DictReader((feats / "word_vectors.csv").open(encoding="utf-8-sig")))
        model = feats / "models" / "word_vectors.json"
        clouds = sorted(q.name for q in (feats / "figures" / "wordclouds" / "word_vectors").glob("*.png"))
        report = (feats / "models" / "word_vectors_report.md")
        problems = []
        if len(rows) != 160 or not rows[0].get("wv_1"):
            problems.append(f"features: {len(rows)} rows, columns {list(rows[0])[:5] if rows else None}")
        for name in ("word_vectors.json", "word_vectors.npy", "word_vectors_neighbors.csv",
                     "word_vectors_report.md", "word_vectors_loss.png"):
            if not (feats / "models" / name).is_file():
                problems.append(f"missing models/{name}")
        if not clouds:
            problems.append("no neighbor clouds")
        if report.is_file() and "## Methods paragraph" not in report.read_text(encoding="utf-8"):
            problems.append("report lacks the methods paragraph")
        described = (folder / "stats_descriptives" / "word_vectors.csv").is_file()
        if not described:
            problems.append("no descriptives for the word-vector table")
        if problems:
            self.bad("V outputs", "; ".join(problems))
        else:
            self.ok("V outputs", f"{len(rows)} rows, {len(clouds)} neighbor clouds, report")

        # V1b: same corpus, but now with a concept dictionary spelled out LIWC-22
        # style (categories as columns, X and weights as cells, a wildcard), so
        # every category becomes a sim_ column and the report says how each one
        # met the vocabulary.
        from taters.text.word_vectors import train_word_vectors

        dicx = self.proj / "study_concepts.dicx"
        dicx.write_text("DicTerm,Stress,Rest\nstress*,X,\ntired,2,\nslept,,X\n"
                        "exercise,,X\nrecover*,,X\n", encoding="utf-8")
        try:
            out = train_word_vectors(
                csv_path=self.study, text_cols=["text"], id_cols=["pid"],
                out_features_csv=self.proj / "v_concepts" / "word_vectors.csv",
                concept_dicts=[dicx], min_count=2, vector_size=32, epochs=5,
                overwrite_existing=True, verbose=False)
            rows = list(csv.DictReader(out.open(encoding="utf-8-sig")))
            report = (self.proj / "v_concepts" / "models" / "word_vectors_report.md").read_text(encoding="utf-8")
            cols = [c for c in rows[0] if c.startswith("sim_")]
            if cols == ["sim_study_concepts__Stress", "sim_study_concepts__Rest"] \
                    and "## Concepts" in report and all(r[cols[0]] for r in rows):
                self.ok("V1b concept dictionary", f"{cols}; report has the Concepts table")
            else:
                self.bad("V1b concept dictionary", f"columns {cols}; Concepts in report: "
                                                   f"{'## Concepts' in report}")
        except Exception as e:
            self.bad("V1b concept dictionary", f"{type(e).__name__}: {e}")

        # now we import the model and apply it to a second study from the
        # checklist. other flows may have left models in the library too, in
        # which case the wizard asks which; picking one opens the options screen
        # on its list.
        import_into(kind_by_id("models"), model, replace=True)
        models_dir = kind_dir(kind_by_id("models"))
        several = len(list(models_dir.glob("*.json"))) > 1
        answers = ["csv", *self.browse(self.study2), ["text"], True, ["pid"],
                   ["score_with_model"], "row"]
        if several:
            answers += [str((models_dir / "word_vectors.json").resolve()), [], ":done"]
        else:
            answers += [[], False]
        s = self.flow("V2 score a second study with the word vectors",
                      answers + ["Flow V2", "save"])
        if s:
            scores = s[0] / "features" / "model_scores.csv"
            srows = list(csv.DictReader(scores.open(encoding="utf-8-sig"))) \
                if scores.exists() else []
            if len(srows) == 60 and "wv_1" in (srows[0] if srows else {}):
                self.ok("V2 columns", f"{len(srows)} rows, {len(srows[0])} columns")
            else:
                self.bad("V2 columns", f"{len(srows)} rows; {list(srows[0])[:6] if srows else None}")

        # a ridge over the word vectors, then that ridge scoring the second
        # study: the replay has to apply the word-vector model carried in the
        # ridge, and that matrix only lives in the library.
        r = self.flow("V3 a ridge over word vectors", [
            "csv", *self.browse(self.study), ["text"], True, ["pid"],
            ["word_vectors_train"], "row",
            ["stats_ridge_fit"], ["openness"],
            False,                  # no controls
            False,                  # no row filter (one set, so no set question/FDR)
            ":done", "Flow V3", "save",
        ])
        if r:
            ridge = r[0] / "stats_results" / "models" / "ridge__all.json"
            if not ridge.exists():
                ridge = next((r[0] / "stats_results" / "models").glob("ridge__*.json"), None)
            if ridge is None:
                self.bad("V3", "no ridge model written")
                return v
            # the ridge carries the manifest of the word vectors V3 trained
            # (a different matrix from V's, since we ran several threads), so the
            # replay only finds the weights if *that* model is in the library --
            # that's what the finish screen's "keep" offer is for.
            import_into(kind_by_id("models"), r[0] / "features" / "models" / "word_vectors.json",
                        replace=True)
            import_into(kind_by_id("models"), ridge, replace=True)
            f = self.flow("V4 the ridge scores the second study through the replay", [
                "csv", *self.browse(self.study2), ["text"], True, ["pid"],
                ["score_with_model"], "row",
                # two models in the library now, so the wizard asks which, and
                # picking one opens the options screen on its list.
                str((kind_dir(kind_by_id("models")) / ridge.name).resolve()),
                [],                 # no statistics on top of that
                ":done", "Flow V4", "save",
            ])
            if f:
                scores = f[0] / "features" / "model_scores.csv"
                frows = list(csv.DictReader(scores.open(encoding="utf-8-sig"))) \
                    if scores.exists() else []
                scored = sum(1 for x in frows if x.get("pred_openness"))
                if frows and scored >= 50:
                    self.ok("V4 replay", f"{scored} of {len(frows)} scored")
                else:
                    self.bad("V4 replay", f"{scored} of {len(frows)} scored")
        return v

    def flow_t(self):
        """Real transformers on the real machine (opt-in, --slow): adapt the
        default encoder to the study, fine-tune a predictor on openness and
        condition, import both, then embed the second study and score it
        with the predictor. Defaults throughout (distilroberta, 3 epochs, 5
        folds): on a GPU a few minutes, on a CPU an hour."""
        import importlib.util
        import json

        if importlib.util.find_spec("torch") is None:
            print("skip T: torch is not installed", flush=True)
            return None
        from taters.helpers.library import import_into, kind_by_id, kind_dir
        from taters.ui.train import ask_outcomes

        a = self.flow("T1 adapt an encoder (Train a model)", [
            "csv", *self.browse(self.study), ["text"], True, ["pid"], "row",
            False, "Flow T1", "save",
        ], analyses=False, preselected=["adapt_encoder"], text_only=True)
        if not a:
            return None
        enc = a[0] / "features" / "models" / "adapted_encoder.json"
        report = a[0] / "features" / "models" / "adapted_encoder_report.md"
        if enc.is_file() and report.is_file() and "perplexity" in report.read_text(encoding="utf-8"):
            ev = json.loads(enc.read_text(encoding="utf-8"))["evaluation"]
            self.ok("T1 outputs", f"perplexity {ev['perplexity_before']:.1f} -> "
                                  f"{ev['perplexity_after']:.1f}")
            import_into(kind_by_id("encoders"), enc)
        else:
            self.bad("T1 outputs", "no encoder manifest or report")
            return a

        f = self.flow("T2 fine-tune a predictor (Train a model)", [
            "csv", *self.browse(self.study), ["text"], True, ["pid"], "row",
            ["openness", "condition"],      # the columns we want to predict
            ":done",                        # picking one opens the options list
            "Flow T2", "save",
        ], analyses=False, preselected=["finetune_text_predictor"], text_only=True,
            before_options=ask_outcomes)
        if not f:
            return a
        metrics = f[0] / "stats_results" / "text_predictor_cv_metrics.csv"
        rows = list(csv.DictReader(metrics.open(encoding="utf-8-sig"))) if metrics.exists() else []
        model = f[0] / "features" / "models" / "text_predictor__openness_condition.json"
        if len(rows) == 2 and model.is_file():
            self.ok("T2 outputs", "; ".join(
                f"{r['outcome']}: " + (f"R2 {r['cv_r2']}" if r["cv_r2"] else f"acc {r['accuracy']}")
                for r in rows))
            import_into(kind_by_id("models"), model)
        else:
            self.bad("T2 outputs", f"{len(rows)} metric rows; model {model.is_file()}")
            return a

        models_dir = kind_dir(kind_by_id("models"))
        several = len(list(models_dir.glob("*.json"))) > 1
        answers = ["csv", *self.browse(self.study2), ["text"], True, ["pid"],
                   ["transformer_embeddings", "score_with_model"], "row"]
        if several:
            answers += [str((models_dir / model.name).resolve())]
        # the encoder is asked at the features stage now, right after the
        # models question. we type the default's name rather than pick a row,
        # since which rows the picker shows depends on what this machine has
        # cached
        answers += [":type", "sentence-transformers/all-roberta-large-v1"]
        answers += [[], ":done"]         # no analyses; no settings changes
        answers += ["Flow T3", "save"]
        s = self.flow("T3 embed and score the second study", answers)
        if s:
            emb = s[0] / "features" / "transformer_embeddings.csv"
            scores = s[0] / "features" / "model_scores.csv"
            e_rows = list(csv.DictReader(emb.open(encoding="utf-8-sig"))) if emb.exists() else []
            s_rows = list(csv.DictReader(scores.open(encoding="utf-8-sig"))) if scores.exists() else []
            if len(e_rows) == 60 and "e_1" in e_rows[0] and len(s_rows) == 60 \
                    and {"pred_openness", "pred_condition"} <= set(s_rows[0]):
                self.ok("T3 outputs", f"{len(e_rows[0]) - 3} dims; classes "
                                      f"{sorted({r['pred_condition'] for r in s_rows})}")
            else:
                self.bad("T3 outputs", f"embeddings {len(e_rows)} rows "
                                       f"{list(e_rows[0])[:5] if e_rows else None}; scores "
                                       f"{list(s_rows[0]) if s_rows else None}")
        return a

    def flow_m(self, media: Path):
        """A recording through the wizard: transcribe, measure the text,
        measure the voice. Slow (a real transcription), so opt-in."""
        folder = self.proj / "media"
        folder.mkdir(exist_ok=True)
        clip = folder / f"clip{media.suffix}"
        if not clip.exists():
            cp = subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-ss", "60", "-t", "90",
                 "-i", str(media), "-c", "copy", str(clip)],
                capture_output=True, text=True)
            if cp.returncode:
                self.bad("M clip", cp.stderr[-400:])
                return None
        kind = "video" if media.suffix.lower() in (".mkv", ".mp4", ".mov") \
            else "audio"
        return self.flow("M a recording: transcribe, text and voice", [
            kind, *self.browse(folder),
            ["readability", "lexical_richness", "acoustics"],
            "transcribe", "speaker", False, "Flow M", "save",
        ])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", type=Path, default=None,
                    help="scratch folder (default: a fresh temp folder)")
    ap.add_argument("--media", type=Path, default=None,
                    help="a recording to drive the media flow with (slow)")
    ap.add_argument("--flows", nargs="*", default=None,
                    help="which flows to run (letters); default all")
    ap.add_argument("--keep", action="store_true",
                    help="leave the scratch folder in place")
    ap.add_argument("--slow", action="store_true",
                    help="also run flow T: real transformers on this machine")
    args = ap.parse_args(argv)
    # we resolve this before the harness chdir's into its project folder,
    # otherwise a relative path stops meaning what it meant on the command line.
    if args.media is not None:
        args.media = args.media.resolve()
        if not args.media.is_file():
            print(f"!! no such recording: {args.media}")
            return 2

    work = args.work or Path(tempfile.mkdtemp(prefix="taters-e2e-"))
    if work.exists() and any(work.iterdir()) and args.work is not None:
        print(f"!! {work} is not empty; finished outputs would short-circuit. "
              f"Point --work at a fresh folder.")
        return 2
    h = Harness(work, keep=args.keep or args.work is not None)
    wanted = {f.upper() for f in (args.flows or list("ABCDEFGHRVTM"))}
    print(f"scratch: {work}")

    runs = {}
    if "A" in wanted:
        runs["A"] = h.flow_a()
    if "B" in wanted:
        runs["B"] = h.flow_b()
    if "C" in wanted:
        runs["C"] = h.flow_c()
    if "D" in wanted:
        runs["D"] = h.flow_d()
    if "E" in wanted:
        h.flow_e(runs)
    if "F" in wanted and runs.get("A"):
        h.flow_f(runs["A"])
    if "G" in wanted and runs.get("A"):
        h.flow_g(runs["A"])
    if "H" in wanted and runs.get("A"):
        h.flow_h(runs["A"])
    if "R" in wanted and runs.get("A"):
        h.flow_r(runs["A"])
    if "V" in wanted:
        h.flow_v()
    if "T" in wanted and args.slow:
        h.flow_t()
    if "M" in wanted and args.media:
        h.flow_m(args.media)

    print()
    if h.failures:
        print(f"FAILED: {', '.join(h.failures)}")
    else:
        print("every flow passed")
    if not h.keep:
        shutil.rmtree(work, ignore_errors=True)
    return 1 if h.failures else 0


if __name__ == "__main__":
    sys.exit(main())
