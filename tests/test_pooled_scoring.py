"""
Every per-row text scorer runs on the shared pooled-row driver
(`helpers.row_map.map_text_rows`): readability, lexical richness, the DTM
scan, POS tagging, and a saved MEM model's apply. Two contracts, tested for
each: the output file is byte-identical whatever the worker count, and the
phase announces itself with the documents in flight (which draws the live
sub-bars). Explicit worker counts throughout -- the automatic setting
deliberately stays serial on corpora this small.
"""

from __future__ import annotations

from pathlib import Path

import pytest

N_DOCS = 10


def _ready(tmp_path, name="ready.csv"):
    docs = [
        f"the cat number {i} sat on the mat and slept deeply while "
        f"dreaming of gravy dinner potato and a long meeting deadline"
        for i in range(N_DOCS)
    ]
    src = tmp_path / name
    src.write_text("text_id,text\n" +
                   "".join(f"d{i},{t}\n" for i, t in enumerate(docs)),
                   encoding="utf-8")
    return src


def _freq_list(tmp_path, ready):
    from taters.text.analyze_ngram_frequencies import analyze_ngram_frequencies
    return analyze_ngram_frequencies(
        analysis_csv=ready, out_features_csv=tmp_path / "freq.csv",
        min_freq=1, min_obs_pct=0, min_token_count=1,
        overwrite_existing=True)


def _mem_model(tmp_path, ready):
    # MEM builds its own frequency list and matrix now -- every topic model
    # wants a different one, and a shared matrix meant two of them rebuilding
    # over each other -- so this just hands it the gathered corpus.
    from taters.text.topic_model_mem import topic_model_mem
    topic_model_mem(analysis_csv=ready, n_components=2,
                    min_freq=1, min_obs_pct=0, min_token_count=1,
                    out_features_csv=tmp_path / "mem.csv",
                    overwrite_existing=True)
    return tmp_path / "mem_model.json"


def _run_readability(tmp_path, ready, out, workers, **_):
    from taters.text.analyze_readability import analyze_readability
    return analyze_readability(analysis_csv=ready, out_features_csv=out,
                               workers=workers, overwrite_existing=True)


def _run_lexical(tmp_path, ready, out, workers, **_):
    from taters.text.analyze_lexical_richness import analyze_lexical_richness
    return analyze_lexical_richness(analysis_csv=ready, out_features_csv=out,
                                    workers=workers, overwrite_existing=True)


def _run_dtm(tmp_path, ready, out, workers, freq=None, **_):
    from taters.text.build_doc_term_matrix import build_doc_term_matrix
    return build_doc_term_matrix(freq_list_csv=freq, analysis_csv=ready,
                                 out_features_csv=out, workers=workers,
                                 overwrite_existing=True)


def _run_pos(tmp_path, ready, out, workers, **_):
    from taters.text.analyze_parts_of_speech import analyze_parts_of_speech
    return analyze_parts_of_speech(analysis_csv=ready, out_features_csv=out,
                                   workers=workers, overwrite_existing=True)


def _run_mem_apply(tmp_path, ready, out, workers, model=None, **_):
    from taters.text.topic_model_mem import apply_mem_model
    return apply_mem_model(model_json=model, analysis_csv=ready,
                           out_features_csv=out, workers=workers,
                           overwrite_existing=True)


def _run_cohesion(tmp_path, ready, out, workers, **_):
    from taters.text.analyze_cohesion import analyze_cohesion
    return analyze_cohesion(analysis_csv=ready, out_features_csv=out,
                            workers=workers, semantic_model="none",
                            overwrite_existing=True)


CASES = [
    ("readability", _run_readability, "measuring readability"),
    ("lexical", _run_lexical, "measuring lexical richness"),
    ("dtm", _run_dtm, "building the matrix"),
    ("pos", _run_pos, "tagging documents"),
    ("mem_apply", _run_mem_apply, "scoring themes"),
    ("cohesion", _run_cohesion, "measuring cohesion"),
]


def _extras(tmp_path, ready, case):
    if case == "dtm":
        return {"freq": _freq_list(tmp_path, ready)}
    if case == "mem_apply":
        return {"model": _mem_model(tmp_path, ready)}
    return {}


@pytest.mark.parametrize("case,run,_msg", CASES, ids=[c[0] for c in CASES])
def test_the_output_is_identical_whatever_the_worker_count(tmp_path, case, run, _msg):
    ready = _ready(tmp_path)
    extras = _extras(tmp_path, ready, case)
    serial = run(tmp_path, ready, tmp_path / "serial.csv", 1, **extras)
    pooled = run(tmp_path, ready, tmp_path / "pooled.csv", 2, **extras)
    assert Path(pooled).read_bytes() == Path(serial).read_bytes(), case
    assert Path(serial).stat().st_size > 100, "an empty file proves nothing"


@pytest.mark.parametrize("case,run,msg", CASES, ids=[c[0] for c in CASES])
def test_the_phase_names_itself_with_documents_in_flight(tmp_path, case, run, msg):
    """The FlightReporter contract: the phase message plus the in-flight
    document ids -- what the display turns into one sub-bar per document."""
    ready = _ready(tmp_path)
    extras = _extras(tmp_path, ready, case)
    events = []

    def sink(done, total, message=None, unit=None, **kw):
        events.append((message, kw.get("inflight")))

    if case == "readability":
        from taters.text.analyze_readability import analyze_readability
        analyze_readability(analysis_csv=ready, out_features_csv=tmp_path / "o.csv",
                            workers=1, overwrite_existing=True, on_progress=sink)
    elif case == "lexical":
        from taters.text.analyze_lexical_richness import analyze_lexical_richness
        analyze_lexical_richness(analysis_csv=ready, out_features_csv=tmp_path / "o.csv",
                                 workers=1, overwrite_existing=True, on_progress=sink)
    elif case == "dtm":
        from taters.text.build_doc_term_matrix import build_doc_term_matrix
        build_doc_term_matrix(freq_list_csv=extras["freq"], analysis_csv=ready,
                              out_features_csv=tmp_path / "o.csv",
                              workers=1, overwrite_existing=True, on_progress=sink)
    elif case == "pos":
        from taters.text.analyze_parts_of_speech import analyze_parts_of_speech
        analyze_parts_of_speech(analysis_csv=ready, out_features_csv=tmp_path / "o.csv",
                                workers=1, overwrite_existing=True, on_progress=sink)
    elif case == "mem_apply":
        from taters.text.topic_model_mem import apply_mem_model
        apply_mem_model(model_json=extras["model"], analysis_csv=ready,
                        out_features_csv=tmp_path / "o.csv",
                        workers=1, overwrite_existing=True, on_progress=sink)
    else:
        from taters.text.analyze_cohesion import analyze_cohesion
        analyze_cohesion(analysis_csv=ready, out_features_csv=tmp_path / "o.csv",
                         workers=1, semantic_model="none",
                         overwrite_existing=True, on_progress=sink)

    phase = [(m, i) for m, i in events if m == msg]
    assert phase, f"{case}: the phase never announced itself as {msg!r}"
    assert any(i for _m, i in phase), f"{case}: no document was shown in flight"


def test_the_row_count_is_taken_even_when_nobody_is_watching(tmp_path):
    """
    `count_rows` returned 0 without a progress sink, on the theory that the
    count was only ever a progress bar's denominator. `map_text_rows` sizes
    its worker pool from it, so every direct API call ran on one worker
    while the same call from the wizard ran on twelve.
    """
    from taters.helpers.progress import count_rows

    path = tmp_path / "rows.csv"
    path.write_text("text_id,text\n" + "".join(f"d{i},words {i}\n"
                                                for i in range(120)),
                    encoding="utf-8")
    assert count_rows(path, on_progress=None) == 120
