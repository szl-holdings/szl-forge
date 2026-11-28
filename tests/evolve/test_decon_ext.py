from szl_evolve import decon_ext as d

def mk(t): return {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": t}, {"role": "assistant", "content": "a"}]}
Q = "Explain why the estate receipt chain must never claim a deployed model is running without current direct evidence from the host"

def test_exact_and_edited_copy_detected():
    train = [mk("totally unrelated text about kernels and tokenizers and warp shuffle reductions in cuda " * 2),
             mk("Preamble. " + Q + ". Trailing words added by a template.")]
    idx = d.NgramIndex([Q], n=5)
    hits = d.detect(idx, [d.row_prompt(r) for r in train], threshold=0.8)
    assert [h["train_index"] for h in hits] == [1] and hits[0]["score"] >= 0.8

def test_light_edit_tolerated_by_cluster_expansion():
    edited = Q.replace("must never claim", "should never claim").replace("current direct", "current, direct")
    idx = d.NgramIndex([Q], n=5)
    assert d.detect(idx, [edited], threshold=0.6, max_misses=6)

def test_unrelated_not_flagged():
    idx = d.NgramIndex([Q], n=5)
    assert d.detect(idx, ["a completely different sentence about lunch menus and parking permits for the downtown office building"]) == []

def test_short_reference_exact_phrase_only():
    idx = d.NgramIndex(["what is two plus three exactly"], n=8)
    assert d.detect(idx, ["User asks: what is two plus three exactly? Answer five."])
    assert not d.detect(idx, ["what is two plus four exactly"])

def test_minhash_near_dup_and_threshold():
    a = "the quick brown fox jumps over the lazy dog near the quiet river bank at dawn today"
    b = a.replace("quiet", "calm")
    c = "orthogonal content about gradient checkpointing optimizer state and learning rate schedules in training"
    res = d.near_duplicates([a], [b, c], threshold=0.6)
    assert len(res) == 1 and res[0]["right_index"] == 0

def test_minhash_deterministic():
    m1, m2 = d.MinHasher(seed=7), d.MinHasher(seed=7)
    assert m1.signature("a b c d e f g") == m2.signature("a b c d e f g")

def test_group_overlap_detects_shared_groups_and_missing_keys():
    tr = [{"source_dataset": "x", "source_revision": "r1"}, {"source_dataset": "y"}]
    ho = [{"source_dataset": "x", "source_revision": "r1"}, {"source_dataset": "z"}, {"foo": 1}]
    g = d.group_overlap(tr, ho)
    assert g["holdout_rows_sharing_group_with_train"] == 1 and g["rows_without_group_key"] == 1 and not g["group_disjoint"]

def test_run_reject_and_clean():
    train = [mk(Q + " yes")]
    rep = d.run(train, [mk(Q)])
    assert rep["verdict"] == "REJECT" and rep["holdout_rows_contaminated"] == 1
    clean = d.run([{"source_dataset": "a", "messages": mk("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda")["messages"]}],
                  [{"source_dataset": "b", "messages": mk("mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega")["messages"]}])
    assert clean["verdict"] == "CLEAN_NOT_LAUNCHED"

def test_cross_matrix_flags_leak_direction():
    A = {"train": [mk(Q), mk("only in a: ledger rows are append only and never rewritten by any lane writer ever")], "holdout": [mk("a holdout prompt about khipu abstention gates and sealed evaluation sets in estate")]}
    B = {"train": [mk("b train unrelated prompt about retrieval augmented generation of kernel documentation pages")], "holdout": [mk(Q)]}
    m = d.cross_matrix({"A": A, "B": B})
    assert m["A"]["B"]["exact"] == 1 and m["B"]["A"]["exact"] == 0

def test_boilerplate_strip_removes_shared_persona_text():
    persona = "You are SZL-1 the sovereign model of SZL Holdings fine tuned and served on its own hardware"
    tr = [persona + f" What is topic{i} about in the estate record number {i}" for i in range(12)]
    out = d.strip_boilerplate(tr)
    assert all("sovereign" not in o for o in out) and all(f"topic{i}" in o for i, o in enumerate(out))

def test_run_persona_prefix_not_false_positive_but_real_dup_is():
    persona = "You are SZL-1 the sovereign model of SZL Holdings fine tuned and served on its own hardware"
    train = [{"messages": [{"role": "user", "content": persona + f" What is thing{i} in the estate registry ledger"}]} for i in range(12)]
    ho_fp = [{"messages": [{"role": "user", "content": persona + " What is your name"}]}]
    assert d.run(train, ho_fp)["holdout_rows_contaminated"] == 0
    ho_tp = [{"messages": [{"role": "user", "content": persona + " What is thing3 in the estate registry ledger"}]}]
    assert d.run(train, ho_tp)["holdout_rows_contaminated"] == 1
