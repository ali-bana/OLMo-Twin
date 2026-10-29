import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_posttrain_corpus as bp  # noqa: E402


def test_sft_row_is_one_record_with_every_message():
    row = {"id": "x1", "source": "s", "messages": [{"role": "user", "content": "first message here"},
                                                   {"role": "assistant", "content": "second message here"}]}
    recs = list(bp.records_from_row("sft", row, 0))
    assert len(recs) == 1
    assert recs[0]["id"] == "x1"
    assert "first message here" in recs[0]["text"] and "second message here" in recs[0]["text"]


def test_dpo_row_yields_chosen_and_rejected():
    row = {"id": "p1", "chosen": [{"role": "user", "content": "prompt"}, {"role": "assistant", "content": "good answer"}],
           "rejected": [{"role": "user", "content": "prompt"}, {"role": "assistant", "content": "bad answer"}],
           "chosen_model": "a", "rejected_model": "b", "source": "s"}
    recs = list(bp.records_from_row("dpo", row, 0))
    assert [r["id"] for r in recs] == ["p1#chosen", "p1#rejected"]
    assert "good answer" in recs[0]["text"] and "bad answer" in recs[1]["text"]
    assert "good answer" not in recs[1]["text"]


def test_rlvr_row_includes_ground_truth_and_constraint():
    row = {"messages": [{"role": "user", "content": "solve this"}], "ground_truth": "42",
           "dataset": "MATH", "constraint_type": "format", "constraint": "answer in a box"}
    recs = list(bp.records_from_row("rlvr1", row, 7))
    assert len(recs) == 1 and recs[0]["id"] == "rlvr1-7"
    for piece in ("solve this", "42", "answer in a box"):
        assert piece in recs[0]["text"]


def test_empty_and_malformed_messages_are_skipped():
    assert list(bp.records_from_row("sft", {"id": "e", "messages": []}, 0)) == []
    assert list(bp.records_from_row("sft", {"id": "e", "messages": None}, 0)) == []
    row = {"id": "e", "messages": [{"role": "user", "content": None}, {"role": "a", "content": "  "},
                                   {"role": "b", "content": "real text"}]}
    recs = list(bp.records_from_row("sft", row, 0))
    assert len(recs) == 1 and recs[0]["text"] == "real text"


def test_concatenation_preserves_within_message_spans():
    """The audit's guarantee: any span inside one message survives verbatim."""
    span = "a long distinctive span of words that must survive the join intact"
    row = {"id": "s", "messages": [{"role": "user", "content": "intro"},
                                   {"role": "assistant", "content": f"before {span} after"}]}
    text = list(bp.records_from_row("sft", row, 0))[0]["text"]
    assert span in text


def test_shard_counts_cover_every_component():
    assert set(bp.SHARDS) == {c for c, _ in bp.SOURCES.values()}
