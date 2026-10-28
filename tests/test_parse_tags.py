from hfpa.data.parse_tags import parse_parents


def test_typed_over_bare():
    tags = ["base_model:meta-llama/Llama-3-8B", "base_model:finetune:meta-llama/Llama-3-8B", "x"]
    assert parse_parents(tags) == [("meta-llama/Llama-3-8B", "finetune")]


def test_bare_and_none():
    assert parse_parents(["base_model:a/b"]) == [("a/b", "unspecified")]
    assert parse_parents(None) == []
