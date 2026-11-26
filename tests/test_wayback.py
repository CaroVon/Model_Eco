import gzip
from pathlib import Path

from hfpa.data.wayback import parse_page

FIX = Path(__file__).parent / "fixtures" / "wayback"


def page(name):
    return gzip.open(FIX / f"{name}.html.gz", "rt", encoding="utf-8", errors="ignore").read()


def test_2023_json():
    r = parse_page(page("llama2_2309"), ["meta-llama/Llama-2-7b-hf"])
    assert r["fmt"] == "json" and r["likes"] == 485 and r["downloads30"] == 448596 and r["gated"] == "manual"
    assert r["last_modified"].startswith("2023-08-09")


def test_2024_json():
    r = parse_page(page("bloke_2401"), ["TheBloke/Llama-2-7B-GGUF"])
    assert (r["likes"], r["downloads30"], r["gated"]) == (93, 676, "false")


def test_2022_text_downloads_and_alias():
    # current name google-bert/bert-base-uncased; the 2022 page uses the old name
    r = parse_page(page("bert_2202"), ["google-bert/bert-base-uncased", "bert-base-uncased"])
    assert r["found_id"] == "bert-base-uncased" and r["likes"] == 99 and r["downloads30"] == 10611065
    assert r["gated"] == "false"


def test_wrong_model_not_matched():
    r = parse_page(page("llama2_2309"), ["someone/else"])
    assert r["fmt"] != "json" or r["found_id"] is None
