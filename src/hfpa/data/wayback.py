"""Parse archived Hugging Face model pages (Internet Archive Wayback Machine, raw `id_` captures).

A model page embeds its metadata as JSON in `data-props="..."` attributes (html-escaped). The model object is
the dict under key "model" whose "id" is the repo name; it carries likes, downloads (last 30 days), gated and
lastModified. Format by period (checked 2026-11-24 on 2022-02, 2023-09, 2024-01 captures):
  2023+   likes, downloads, gated all in the JSON model object
  2022    likes, gated in JSON; downloads only as visible text "Downloads last month <n>"
Older / other layouts: visible text fallbacks for likes ("like <n>") and downloads.
"""
import html
import json
import re

PROPS = re.compile(r'data-props="([^"]*)"')
DL_TEXT = re.compile(r"Downloads last month\s*(?:<[^>]+>\s*)*([\d,]+)")
LIKE_TEXT = re.compile(r">\s*like\s*(?:<[^>]+>\s*)*([\d,]+(?:\.\d+)?k?)\s*<", re.I)


def _num(s):
    s = s.replace(",", "").lower()
    return int(round(float(s[:-1]) * 1000)) if s.endswith("k") else int(float(s))


def _walk(o):
    if isinstance(o, dict):
        yield o
        for v in o.values():
            yield from _walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk(v)


def parse_page(text, names):
    """text: raw html of one capture; names: accepted repo ids (current name + aliases), compared case-insensitively.
    Returns dict(found_id, likes, downloads30, gated, last_modified, fmt) -- None for fields not on the page.
    fmt: json (model object found), text (visible-text fallback only), none."""
    want = {n.lower() for n in names}
    out = dict(found_id=None, likes=None, downloads30=None, gated=None, last_modified=None, fmt="none")
    for p in PROPS.findall(text):
        try:
            o = json.loads(html.unescape(p))
        except ValueError:
            continue
        for d in _walk(o):
            mid = d.get("id") or d.get("modelId")
            if isinstance(mid, str) and mid.lower() in want and "likes" in d:
                out.update(found_id=mid, likes=d.get("likes"), fmt="json",
                           last_modified=d.get("lastModified") or out["last_modified"])
                if isinstance(d.get("downloads"), (int, float)):
                    out["downloads30"] = int(d["downloads"])
                if "gated" in d:
                    out["gated"] = str(d["gated"]).lower()  # false | manual | auto
        if out["fmt"] == "json" and out["downloads30"] is not None:
            break
    if out["downloads30"] is None:
        m = DL_TEXT.search(text)
        if m:
            out["downloads30"] = _num(m.group(1))
    if out["likes"] is None:
        m = LIKE_TEXT.search(text)
        if m:
            out["likes"], out["fmt"] = _num(m.group(1)), "text"
    return out
