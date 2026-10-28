"""Parse base_model tags into (parent, rel_type) pairs (plan section 7.1)."""
import re

REL_RE = re.compile(r"^base_model:(finetune|adapter|quantized|merge):(.+)$")
BARE_RE = re.compile(r"^base_model:(?!finetune:|adapter:|quantized:|merge:)(.+)$")


def parse_parents(tags):
    typed, bare = [], []
    for t in tags if tags is not None else []:
        m = REL_RE.match(t)
        if m:
            typed.append((m.group(2).strip(), m.group(1)))
            continue
        m = BARE_RE.match(t)
        if m:
            bare.append((m.group(1).strip(), "unspecified"))
    typed_ids = {p for p, _ in typed}  # typed tags take priority for the same parent
    return typed + [(p, r) for p, r in bare if p not in typed_ids]
