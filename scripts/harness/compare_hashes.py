"""Compare two harness hash files and say, per bundle and take, what differs and whether it was declared.

Inputs are files written by hash_outputs.py, a reference hash file of the production bundles'
sample subjects (same format, fewer details; given by path like any other input), or a corpus
ARRAY_SHA256.json
from _superseded/ (attached to a side with --a-corpus / --b-corpus NAME=PATH; its keys are
relative to the corpus root).

Every difference found is put in one category. The declared difference categories (differences a
code change may legitimately introduce):

  C0 scope        a sample run against a full production bundle: the bundle-root aggregates
                  (INDEX.json, DATA_DESCRIPTION_EN.md, reduced_model_fit.json, the PRISM policy
                  file) describe different take sets, and README / VALIDATION_REPORT /
                  validation_ledger exist only where validate and readme were run
  C1 smpl18       the smpl18 settings/profile hashes: reduced_model_fit.json settings and each
                  manifest's anthro_reconstruction.reduced_model_fit.settings
  C2 corpus code  the retarget corpus content_hashes (code=) and the corpus-file digests that follow.
                  A content_hashes value (`adapter=..;config=..;code=..;model=..`, in the corpus
                  npz and copied into the AddBio manifests) is C2 only when both values are
                  recorded and `code` is the one token that differs; another token (a different
                  body model, source file or config) is XX, and a value the hash files do not
                  record is UL. A manifest without a recorded value is C2 only when the corpus file
                  its take was made from (its `source_asset_id`) shows a code-only difference
  C3 pair_id      pair_id changes that follow from the corpus file bytes (C2, or C4 for HKNU)
  C4 hknu env     an HKNU corpus built under a different Python/numpy build than the compared
                  run; a rebuild differs at the float32 last digit there, and
                  everything downstream of it moves (the spine1/spine2 reduce fit is flat, so
                  its constants move visibly). Only declared against a reference; between two
                  harness runs in one environment an HKNU difference is undeclared
  C5 prefix       the corpus-location prefix tmp/harness/<run>/ in source_asset_id / relative_path
                  (absorbed when --normalize-prefix is given; counted as `normalized`)

C5 is decided by value, never by field name: a difference is C5 only when the two sides become
equal once each side's RECORDED prefix is removed (the normalized hashes hash_outputs.py writes,
or the values themselves). Anything that still differs after that is a real difference, whatever
field it sits in, so a relative_path that changed in any other way is XX.

Anything else is  XX undeclared, and a JSON whose difference cannot be located (no subtree hashes on
either side, or a depth-3 composite that mixes a declared digest with a rewritten path
(relative_path) and was hashed without its depth-4 split) is  UL unlocalized. The exit status is 1
when either appears.

Between two harness runs (both hash files record `scratch`), a corpus file that only one side has
is XX: the runs rebuilt the same sample, so a trial npz, _run.json, SUMMARY.json or _done.json the
code stopped (or started) writing is a difference. Against an ARRAY_SHA256 reference, which covers
the whole corpus while a harness run covers its sample, such files are only counted.

    python scripts/harness/compare_hashes.py A.json B.json [--normalize-prefix tmp/harness/<run>/=] \\
        [--b-detail reference_detail.json] \\
        [--b-corpus hknu_smpl24_paired=.../ARRAY_SHA256.json] [--out diff.json]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib

CORPUS_OF = {"hknu_unified8": "hknu_smpl24_paired", "gaitex_unified8": "gaitex_smpl24",
             "addbio_unified8": "addbio_smpl24_raw"}
SCOPE_FILES = {"INDEX.json", "DATA_DESCRIPTION_EN.md", "reduced_model_fit.json",
               "prism_faithful_reduced_model_policy_v1.json"}
STEP_FILES = {"README.md", "VALIDATION_REPORT.json", "validation_ledger.json"}
#: Depth-3 manifest subtrees that hold a declared digest (C2) next to a path that may be rewritten
#: (relative_path). Without the depth-4 split on both sides their difference cannot be attributed
#: to one or the other.
COMPOSITES = ("small_synthesis.inputs.retarget_npz",)
#: The wall-clock fields hash_outputs.py masks (kept in step with its MASK).
MASK = {"generated_utc", "applied_utc", "generated_at", "elapsed_s", "elapsed_seconds"}
DECLARED = ("C0", "C1", "C2", "C3", "C4", "C5")
LABELS = {
    "C0": "scope (sample vs full bundle; validate/readme not run)",
    "C1": "smpl18 settings/profile hashes",
    "C2": "corpus code= / content_hashes / corpus-file digest",
    "C3": "pair_id following the corpus file bytes",
    "C4": "HKNU corpus built under a different interpreter/numpy",
    "C5": "corpus-location prefix",
    "XX": "UNDECLARED",
    "UL": "UNLOCALIZED (differs where the hash files cannot say which field)",
}


# --------------------------------------------------------------------------- loading
def load(path: str) -> dict:
    doc = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    doc.setdefault("bundles", {})
    doc.setdefault("corpora", {})
    return doc


def attach_corpus(doc: dict, spec: str) -> None:
    name, _, path = spec.partition("=")
    ref = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    files = {rel: {"members": members} for rel, members in ref["files"].items()}
    doc["corpora"][name] = {"files": files, "array_reference": path}


def _masked(obj):
    if isinstance(obj, dict):
        return {k: ("<masked>" if k in MASK else _masked(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_masked(v) for v in obj]
    return obj


def _canonical_sha(obj) -> str:
    """hash_outputs.py's masked_sha256 of a parsed document."""
    return hashlib.sha256(json.dumps(_masked(obj), sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


class Side:
    """One input, with the normalization it may use and optional detail from another hash file.

    `use_norm` says whether the caller asked for this side's recorded prefix to be removed before
    comparing; `strip` removes it regardless, which is how a prefix-only difference is recognised
    (and classified C5) when normalization was not asked for.
    """

    def __init__(self, doc: dict, normalize: list[tuple[str, str]], detail: dict | None = None):
        self.doc = doc
        recorded = doc.get("normalization") or {}
        self.prefix = recorded.get("prefix") or None
        pair = (self.prefix, recorded.get("replacement", ""))
        # `auto` accepts whatever prefix the side's own hash file says it may remove
        self.use_norm = bool(self.prefix) and (("auto", "") in normalize or pair in normalize)
        self.detail = detail

    # -- values
    def strip(self, value):
        """The value with this side's recorded prefix (either slash style) removed."""
        if not self.prefix or not isinstance(value, str):
            return value
        for variant in (self.prefix, self.prefix.replace("/", "\\")):
            value = value.replace(variant, "")
        return value

    def norm_value(self, value):
        return self.strip(value) if self.use_norm else value

    # -- npz members
    def member(self, m: dict) -> tuple:
        return self.full_member(m) if self.use_norm else self.raw_member(m)

    @staticmethod
    def raw_member(m: dict) -> tuple:
        return (m["dtype"], tuple(m["shape"]), m["data_sha256"])

    @staticmethod
    def full_member(m: dict) -> tuple:
        if "norm_data_sha256" in m:
            return (m.get("norm_dtype", m["dtype"]), tuple(m["shape"]), m["norm_data_sha256"])
        return (m["dtype"], tuple(m["shape"]), m["data_sha256"])

    # -- json
    def masked(self, rec: dict) -> str | None:
        return self.full_masked(rec) if self.use_norm else rec.get("masked_sha256")

    @staticmethod
    def full_masked(rec: dict) -> str | None:
        return rec.get("norm_masked_sha256", rec.get("masked_sha256"))

    def tree_record(self, rec: dict, bundle: str, where: tuple) -> dict | None:
        """The record whose subtree hashes describe `rec`: itself, or the detail file's record
        of the same content (a reference hash file carries none of its own)."""
        if rec.get("subtrees") is not None:
            return rec
        if self.detail is not None:
            other = _dig(self.detail, bundle, where)
            if other and other.get("masked_sha256") == rec.get("masked_sha256") \
                    and other.get("subtrees") is not None:
                return other
        return None


def _dig(doc: dict, bundle: str, where: tuple) -> dict | None:
    node = doc.get("bundles", {}).get(bundle)
    if node is None:
        return None
    kind = where[0]
    if kind == "root":
        return node.get("root_files", {}).get(where[1])
    return ((node.get("takes", {}).get(where[1]) or {}).get("files") or {}).get(where[2])


def _trees(rec: dict, normalized: bool, deep: bool) -> dict:
    trees = dict(rec["subtrees"])
    if normalized:
        trees.update(rec.get("norm_subtrees", {}))
    if deep:
        trees.update(rec.get("deep_subtrees", {}))
        if normalized:
            trees.update(rec.get("norm_deep_subtrees", {}))
    return trees


def json_differences(A: Side, B: Side, ra: dict, rb: dict) -> list[tuple[str, str, bool]] | None:
    """[(path, note, prefix_only)] for two json records with subtree hashes, or None without them.

    Compared at depth 4 when both records carry the depth-4 layer, else at depth 3. A path is
    prefix-only when both sides' hashes agree once each side's recorded prefix is removed.
    """
    deep = "deep_subtrees" in ra and "deep_subtrees" in rb
    ta, tb = _trees(ra, A.use_norm, deep), _trees(rb, B.use_norm, deep)
    fa, fb = _trees(ra, True, deep), _trees(rb, True, deep)
    out = []
    for path in leaf_differences(ta, tb):
        if (path in ta) != (path in tb):
            out.append((path, "only in " + ("A" if path in ta else "B"), False))
            continue
        out.append((path, "" if deep else "depth 3", fa.get(path) == fb.get(path)))
    return out


# ------------------------------------------------------------------------ classifying
def classify_member(bundle: str, name: str, corpus_evidence: dict) -> str:
    """A differing npz member that is NOT a prefix-only difference (that was ruled out first)."""
    fed = bundle in CORPUS_OF
    if name == "pair_id":
        return "C3" if fed else "XX"
    if bundle == "hknu_unified8" and corpus_evidence.get("hknu_smpl24_paired", {}).get("numeric"):
        return "C4"
    return "XX"


def classify_path(bundle: str, path: str, corpus_evidence: dict, split: bool = True) -> str:
    """A differing manifest subtree that is NOT a prefix-only difference.

    `split` is False when the comparison ran at depth 3 only, so a composite in COMPOSITES is a
    single hash covering a declared digest and a rewritten path together: that is UL, not C2."""
    fed = bundle in CORPUS_OF
    hknu_env = bundle == "hknu_unified8" and corpus_evidence.get("hknu_smpl24_paired", {}).get("numeric")
    if "reduced_model_fit" in path and (path.endswith(".settings") or ".settings." in path):
        return "C1"
    if path == "identity.pair_id" and fed:
        return "C3"
    if path in COMPOSITES and fed and not split:
        return "UL"
    if fed and (path == "source.source_asset_sha256" or path.endswith("retarget_npz.sha256")):
        if bundle != "hknu_unified8":
            return "C2"
        return "C4" if hknu_env else "XX"
    if hknu_env:
        return "C4"
    return "XX"


def content_hash_tokens(value) -> dict | None:
    """`adapter=..;config=..;code=..;model=..` as a dict; None when it is not in that form."""
    if not isinstance(value, str) or "=" not in value:
        return None
    tokens = {}
    for part in value.split(";"):
        key, sep, token = part.partition("=")
        if not sep or not key or key in tokens:
            return None
        tokens[key] = token
    return tokens


def content_hashes_category(va, vb) -> tuple[str, str]:
    """(category, detail) for two differing content_hashes values.

    C2 only when both values are recorded and the one token that differs is `code` (the 2026-09-25
    decision declares that token and no other); XX when another token differs; UL when
    either value is missing, since the hash then says nothing about which token moved."""
    if va is None or vb is None:
        missing = "A" if va is None else "B"
        return "UL", (f"side {missing} records no value, so which token differs is unknown; rehash "
                      "it with the current hash_outputs.py")
    ta, tb = content_hash_tokens(va), content_hash_tokens(vb)
    if ta is None or tb is None:
        return "XX", f"not in the key=value;... form: A={va!r} B={vb!r}"
    keys = sorted(k for k in set(ta) | set(tb) if ta.get(k) != tb.get(k))
    detail = "; ".join(f"{k}: A={ta.get(k)} B={tb.get(k)}" for k in keys)
    return ("C2" if keys == ["code"] else "XX"), detail


def _json_values(rec: dict | None) -> dict:
    return (rec or {}).get("values") or {}


def corpus_file_of(side: Side, bundle: str, rel: str) -> str | None:
    """The corpus file (relative to the corpus root) a bundle take was made from, as this side's
    take npz name it in `source_asset_id`; None when the side does not record it."""
    corpus = CORPUS_OF.get(bundle)
    take = ((side.doc.get("bundles", {}).get(bundle) or {}).get("takes", {}).get(rel) or {})
    marker = f"/{corpus}/"
    for rec in (take.get("files") or {}).values():
        value = ((rec.get("members") or {}).get("source_asset_id") or {}).get("value")
        if isinstance(value, str):
            text = "/" + value.replace("\\", "/")
            if marker in text:
                return text.split(marker, 1)[1]
    return None


def manifest_content_hashes(A: Side, B: Side, bundle: str, key: tuple, path: str,
                            records: tuple, evidence: dict) -> tuple[str, str]:
    """(category, note) for a manifest's `...retarget_content_hashes`, a copy of the corpus npz's
    content_hashes. With both values recorded, their tokens decide. Without them, the take's own
    corpus file decides: C2 when the corpus comparison found a code-only difference in exactly that
    file, UL otherwise (no corpus evidence, or the take's corpus file cannot be named)."""
    va = next((_json_values(r).get(path) for r in records[0] if _json_values(r).get(path)), None)
    vb = next((_json_values(r).get(path) for r in records[1] if _json_values(r).get(path)), None)
    if va is not None and vb is not None:
        return content_hashes_category(va, vb)
    names = {n for n in (corpus_file_of(A, bundle, key[1]), corpus_file_of(B, bundle, key[1]))
             if n is not None}
    by_file = evidence.get(CORPUS_OF.get(bundle), {}).get("content_hashes_by_file", {})
    if len(names) == 1:
        name = names.pop()
        if by_file.get(name) == "C2":
            return "C2", (f"no recorded value; follows the corpus file {name}, whose content_hashes "
                          "differ in code= only")
        return "UL", (f"no recorded value, and the corpus comparison shows no code-only difference "
                      f"for {name} ({by_file.get(name, 'not compared')}); rehash with the current "
                      "hash_outputs.py")
    return "UL", ("no recorded value, and the take's corpus file cannot be named; rehash with the "
                  "current hash_outputs.py")


def leaf_differences(a: dict, b: dict) -> list[str]:
    """The most specific subtree paths whose hashes differ (or exist on one side only)."""
    differing = {p for p in set(a) | set(b) if a.get(p) != b.get(p)}
    out = []
    for path in sorted(differing):
        if not any(q.startswith(path + ".") for q in differing):
            out.append(path)
    return out


# ------------------------------------------------------------------------ comparing
class Report:
    def __init__(self) -> None:
        self.items: list[dict] = []
        self.stats = collections.defaultdict(collections.Counter)

    def add(self, bundle: str, category: str, where: str, what: str, detail: str = "") -> None:
        self.items.append({"bundle": bundle, "category": category, "where": where, "what": what,
                           "detail": detail})
        self.stats[bundle][category] += 1


def compare_npz(rep: Report, bundle: str, where: str, fa: dict, fb: dict, A: Side, B: Side,
                evidence: dict) -> str:
    ma, mb = fa.get("members", {}), fb.get("members", {})
    verdict = "identical"
    for name in sorted(set(ma) | set(mb)):
        if name not in ma or name not in mb:
            rep.add(bundle, "XX", where, f"member {name}", "only in " + ("A" if name in ma else "B"))
            verdict = "differs"
            continue
        ea, eb = A.member(ma[name]), B.member(mb[name])
        if ea == eb:
            if Side.raw_member(ma[name]) != Side.raw_member(mb[name]):
                rep.stats[bundle]["normalized"] += 1
                rep.stats[bundle]["C5_absorbed"] += 1
            rep.stats[bundle]["members_equal"] += 1
            continue
        verdict = "differs"
        parts = [k for k, x, y in zip(("dtype", "shape", "data"), ea, eb) if x != y]
        values = (A.norm_value(ma[name].get("value")), B.norm_value(mb[name].get("value")))
        if Side.full_member(ma[name]) == Side.full_member(mb[name]):
            cat = "C5"               # equal once each side's recorded prefix is removed
        else:
            cat = classify_member(bundle, name, evidence)
        shown = "" if values == (None, None) else f" A={str(values[0])[:120]!r} B={str(values[1])[:120]!r}"
        rep.add(bundle, cat, where, f"member {name}", ",".join(parts) + shown)
    rep.stats[bundle]["npz_" + verdict] += 1
    return verdict


def compare_json(rep: Report, bundle: str, where: str, key: tuple, fa: dict, fb: dict,
                 A: Side, B: Side, evidence: dict) -> str:
    if A.masked(fa) == B.masked(fb):
        if fa.get("masked_sha256") != fb.get("masked_sha256"):
            rep.stats[bundle]["C5_absorbed"] += 1
        kind = "text" if fa.get("kind") == "text" else "json"
        if fa.get("sha256") != fb.get("sha256"):
            rep.stats[bundle][f"{kind}_equal_after_mask"] += 1
        if fa.get("crlf") != fb.get("crlf"):
            rep.add(bundle, "XX", where, "line endings", f"crlf A={fa.get('crlf')} B={fb.get('crlf')}")
        rep.stats[bundle][f"{kind}_identical"] += 1
        return "identical"
    if fa.get("kind") == "text" or fb.get("kind") == "text":
        if Side.full_masked(fa) == Side.full_masked(fb):
            rep.add(bundle, "C5", where, "text", "equal once the recorded prefixes are removed")
        else:
            rep.add(bundle, "XX", where, "text", "differs with ISO-8601 timestamps masked")
        rep.stats[bundle]["text_differs"] += 1
        return "differs"
    ra, rb = A.tree_record(fa, bundle, key), B.tree_record(fb, bundle, key)
    found = None if ra is None or rb is None else json_differences(A, B, ra, rb)
    if found is None:
        rep.add(bundle, "UL", where, "json", "masked hash differs; no subtree hashes to locate it")
    else:
        split = all("deep_subtrees" in r for r in (ra, rb))
        for path, note, prefix_only in found:
            if prefix_only:
                cat = "C5"
            elif path.endswith("retarget_content_hashes") and key[0] == "take":
                cat, note = manifest_content_hashes(A, B, bundle, key, path, ((fa, ra), (fb, rb)),
                                                    evidence)
                rep.add(bundle, cat, where, f"json {path}", note)
                continue
            else:
                cat = classify_path(bundle, path, evidence, split)
            if cat == "UL":
                note = ("depth-3 composite (a C2 digest beside a rewritten relative_path); rehash "
                        "both sides with the current hash_outputs.py to split it")
            rep.add(bundle, cat, where, f"json {path}", note)
    rep.stats[bundle]["json_differs"] += 1
    return "differs"


def compare_file(rep, bundle, where, key, fa, fb, A, B, evidence) -> str:
    if fa.get("sha256") == fb.get("sha256"):
        rep.stats[bundle]["files_byte_identical"] += 1
        if "members" in fa:
            rep.stats[bundle]["members_equal"] += len(fa["members"])
        return "identical"
    if "members" in fa and "members" in fb:
        return compare_npz(rep, bundle, where, fa, fb, A, B, evidence)
    if "masked_sha256" in fa and "masked_sha256" in fb:
        return compare_json(rep, bundle, where, key, fa, fb, A, B, evidence)
    rep.add(bundle, "XX", where, "file bytes", f"sha256 differs ({fa.get('bytes')} vs {fb.get('bytes')} bytes)")
    return "differs"


def _value_category(A: Side, B: Side, va, vb) -> str | None:
    """None when the values agree under the requested normalization, C5 when they agree once
    each side's recorded prefix is removed, else '' (a real difference, for the caller to name)."""
    if A.norm_value(va) == B.norm_value(vb):
        return None
    if A.strip(va) == B.strip(vb):
        return "C5"
    return ""


def compare_index_file(rep: Report, name: str, a: dict, b: dict, fa: dict, fb: dict,
                       A: Side, B: Side) -> None:
    """INDEX.json of the same take set. The entries are compared one by one from index_sample
    (see compare_bundle); this checks everything else: the head (every key but `takes`) and the
    take order, which with the entries make up the whole file.

    A format-1 hash file records neither. Against a format-2 file it is checked by rebuilding it:
    the other side's head and order with this side's own entries must hash to this side's
    recorded masked_sha256. Two format-1 files leave it unlocalized."""
    where = "<root>/INDEX.json"
    if fa.get("sha256") == fb.get("sha256"):
        rep.stats[name]["files_byte_identical"] += 1
        return
    if A.masked(fa) == B.masked(fb):
        rep.stats[name]["json_equal_after_mask" if fa.get("masked_sha256") == fb.get("masked_sha256")
                        else "C5_absorbed"] += 1
        if fa.get("crlf") != fb.get("crlf"):
            rep.add(name, "XX", where, "line endings", f"crlf A={fa.get('crlf')} B={fb.get('crlf')}")
        return
    ha, hb = a.get("index_head"), b.get("index_head")
    oa, ob = a.get("index_order"), b.get("index_order")
    if ha is not None and hb is not None and oa is not None and ob is not None:
        for key in sorted(set(ha) | set(hb)):
            if key not in ha or key not in hb:
                rep.add(name, "XX", where, f"head {key}", "only in " + ("A" if key in ha else "B"))
                continue
            cat = _value_category(A, B, ha[key], hb[key])
            if cat is not None:
                rep.add(name, cat or "XX", where, f"head {key}",
                        f"A={json.dumps(ha[key])[:100]} B={json.dumps(hb[key])[:100]}")
        if oa != ob:
            detail = ("same takes, other order" if sorted(oa) == sorted(ob)
                      else f"A={len(oa)} takes B={len(ob)} takes")
            rep.add(name, "XX", where, "take order", detail)
        sample_a = {e["rel"] for e in a.get("index_sample", [])}
        sample_b = {e["rel"] for e in b.get("index_sample", [])}
        if sample_a != set(oa) or sample_b != set(ob):
            rep.add(name, "UL", where, "entries", "index_sample does not cover every INDEX take")
            return
        rep.stats[name]["index_head_and_order_compared"] += 1
        return
    if (ha is None) == (hb is None):
        rep.add(name, "UL", where, "json",
                "masked hash differs; neither hash file records the INDEX head and take order "
                "(format 1); rehash with the current hash_outputs.py")
        return
    new, old, old_rec, side = (a, b, fb, "B") if ha is not None else (b, a, fa, "A")
    order = new.get("index_order")
    entries = {e["rel"]: e for e in old.get("index_sample", [])}
    if order is None or set(order) != set(entries):
        rep.add(name, "UL", where, "json", f"format-1 side {side}: its entries do not cover the "
                                           "other side's take list; the head cannot be rebuilt")
        return
    rebuilt = dict(new["index_head"])
    rebuilt["takes"] = [entries[rel] for rel in order]
    if _canonical_sha(rebuilt) == old_rec.get("masked_sha256"):
        rep.stats[name]["index_head_and_order_verified_by_rebuild"] += 1
        return
    rep.add(name, "UL", where, "json", f"format-1 side {side}: its INDEX does not rebuild from the "
                                       "other side's head and order, so the head or the order "
                                       "differs (the entries are compared separately)")


def compare_fit(rep: Report, name: str, fa: dict, fb: dict, same_scope: bool,
                evidence: dict) -> None:
    if fa.get("settings_sha") != fb.get("settings_sha"):
        sa, sb = fa.get("settings") or {}, fb.get("settings") or {}
        keys = sorted(k for k in set(sa) | set(sb) if sa.get(k) != sb.get(k))
        rep.add(name, "C1", "<root>/reduced_model_fit.json", "settings",
                "; ".join(f"{k}: A={json.dumps(sa.get(k))[:160]} B={json.dumps(sb.get(k))[:160]}" for k in keys))
    for group in sorted(set(fa["groups"]) | set(fb["groups"])):
        ga, gb = fa["groups"].get(group), fb["groups"].get(group)
        if ga is None or gb is None:
            rep.add(name, "XX", f"fit[{group}]", "group", "only in " + ("A" if ga else "B"))
            continue
        if ga["settings_sha"] != gb["settings_sha"]:
            rep.add(name, "C1", f"fit[{group}]", "settings", "")
        if ga["record_without_settings_sha"] != gb["record_without_settings_sha"]:
            cat = "C4" if name == "hknu_unified8" and evidence.get("hknu_smpl24_paired", {}).get("numeric") else "XX"
            detail = (f"constants {'equal' if ga['constants_sha'] == gb['constants_sha'] else 'differ'}; "
                      f"residual A={ga.get('residual_fitted_m')} B={gb.get('residual_fitted_m')}")
            rep.add(name, cat, f"fit[{group}]", "record (constants/residuals)", detail)
    if fa.get("takes", {}) != fb.get("takes", {}):
        rep.add(name, "XX", "fit.takes", "take -> group", f"A={len(fa.get('takes', {}))} B={len(fb.get('takes', {}))}")
    if fa.get("unfitted", {}) != fb.get("unfitted", {}):
        rep.add(name, "XX", "fit.unfitted", "unfitted takes",
                f"A={json.dumps(fa.get('unfitted'))[:100]} B={json.dumps(fb.get('unfitted'))[:100]}")
    ha, hb = fa.get("head"), fb.get("head")
    if same_scope and ha is not None and hb is not None:
        for key in sorted(k for k in set(ha) | set(hb) if ha.get(k) != hb.get(k)):
            rep.add(name, "XX", "<root>/reduced_model_fit.json", f"head {key}",
                    f"A={json.dumps(ha.get(key))[:100]} B={json.dumps(hb.get(key))[:100]}")


def _fit_covers_file(a: dict, b: dict) -> bool:
    """True when both `fit` summaries describe the whole file (format 2, not sample-filtered)."""
    fa, fb = a.get("fit") or {}, b.get("fit") or {}
    return all(f.get("complete") and f.get("head") is not None for f in (fa, fb))


def compare_bundle(rep: Report, name: str, a: dict, b: dict, A: Side, B: Side, evidence: dict) -> None:
    scope_differs = (a.get("counts") or {}).get("total") != (b.get("counts") or {}).get("total")
    rep.stats[name]["takes_A"] = len(a.get("takes", {}))
    rep.stats[name]["takes_B"] = len(b.get("takes", {}))
    # --- takes
    for rel in sorted(set(a.get("takes", {})) | set(b.get("takes", {}))):
        ta, tb = a.get("takes", {}).get(rel), b.get("takes", {}).get(rel)
        if ta is None or tb is None:
            rep.add(name, "XX", rel, "take", "only in " + ("A" if ta else "B"))
            continue
        if ta.get("status") != tb.get("status"):
            rep.add(name, "XX", rel, "status", f"A={ta.get('status')} B={tb.get('status')}")
        fa, fb = ta.get("files", {}), tb.get("files", {})
        for fn in sorted(set(fa) | set(fb)):
            if fn not in fa or fn not in fb:
                rep.add(name, "XX", f"{rel}/{fn}", "file", "only in " + ("A" if fn in fa else "B"))
                continue
            compare_file(rep, name, f"{rel}/{fn}", ("take", rel, fn), fa[fn], fb[fn], A, B, evidence)
        rep.stats[name]["takes_compared"] += 1
    # --- root files
    ra, rb = a.get("root_files", {}), b.get("root_files", {})
    for fn in sorted(set(ra) | set(rb)):
        if fn not in ra or fn not in rb:
            cat = "C0" if fn in STEP_FILES or (fn.startswith("INDEX_shard")) else "XX"
            rep.add(name, cat, f"<root>/{fn}", "file", "only in " + ("A" if fn in ra else "B"))
            continue
        if scope_differs and fn in SCOPE_FILES:
            if ra[fn].get("sha256") != rb[fn].get("sha256"):
                rep.add(name, "C0", f"<root>/{fn}", "file", "aggregate over a different take set")
            continue
        if fn == "INDEX.json":
            compare_index_file(rep, name, a, b, ra[fn], rb[fn], A, B)
            continue
        if fn == "reduced_model_fit.json" and ra[fn].get("sha256") != rb[fn].get("sha256") \
                and _fit_covers_file(a, b):
            rep.stats[name]["fit_file_compared_by_summary"] += 1
            continue
        compare_file(rep, name, f"<root>/{fn}", ("root", fn), ra[fn], rb[fn], A, B, evidence)
    # --- INDEX entries of the compared takes
    ia = {e["rel"]: e for e in a.get("index_sample", [])}
    ib = {e["rel"]: e for e in b.get("index_sample", [])}
    for rel in sorted(set(ia) & set(ib)):
        for key in sorted(set(ia[rel]) | set(ib[rel])):
            va, vb = ia[rel].get(key), ib[rel].get(key)
            if va == vb:
                continue
            cat = _value_category(A, B, va, vb)
            if cat is None:
                rep.stats[name]["C5_absorbed"] += 1
                continue
            if not cat:
                cat = ("C3" if name in CORPUS_OF else "XX") if key == "pair_id" else "XX"
            rep.add(name, cat, f"INDEX[{rel}]", key, f"A={str(va)[:100]!r} B={str(vb)[:100]!r}")
    # --- the fit records of the compared subjects
    fa, fb = a.get("fit"), b.get("fit")
    if fa and fb:
        compare_fit(rep, name, fa, fb, same_scope=not scope_differs, evidence=evidence)


def compare_corpus(rep: Report, name: str, a: dict, b: dict, A: Side, B: Side,
                   both_runs: bool) -> dict:
    """Members of the npz both sides carry, and the run records both carry.

    A file only one side has is XX between two harness runs (`both_runs`), which rebuilt the same
    sample; against an array reference, which covers the whole corpus while a harness run covers
    its sample subjects, it is counted, not judged.

    C4 is declared only against a reference corpus (built under a different CPython/numpy than the
    run); between two harness runs in one environment an HKNU corpus difference is a regression.
    """
    env_differs = not both_runs
    label = f"corpus:{name}"
    fa, fb = a.get("files", {}), b.get("files", {})
    both = sorted(k for k in set(fa) & set(fb) if k.endswith(".npz"))
    rep.stats[label]["npz_compared"] = len(both)
    rep.stats[label]["npz_only_A"] = sum(1 for k in fa if k.endswith(".npz") and k not in fb)
    rep.stats[label]["npz_only_B"] = sum(1 for k in fb if k.endswith(".npz") and k not in fa)
    if both_runs:
        for key in sorted(set(fa) ^ set(fb)):
            rep.add(label, "XX", key, "file", "only in " + ("A" if key in fa else "B"))
    numeric = False
    content_hashes_by_file: dict[str, str] = {}
    for key in both:
        ma, mb = fa[key]["members"], fb[key]["members"]
        same_file = True
        for member in sorted(set(ma) | set(mb)):
            if member not in ma or member not in mb:
                rep.add(label, "XX", key, f"member {member}", "only in " + ("A" if member in ma else "B"))
                same_file = False
                continue
            ea = (ma[member]["dtype"], tuple(ma[member]["shape"]), ma[member]["data_sha256"])
            eb = (mb[member]["dtype"], tuple(mb[member]["shape"]), mb[member]["data_sha256"])
            if ea == eb:
                rep.stats[label]["members_equal"] += 1
                continue
            same_file = False
            parts = [k for k, x, y in zip(("dtype", "shape", "data"), ea, eb) if x != y]
            if Side.full_member(ma[member]) == Side.full_member(mb[member]):
                cat, detail = "C5", ",".join(parts)
            elif member == "content_hashes":
                cat, tokens = content_hashes_category(ma[member].get("value"),
                                                      mb[member].get("value"))
                content_hashes_by_file[key] = cat
                detail = ",".join(parts) + f" ({tokens})"
            elif name == "hknu_smpl24_paired" and env_differs:
                cat, detail = "C4", ",".join(parts)
                if ea[0][1:2] in ("f", "O") or ea[0] == "|O":
                    numeric = True
            else:
                cat, detail = "XX", ",".join(parts)
            rep.add(label, cat, key, f"member {member}", detail)
        rep.stats[label]["npz_members_identical" if same_file else "npz_differs"] += 1
    # the corpus run records (SUMMARY.json, _run.json, _done.json), when both sides carry them
    for key in sorted(k for k in set(fa) & set(fb) if "masked_sha256" in fa[k] and "masked_sha256" in fb[k]):
        if A.masked(fa[key]) == B.masked(fb[key]):
            rep.stats[label]["json_identical"] += 1
            continue
        found = None
        if fa[key].get("subtrees") is not None and fb[key].get("subtrees") is not None:
            found = json_differences(A, B, fa[key], fb[key])
        if found is None:
            rep.add(label, "UL", key, "json", "masked hash differs; no subtree hashes to locate it")
            continue
        for path, note, prefix_only in found:
            if prefix_only:
                cat = "C5"
            else:
                cat = "C4" if name == "hknu_smpl24_paired" and env_differs else "XX"
            rep.add(label, cat, key, f"json {path}", note)
    return {"numeric": numeric, "content_hashes_by_file": content_hashes_by_file}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--normalize-prefix", action="append", default=[],
                        help="OLD=NEW, e.g. tmp/harness/h1/= (repeatable: one per run compared); "
                             "`auto` lets each side remove the prefix its own hash file records")
    parser.add_argument("--a-corpus", action="append", default=[], help="NAME=ARRAY_SHA256.json")
    parser.add_argument("--b-corpus", action="append", default=[], help="NAME=ARRAY_SHA256.json")
    parser.add_argument("--a-detail", default=None, help="hash file lending subtree hashes to A")
    parser.add_argument("--b-detail", default=None, help="hash file lending subtree hashes to B")
    parser.add_argument("--out", default=None, help="write every difference as JSON")
    parser.add_argument("--list", type=int, default=40, help="differences printed per category")
    args = parser.parse_args(argv)

    normalize: list[tuple[str, str]] = []
    for spec in args.normalize_prefix:
        if spec == "auto":
            normalize.append(("auto", ""))
            continue
        if "=" not in spec:
            parser.error("--normalize-prefix takes OLD=NEW or auto")
        old, new = spec.split("=", 1)
        normalize.append((old, new))
    da, db = load(args.a), load(args.b)
    for spec in args.a_corpus:
        attach_corpus(da, spec)
    for spec in args.b_corpus:
        attach_corpus(db, spec)
    A = Side(da, normalize, load(args.a_detail) if args.a_detail else None)
    B = Side(db, normalize, load(args.b_detail) if args.b_detail else None)

    rep = Report()
    evidence = {}
    # A harness run records its scratch directory; the reference hashes (and their detail) do not,
    # and a corpus attached from an ARRAY_SHA256 reference is the production corpus whatever file
    # it was attached to.
    both_runs = "scratch" in da and "scratch" in db
    for name in sorted(set(da["corpora"]) & set(db["corpora"])):
        ca, cb = da["corpora"][name], db["corpora"][name]
        runs = both_runs and not any("array_reference" in c for c in (ca, cb))
        evidence[name] = compare_corpus(rep, name, ca, cb, A, B, runs)
    for name in sorted(set(da["bundles"]) | set(db["bundles"])):
        if name not in da["bundles"] or name not in db["bundles"]:
            rep.add(name, "XX", "<bundle>", "bundle", "only in " + ("A" if name in da["bundles"] else "B"))
            continue
        compare_bundle(rep, name, da["bundles"][name], db["bundles"][name], A, B, evidence)

    print(f"A: {args.a}  (normalized: {A.use_norm}, format {da.get('format', 1)})")
    print(f"B: {args.b}  (normalized: {B.use_norm}, format {db.get('format', 1)})")
    for scope in sorted(rep.stats):
        stats = rep.stats[scope]
        cats = {c: stats[c] for c in (*DECLARED, "XX", "UL") if stats[c]}
        counters = {k: v for k, v in stats.items() if k not in cats}
        print(f"\n== {scope}")
        print("   counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counters.items())))
        print("   differences: " + (", ".join(f"{c}={n}" for c, n in cats.items()) or "none"))
    by_cat = collections.defaultdict(list)
    for item in rep.items:
        by_cat[item["category"]].append(item)
    for cat in (*DECLARED, "XX", "UL"):
        if not by_cat[cat]:
            continue
        print(f"\n-- {cat} {LABELS[cat]}: {len(by_cat[cat])}")
        for item in by_cat[cat][: args.list]:
            print(f"   {item['bundle']:<26} {item['where']:<58} {item['what']:<48} {item['detail'][:200]}")
        if len(by_cat[cat]) > args.list:
            print(f"   ... {len(by_cat[cat]) - args.list} more")
    undeclared = len(by_cat["XX"]) + len(by_cat["UL"])
    print(f"\nVERDICT: {'only declared differences' if not undeclared else f'{undeclared} undeclared/unlocalized differences'}")
    if args.out:
        summary = {scope: dict(stats) for scope, stats in rep.stats.items()}
        pathlib.Path(args.out).write_text(
            json.dumps({"a": args.a, "b": args.b, "normalize_prefix": args.normalize_prefix,
                        "a_normalized": A.use_norm, "b_normalized": B.use_norm, "labels": LABELS,
                        "summary": summary, "items": rep.items}, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8", newline="\n")
    return 1 if undeclared else 0


if __name__ == "__main__":
    raise SystemExit(main())
