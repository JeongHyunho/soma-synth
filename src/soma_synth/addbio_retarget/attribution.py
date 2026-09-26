"""Which study an artifact came from, and where that study was published.

AddBiomechanics is CC BY 4.0, which permits this derivative and obliges the attribution. It is
also fifteen studies in one archive, so "AddBiomechanics" is not an answer -- each artifact has
to name the study whose participants it holds.

The folder names do not reliably say which that is. `Hammer2013` is a misspelling of Hamner and
appears only on the With_Arm side. `Tan2021` and `Tan2022` map to publications whose years run
the other way. `Tiziana2019` is a first name, not a surname. A plausible guess cites the wrong
paper, so this table is exact and closed: a token that is not in it fails registration.

More than one archive reaches the same retarget (AddBiomechanics and GAITEX). An archive's own
name, version and citation are per-archive facts, and an artifact uses exactly those fields to say
where it came from, so they live on `Archive` rather than as module constants: a row from one
archive must not assert another's origin.

Transcribed from the attribution crosswalk kept with the project's dataset records, which remains
the authority, including for the non-AddBiomechanics archives it records alongside.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ADDBIOMECHANICS",
    "GAITEX",
    "KNOWN_GAPS",
    "SOURCE_BY_TOKEN",
    "Archive",
    "AttributionSourceUnknown",
    "SourceAttribution",
    "attribution_fields",
    "group_key",
    "resolve_source",
    "study_token",
]

#: Repository policy, not an archive fact: nothing derived here leaves the local plane.
DISTRIBUTION_SCOPE = "internal_only"


@dataclass(frozen=True)
class Archive:
    """The published collection a study's data was distributed in.

    `spec_id` names the retarget bundle the archive's artifacts form. It is part of an artifact's
    identity, so two archives must not share one: a consumer reading `spec_id` is asking which
    corpus a file belongs to.
    """

    name: str
    version: str
    license_id: str
    dataset_citation: str
    spec_id: str


ADDBIOMECHANICS = Archive(
    name="addbiomechanics",
    version="1.0_core",
    license_id="CC-BY-4.0",
    dataset_citation=(
        "AddBiomechanics Dataset 1.0 Core, https://addbiomechanics.org/download_data.html"
    ),
    spec_id="addbio_smpl24_raw",
)

#: GAITEX is CC BY 4.0 too, but the project's licence note for GAITEX
#: requires the Zenodo dataset DOI, not the Scientific Data article -- the article is the
#: publication, the Zenodo record is the dataset being credited.
GAITEX = Archive(
    name="gaitex",
    version="1.0.0",
    license_id="CC-BY-4.0",
    dataset_citation=(
        "Munz, M., Spilz, A. and Oppel, H., Dataset GAITEX, Zenodo, "
        "https://doi.org/10.5281/zenodo.15729056"
    ),
    spec_id="gaitex_smpl24_raw",
)


class AttributionSourceUnknown(KeyError):
    """A study folder token that is not in the registry. Registration fails; nothing is guessed."""


@dataclass(frozen=True)
class SourceAttribution:
    token: str
    stable_id: str
    publication: str
    archive: Archive


def _entries() -> dict[str, SourceAttribution]:
    studies: list[tuple[str, str, tuple[str, ...]]] = [
        ("ab_lencioni_2019", "https://www.nature.com/articles/s41597-019-0323-z",
         ("Tiziana2019_Formatted_No_Arm", "Tiziana2019_Formatted_With_Arm")),
        ("ab_carter_2023", "https://doi.org/10.15125/BATH-01341",
         ("Carter2023_Formatted_No_Arm", "Carter2023_Formatted_With_Arm")),
        ("ab_santos_2017", "https://pubmed.ncbi.nlm.nih.gov/28761798/",
         ("Santos2017_Formatted_No_Arm",)),
        ("ab_camargo_2021",
         "https://www.sciencedirect.com/science/article/pii/S0021929021001007",
         ("Camargo2021_Formatted_No_Arm",)),
        # folder says 2022, the publication is 2023
        ("ab_tan_2023", "https://ieeexplore.ieee.org/document/9826418/",
         ("Tan2022_Formatted_No_Arm",)),
        ("ab_moore_2015", "https://pubmed.ncbi.nlm.nih.gov/25945311/",
         ("Moore2015_Formatted_No_Arm", "Moore2015_Formatted_With_Arm")),
        ("ab_falisse_2016", "https://pubmed.ncbi.nlm.nih.gov/27875132/",
         ("Falisse2017_Formatted_No_Arm", "Falisse2017_Formatted_With_Arm")),
        # `Hammer` is the With_Arm folder's spelling of Hamner; one publication, two spellings
        ("ab_hamner_2013", "https://pubmed.ncbi.nlm.nih.gov/23246045/",
         ("Hamner2013_Formatted_No_Arm", "Hammer2013_Formatted_With_Arm")),
        ("ab_van_der_zee_2022", "https://www.nature.com/articles/s41597-022-01817-1",
         ("vanderZee2022_Formatted_No_Arm", "vanderZee2022_Formatted_With_Arm")),
        ("ab_uhlrich_2023",
         "https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011462",
         ("Uhlrich2023_Formatted_No_Arm", "Uhlrich2023_Formatted_With_Arm")),
        # folder says 2021, the publication is 2022 -- the opposite way round from Tan2022
        ("ab_tan_2022", "https://pubmed.ncbi.nlm.nih.gov/35594817/",
         ("Tan2021_Formatted_No_Arm",)),
        ("ab_wang_2023",
         "https://www.cambridge.org/core/journals/wearable-technologies/article/"
         "wearable-realtime-kinetic-measurement-sensor-setup-for-human-locomotion/"
         "488C21B7706FFDFA7FFAB387FD0A1A64",
         ("Wang2023_Formatted_No_Arm",)),
        ("ab_han_2023", "https://arxiv.org/abs/2310.03930",
         ("Han2023_Formatted_No_Arm", "Han2023_Formatted_With_Arm")),
        ("ab_fregly_2012", "https://pubmed.ncbi.nlm.nih.gov/22161745/",
         ("Fregly2012_Formatted_No_Arm", "Fregly2012_Formatted_With_Arm")),
        ("ab_li_2021", "https://pubmed.ncbi.nlm.nih.gov/33490046/",
         ("Li2021_Formatted_No_Arm", "Li2021_Formatted_With_Arm")),
    ]
    table: dict[str, SourceAttribution] = {}
    for stable_id, publication, tokens in studies:
        for token in tokens:
            table[token] = SourceAttribution(token, stable_id, publication, ADDBIOMECHANICS)

    # A separate archive that reaches the same retarget. It is registered here rather than given
    # a second registry: one crosswalk means one place a token can fail to be known, which is the
    # property this table is for. It is listed apart from the loop above because it carries a
    # different archive, and that is the distinction the loop would otherwise hide.
    table["gaitex"] = SourceAttribution(
        "gaitex",
        "gaitex_munz_2025",
        "https://www.nature.com/articles/s41597-025-06439-x",
        GAITEX,
    )
    return table


SOURCE_BY_TOKEN = _entries()

#: Participants the archive does not have. Recorded so a later count mismatch reads as known
#: rather than as loss -- and so nobody duplicates a subject to make the numbers line up.
KNOWN_GAPS = {
    "Tan2021_Formatted_No_Arm": "participant s2 is absent (the archive has subjects s1, s3-s9)",
    "Fregly2012_Formatted_With_Arm": "With_Arm subset is incomplete: 4 of 6 participants",
}


def study_token(folder_name: str) -> str:
    """The study's token in the AddBiomechanics 1.0 Core archive: its folder name as released."""
    return folder_name.strip()


def resolve_source(folder_name: str) -> SourceAttribution:
    token = study_token(folder_name)
    try:
        return SOURCE_BY_TOKEN[token]
    except KeyError as exc:
        raise AttributionSourceUnknown(
            f"ATTRIBUTION_SOURCE_UNKNOWN: {token!r} is not a registered token in any archive "
            "this crosswalk knows. Nothing is matched by similarity; add it to the registry "
            "crosswalk first, with the archive it came from."
        ) from exc


def group_key(folder_name: str, subject: str) -> tuple[str, str]:
    """Study and subject, deliberately without the variant.

    Arm and no-arm are the same participants seen through two model coverages, not two sources.
    Splitting on the folder alone would put one person on both sides of a train/test boundary.
    """
    return (resolve_source(folder_name).stable_id, subject)


def attribution_fields(
    folder_name: str,
    *,
    adapter_hash: str,
    config_hash: str,
    code_hash: str,
    model_hash: str,
) -> dict[str, str]:
    """The ten fields the registry requires on every derived artifact."""
    hashes = {
        "adapter": adapter_hash,
        "config": config_hash,
        "code": code_hash,
        "model": model_hash,
    }
    missing = [name for name, value in hashes.items() if not str(value).strip()]
    if missing:
        raise ValueError(
            f"no {', '.join(missing)} hash supplied; the artifact must say what produced it"
        )
    source = resolve_source(folder_name)
    return {
        "source_name": source.archive.name,
        "source_version": source.archive.version,
        "source_token": source.token,
        "stable_source_id": source.stable_id,
        "dataset_citation": source.archive.dataset_citation,
        "publication": source.publication,
        "license_id": source.archive.license_id,
        "changes_made": "true",
        "content_hashes": ";".join(f"{k}={v}" for k, v in hashes.items()),
        "distribution_scope": DISTRIBUTION_SCOPE,
    }
