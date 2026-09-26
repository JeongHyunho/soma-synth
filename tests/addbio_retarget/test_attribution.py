"""Attribution, which is a licence obligation rather than a nicety.

AddBiomechanics is CC BY 4.0 and is fifteen studies in one archive, so every artifact has to
name which study it came from and where that study was published. The folder names cannot be
trusted to say: `Hammer2013` is a misspelling of Hamner, and `Tan2021` and `Tan2022` map to
publications whose years are the other way round. Getting it wrong cites the wrong paper.

So the table is exact and closed. An unrecognised token fails registration rather than being
matched to whatever looks closest.

A second archive now passes through the same retarget, which makes a second thing checkable:
that a row from one archive does not carry another archive's identity. A count alone cannot see
that -- the count was right while every GAITEX artifact claimed to be AddBiomechanics -- so the
archive-identity test below is the one that matters.
"""

import pytest

from soma_synth.addbio_retarget.attribution import (
    ADDBIOMECHANICS,
    GAITEX,
    AttributionSourceUnknown,
    attribution_fields,
    resolve_source,
    study_token,
)


def test_the_folder_name_yields_the_local_core_token():
    assert study_token("Carter2023_Formatted_With_Arm") == "Carter2023_Formatted_With_Arm"
    assert resolve_source("Carter2023_Formatted_With_Arm").stable_id == "ab_carter_2023"


def test_the_misspelled_folder_still_reaches_hamner():
    """`Hammer2013` is the With_Arm folder's spelling; both variants are one publication."""
    left = resolve_source("Hamner2013_Formatted_No_Arm")
    right = resolve_source("Hammer2013_Formatted_With_Arm")
    assert left.stable_id == right.stable_id == "ab_hamner_2013"
    assert left.publication == right.publication
    assert "23246045" in left.publication


def test_the_two_tan_folders_do_not_take_each_others_paper():
    """Folder year and publication year are swapped here; a plausible guess cites the wrong one."""
    assert resolve_source("Tan2021_Formatted_No_Arm").stable_id == "ab_tan_2022"
    assert resolve_source("Tan2022_Formatted_No_Arm").stable_id == "ab_tan_2023"
    assert "35594817" in resolve_source("Tan2021_Formatted_No_Arm").publication
    assert "9826418" in resolve_source("Tan2022_Formatted_No_Arm").publication


def test_the_other_year_and_name_aliases_resolve():
    assert resolve_source("Tiziana2019_Formatted_No_Arm").stable_id == "ab_lencioni_2019"
    assert resolve_source("Falisse2017_Formatted_With_Arm").stable_id == "ab_falisse_2016"
    assert resolve_source("vanderZee2022_Formatted_No_Arm").stable_id == "ab_van_der_zee_2022"


def test_an_unknown_token_fails_registration_rather_than_guessing():
    with pytest.raises(AttributionSourceUnknown, match="ATTRIBUTION_SOURCE_UNKNOWN"):
        resolve_source("Carter2024_Formatted_No_Arm")
    # nearly-right spellings must not be absorbed by similarity
    with pytest.raises(AttributionSourceUnknown):
        resolve_source("Hamner2013_Formatted_With_Arm")


def test_every_registered_token_belongs_to_exactly_one_study():
    from soma_synth.addbio_retarget.attribution import SOURCE_BY_TOKEN

    # 25 AddBiomechanics folder tokens across 15 studies, plus the one GAITEX token.
    assert len(SOURCE_BY_TOKEN) == 26
    assert len({s.stable_id for s in SOURCE_BY_TOKEN.values()}) == 16


def test_no_token_carries_another_archives_identity():
    """The check a count cannot make.

    `stable_id` is set on the study row; the citation, name and version come from the archive
    object. They are set in two different places, so agreeing is evidence rather than tautology.
    A GAITEX row carrying `dataset_citation` "AddBiomechanics Dataset 1.0 Core" -- which is what
    shipped in 72 artifacts -- fails here.
    """
    from soma_synth.addbio_retarget.attribution import SOURCE_BY_TOKEN

    # The prefix of a stable id names the archive its study was published in.
    archive_by_prefix = {"ab_": ADDBIOMECHANICS, "gaitex_": GAITEX}

    for token, source in SOURCE_BY_TOKEN.items():
        prefixes = [p for p in archive_by_prefix if source.stable_id.startswith(p)]
        assert len(prefixes) == 1, f"{token}: {source.stable_id} names no known archive"
        expected = archive_by_prefix[prefixes[0]]
        assert source.archive == expected, token

        fields = attribution_fields(
            token, adapter_hash="a", config_hash="b", code_hash="c", model_hash="d")
        assert fields["source_name"] == expected.name, token
        assert fields["source_version"] == expected.version, token
        assert fields["dataset_citation"] == expected.dataset_citation, token

        # and the citation must not name a DIFFERENT archive
        for other in archive_by_prefix.values():
            if other is expected:
                continue
            assert other.name.lower() not in fields["dataset_citation"].lower(), token


def test_two_archives_do_not_share_one_corpus_identity():
    """`spec_id` says which corpus a file belongs to. Sharing one would merge two corpora."""
    assert ADDBIOMECHANICS.spec_id != GAITEX.spec_id
    assert ADDBIOMECHANICS.dataset_citation != GAITEX.dataset_citation


def test_the_gaitex_citation_credits_the_zenodo_record_the_licence_names():
    """GAITEX's licence note conditions redistribution on this exact credit.

    The Scientific Data article is the `publication`; the licence note asks for the dataset, so
    the DOI here is the Zenodo record and not the article.
    """
    fields = attribution_fields(
        "gaitex", adapter_hash="a", config_hash="b", code_hash="c", model_hash="d")
    citation = fields["dataset_citation"]
    assert "10.5281/zenodo.15729056" in citation
    for surname in ("Munz", "Spilz", "Oppel"):
        assert surname in citation
    assert fields["publication"] == "https://www.nature.com/articles/s41597-025-06439-x"


def test_the_artifact_carries_all_ten_required_fields():
    fields = attribution_fields(
        "Moore2015_Formatted_No_Arm",
        adapter_hash="a" * 8,
        config_hash="b" * 8,
        code_hash="c" * 8,
        model_hash="d" * 8,
    )
    for key in (
        "source_name", "source_version", "source_token", "stable_source_id",
        "dataset_citation", "publication", "license_id", "changes_made",
        "content_hashes", "distribution_scope",
    ):
        assert key in fields, key
    assert fields["source_name"] == "addbiomechanics"
    assert fields["source_version"] == "1.0_core"
    assert fields["license_id"] == ADDBIOMECHANICS.license_id == "CC-BY-4.0"
    assert fields["changes_made"] == "true"
    assert fields["distribution_scope"] == "internal_only"
    assert fields["dataset_citation"] == ADDBIOMECHANICS.dataset_citation


def test_the_hashes_of_what_produced_it_are_all_present():
    fields = attribution_fields(
        "Li2021_Formatted_With_Arm",
        adapter_hash="1", config_hash="2", code_hash="3", model_hash="4",
    )
    assert fields["content_hashes"] == "adapter=1;config=2;code=3;model=4"


def test_a_missing_hash_is_refused_rather_than_written_as_empty():
    with pytest.raises(ValueError, match="hash"):
        attribution_fields(
            "Li2021_Formatted_With_Arm",
            adapter_hash="1", config_hash="", code_hash="3", model_hash="4",
        )


def test_the_known_gaps_are_recorded_and_not_to_be_filled():
    from soma_synth.addbio_retarget.attribution import KNOWN_GAPS

    assert "Tan2021_Formatted_No_Arm" in KNOWN_GAPS
    assert "Fregly2012_Formatted_With_Arm" in KNOWN_GAPS
    assert "s2" in KNOWN_GAPS["Tan2021_Formatted_No_Arm"]


def test_the_variant_is_coverage_not_identity():
    """Arm and no-arm are the same participants; grouping by study alone would leak splits."""
    from soma_synth.addbio_retarget.attribution import group_key

    left = group_key("Carter2023_Formatted_No_Arm", "P002_split0")
    right = group_key("Carter2023_Formatted_With_Arm", "P002_split0")
    assert left == right
    assert group_key("Carter2023_Formatted_No_Arm", "P010_split0") != left
