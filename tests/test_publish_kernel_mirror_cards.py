# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tools.publish_kernel_mirror_cards import (
    FAMILY_WRITE_SET,
    KERNEL_REPO_OWNED_TARGETS,
    KERNEL_WRITE_SET,
    PROFILES,
    SOURCE_REPOSITORY,
    TARGET_REPO_TYPE,
    WRITE_SET,
    Asset,
    Profile,
    PublicationError,
    canonical_json,
    collect_assets,
    current_matches,
    main,
    publication_report,
    publish,
    resolve_profile,
    validate_readme,
    validate_source_revision,
    validate_svg,
)

ROOT = Path(__file__).resolve().parents[1]
KERNEL_EXPECTED = {
    "szl-formulas": "SZLHOLDINGS/szl-formulas",
}
# Each converged kernel repo's own hf-kernel-publish.yml is the single writer
# of these Hub repos (kernel surface and model twin). Forge must not write them.
KERNEL_REPO_OWNERS = {
    "SZLHOLDINGS/szl-kernels": "szl-holdings/szl-kernels",
    "SZLHOLDINGS/szl-invariants": "szl-holdings/szl-invariants",
    "SZLHOLDINGS/szl-govsign": "szl-holdings/szl-govsign",
    "SZLHOLDINGS/szl-maskmod": "szl-holdings/szl-maskmod",
    "SZLHOLDINGS/szl-receipt-attn": "szl-holdings/szl-receipt-attn",
    "SZLHOLDINGS/szl-blocked": "szl-holdings/szl-blocked",
    "SZLHOLDINGS/szl-ouroboros": "szl-holdings/szl-ouroboros",
    "SZLHOLDINGS/szl-provctl": "szl-holdings/szl-provctl",
    "SZLHOLDINGS/szl-block-kv": "szl-holdings/szl-block-kv",
    "SZLHOLDINGS/YARQA-ATTN": "szl-holdings/YARQA-ATTN",
    "SZLHOLDINGS/szl-lambda-gate": "szl-holdings/szl-lambda-gate",
    "SZLHOLDINGS/szl-governed-norm": "szl-holdings/szl-lambda-gate",
}
WORKFLOW = ROOT / ".github" / "workflows" / "publish-kernel-mirror-cards.yml"
FAMILY_EXPECTED = {
    "khipu-gguf": "SZLHOLDINGS/SZL-Khipu-1.5B-GGUF",
    "nemo": "SZLHOLDINGS/szl-nemo",
    "tinykhipu-nano": "SZLHOLDINGS/TinyKhipu-Nano",
    "receiptagent-nano": "SZLHOLDINGS/ReceiptAgent-Nano",
    "receiptagent-v2": "SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v2",
}
GATED_EXPECTED = {
    "receiptagent": "SZLHOLDINGS/SZL-Forge-1.5B-ReceiptAgent",
}
EXPECTED = {**KERNEL_EXPECTED, **FAMILY_EXPECTED, **GATED_EXPECTED}
SOURCE_REVISION = "1" * 40


def _readme(profile_name: str) -> bytes:
    profile = resolve_profile(profile_name)
    banner = (
        f"https://raw.githubusercontent.com/{SOURCE_REPOSITORY}/main/"
        f"{profile.source_dir}/card/holo-banner.svg"
    )
    return (
        "---\n"
        "library_name: kernels\n"
        "license: apache-2.0\n"
        "---\n\n"
        f'<img src="{banner}" alt="{profile.name}"/>\n'
    ).encode("utf-8")


def _asset(tmp_path: Path, name: str, path_in_repo: str, payload: bytes) -> Asset:
    source = tmp_path / name
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(payload)
    return Asset(
        source_path=source,
        relative_source_path=name,
        path_in_repo=path_in_repo,
        data=payload,
    )


def test_registry_and_write_set_are_closed() -> None:
    assert {name: profile.repo_id for name, profile in PROFILES.items()} == EXPECTED
    assert len({profile.repo_id for profile in PROFILES.values()}) == len(EXPECTED)
    assert all(profile.source_dir == profile.name for profile in PROFILES.values())
    assert TARGET_REPO_TYPE == "model"
    assert WRITE_SET == KERNEL_WRITE_SET == (
        ("README.md", "README.md"),
        ("card/holo-banner.svg", "card/holo-banner.svg"),
    )
    assert FAMILY_WRITE_SET == (
        ("card/README.md", "README.md"),
        ("card/holo-banner.svg", "card/holo-banner.svg"),
    )
    for name in KERNEL_EXPECTED:
        assert PROFILES[name].write_set == KERNEL_WRITE_SET
        assert PROFILES[name].library_name == "kernels"
        assert PROFILES[name].gated is False
    for name in FAMILY_EXPECTED:
        assert PROFILES[name].write_set == FAMILY_WRITE_SET
        assert PROFILES[name].gated is False
    for name in GATED_EXPECTED:
        assert PROFILES[name].write_set == FAMILY_WRITE_SET
        assert PROFILES[name].gated is True
    assert PROFILES["nemo"].library_name is None
    assert PROFILES["khipu-gguf"].library_name == "llama.cpp"
    assert PROFILES["tinykhipu-nano"].library_name == "numpy"
    assert PROFILES["receiptagent-nano"].library_name == "numpy"
    assert PROFILES["receiptagent-v2"].library_name == "peft"
    assert PROFILES["receiptagent"].library_name == "transformers"


def test_kernel_repo_owned_targets_are_closed_and_never_profiles() -> None:
    assert KERNEL_REPO_OWNED_TARGETS == KERNEL_REPO_OWNERS
    targeted = {profile.repo_id for profile in PROFILES.values()}
    assert targeted.isdisjoint(KERNEL_REPO_OWNED_TARGETS)
    assert {name for name in PROFILES if name in {
        repo_id.split("/", 1)[1] for repo_id in KERNEL_REPO_OWNED_TARGETS
    }} == set()


@pytest.mark.parametrize("repo_id", sorted(KERNEL_REPO_OWNERS))
def test_kernel_repo_owned_target_is_refused_before_any_provider_call(
    repo_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = repo_id.split("/", 1)[1]
    profile = Profile(name=name, source_dir=name, repo_id=repo_id)
    owner = KERNEL_REPO_OWNERS[repo_id]
    with pytest.raises(PublicationError, match=re.escape(owner)):
        publish(
            profile=profile,
            source_revision=SOURCE_REVISION,
            assets=[],
            token="masked",
            credential_source="HF_TOKEN",
        )
    monkeypatch.setitem(PROFILES, name, profile)
    with pytest.raises(PublicationError, match="hf-kernel-publish"):
        resolve_profile(name)


@pytest.mark.parametrize(
    "profile_name",
    ["szl-blocked", "szl-govsign", "szl-invariants", "szl-ouroboros", "szl-provctl"],
)
def test_retired_kernel_profiles_are_rejected_by_the_cli(
    profile_name: str, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--profile",
                profile_name,
                "--source-revision",
                SOURCE_REVISION,
                "--root",
                str(ROOT),
                "--report",
                str(tmp_path / "report.json"),
            ]
        )
    assert not (tmp_path / "report.json").exists()


def test_workflow_targets_only_forge_owned_profiles() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    matrix = dict(
        re.findall(
            r"(?m)^ {10}- profile: (\S+)\n {12}repo_id: (\S+)$",
            text,
        )
    )
    publishable = {
        name: profile.repo_id
        for name, profile in PROFILES.items()
        if not profile.gated
    }
    assert matrix == publishable
    dry_run = text.split("for profile in \\\n", 1)[1].split("          do\n", 1)[0]
    assert set(re.findall(r"(?m)^ {12}([a-z0-9-]+)(?: \\)?$", dry_run)) == set(PROFILES)
    for repo_id in KERNEL_REPO_OWNERS:
        assert repo_id not in text
        name = repo_id.split("/", 1)[1]
        assert f'"{name}/README.md"' not in text


@pytest.mark.parametrize("profile_name", sorted(EXPECTED))
def test_all_admitted_repository_assets_validate(profile_name: str) -> None:
    profile = resolve_profile(profile_name)
    assets = collect_assets(ROOT, profile)
    assert [asset.path_in_repo for asset in assets] == [
        "README.md",
        "card/holo-banner.svg",
    ]
    assert all(asset.data for asset in assets)
    assert all(len(asset.evidence()["sha256"]) == 64 for asset in assets)
    if profile_name in KERNEL_EXPECTED:
        assert assets[0].relative_source_path == f"{profile_name}/README.md"
    else:
        assert assets[0].relative_source_path == f"{profile_name}/card/README.md"


def test_readme_requires_exact_profile_banner() -> None:
    profile = resolve_profile("szl-formulas")
    validate_readme(profile, _readme(profile.name))

    foreign = _readme("khipu-gguf")
    with pytest.raises(PublicationError):
        validate_readme(profile, foreign)


@pytest.mark.parametrize(
    "payload",
    [
        b"not frontmatter",
        b"---\nlibrary_name: transformers\nlicense: apache-2.0\n---\n",
        b"---\nlibrary_name: kernels\nlicense: mit\n---\n",
        (
            b"---\nlibrary_name: kernels\nlicense: apache-2.0\n---\n"
            b"<script>alert(1)</script>"
        ),
    ],
)
def test_readme_rejects_unbound_or_active_content(payload: bytes) -> None:
    with pytest.raises(PublicationError):
        validate_readme(resolve_profile("szl-formulas"), payload)


def test_svg_accepts_static_fragment_references() -> None:
    validate_svg(
        b'<svg xmlns="http://www.w3.org/2000/svg">'
        b'<defs><linearGradient id="g"/></defs>'
        b'<rect fill="url(#g)"/></svg>'
    )


@pytest.mark.parametrize(
    "payload",
    [
        b'<svg xmlns="http://www.w3.org/2000/svg"><script/></svg>',
        b'<svg xmlns="http://www.w3.org/2000/svg"><foreignObject/></svg>',
        b'<svg xmlns="http://www.w3.org/2000/svg" onload="alert(1)"/>',
        (
            b'<svg xmlns="http://www.w3.org/2000/svg">'
            b'<image href="https://example.invalid/x.png"/></svg>'
        ),
        (
            b'<svg xmlns="http://www.w3.org/2000/svg">'
            b'<rect style="fill:url(https://example.invalid/x.svg)"/></svg>'
        ),
        (
            b'<!DOCTYPE svg [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
            b'<svg xmlns="http://www.w3.org/2000/svg">&xxe;</svg>'
        ),
        b"<html></html>",
    ],
)
def test_svg_rejects_active_external_or_non_svg_content(payload: bytes) -> None:
    with pytest.raises(PublicationError):
        validate_svg(payload)


def test_current_matches_is_byte_exact(tmp_path: Path) -> None:
    readme = _asset(tmp_path, "source-readme", "README.md", b"readme")
    banner = _asset(
        tmp_path,
        "source-banner",
        "card/holo-banner.svg",
        b"<svg/>",
    )
    hub_readme = tmp_path / "hub-readme"
    hub_banner = tmp_path / "hub-banner"
    hub_readme.write_bytes(readme.data)
    hub_banner.write_bytes(banner.data)
    mapping = {
        "README.md": hub_readme,
        "card/holo-banner.svg": hub_banner,
    }

    def download_fn(**kwargs: object) -> str:
        return str(mapping[str(kwargs["filename"])])

    assert current_matches(
        repo_id="SZLHOLDINGS/szl-formulas",
        revision="2" * 40,
        sibling_paths=mapping,
        assets=[readme, banner],
        token="masked",
        download_fn=download_fn,
    )
    assert not current_matches(
        repo_id="SZLHOLDINGS/szl-formulas",
        revision="2" * 40,
        sibling_paths={"README.md"},
        assets=[readme, banner],
        token="masked",
        download_fn=download_fn,
    )

    hub_banner.write_bytes(b"different")
    assert not current_matches(
        repo_id="SZLHOLDINGS/szl-formulas",
        revision="2" * 40,
        sibling_paths=mapping,
        assets=[readme, banner],
        token="masked",
        download_fn=download_fn,
    )


def test_reports_are_secret_free_and_card_only(tmp_path: Path) -> None:
    asset = _asset(tmp_path, "README.md", "README.md", b"safe")
    error = PublicationError("sensitive-looking hf_example_secret")
    report = publication_report(
        profile=resolve_profile("szl-formulas"),
        source_revision=SOURCE_REVISION,
        assets=[asset],
        status="FAILED_CLOSED",
        credential_source="HF_ORG_TOKEN",
        error=error,
    )
    encoded = canonical_json(report)
    assert "hf_example_secret" not in encoded
    assert report["scope"] == {
        "authority": "CARD_ONLY",
        "write_set": ["README.md"],
        "deletes_allowed": False,
        "model_weights_changed": False,
        "runtime_authority_changed": False,
        "promotion_effect": "NONE",
    }
    assert report["credential"]["token_persisted"] is False
    assert report["credential"]["token_logged"] is False


@pytest.mark.parametrize(
    "revision",
    ["", "abc", "A" * 40, "1" * 39, "1" * 41, "g" * 40],
)
def test_source_revision_must_be_exact_lowercase_sha(revision: str) -> None:
    with pytest.raises(PublicationError):
        validate_source_revision(revision)
    assert validate_source_revision(SOURCE_REVISION) == SOURCE_REVISION


def test_dry_run_cli_writes_machine_readable_evidence(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    result = main(
        [
            "--profile",
            "szl-formulas",
            "--source-revision",
            SOURCE_REVISION,
            "--root",
            str(ROOT),
            "--report",
            str(report_path),
        ]
    )
    assert result == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "VALIDATED_NOT_PUBLISHED"
    assert payload["target"]["repo_id"] == EXPECTED["szl-formulas"]
    assert payload["publication"]["exact_readback"] is False
    assert payload["scope"]["promotion_effect"] == "NONE"
    assert {item["path_in_repo"] for item in payload["files"]} == {
        "README.md",
        "card/holo-banner.svg",
    }


def test_nemo_readme_rejects_library_name() -> None:
    profile = resolve_profile("nemo")
    payload = (
        "---\n"
        "library_name: sklearn\n"
        "license: apache-2.0\n"
        "---\n\n"
        f'<img src="https://raw.githubusercontent.com/{SOURCE_REPOSITORY}/main/'
        f'{profile.source_dir}/card/holo-banner.svg"/>\n'
    ).encode("utf-8")
    with pytest.raises(PublicationError, match="must not declare library_name"):
        validate_readme(profile, payload)


def test_family_readme_requires_declared_library_and_banner() -> None:
    profile = resolve_profile("khipu-gguf")
    banner = (
        f"https://raw.githubusercontent.com/{SOURCE_REPOSITORY}/main/"
        f"{profile.source_dir}/card/holo-banner.svg"
    )
    validate_readme(
        profile,
        (
            "---\n"
            "library_name: llama.cpp\n"
            "license: apache-2.0\n"
            "---\n\n"
            f'<img src="{banner}"/>\n'
        ).encode("utf-8"),
    )
    with pytest.raises(PublicationError):
        validate_readme(
            profile,
            (
                "---\n"
                "library_name: kernels\n"
                "license: apache-2.0\n"
                "---\n\n"
                f'<img src="{banner}"/>\n'
            ).encode("utf-8"),
        )


def test_gated_receiptagent_validates_and_publish_fails_closed(tmp_path: Path) -> None:
    profile = resolve_profile("receiptagent")
    assets = collect_assets(ROOT, profile)
    assert profile.gated is True
    with pytest.raises(PublicationError, match="publish_rights"):
        publish(
            profile=profile,
            source_revision=SOURCE_REVISION,
            assets=assets,
            token="masked",
            credential_source="HF_ORG_TOKEN",
        )
    report_path = tmp_path / "receiptagent-gated.json"
    result = main(
        [
            "--profile",
            "receiptagent",
            "--source-revision",
            SOURCE_REVISION,
            "--root",
            str(ROOT),
            "--report",
            str(report_path),
        ]
    )
    assert result == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "VALIDATED_NOT_PUBLISHED"
    assert payload["target"]["gated"] is True
    assert payload["target"]["repo_id"] == GATED_EXPECTED["receiptagent"]


@pytest.mark.parametrize("profile_name", sorted(FAMILY_EXPECTED))
def test_family_dry_run_cli_validates_without_publishing(profile_name: str, tmp_path: Path) -> None:
    report_path = tmp_path / f"{profile_name}.json"
    result = main(
        [
            "--profile",
            profile_name,
            "--source-revision",
            SOURCE_REVISION,
            "--root",
            str(ROOT),
            "--report",
            str(report_path),
        ]
    )
    assert result == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "VALIDATED_NOT_PUBLISHED"
    assert payload["target"]["repo_id"] == FAMILY_EXPECTED[profile_name]
    assert payload["target"]["gated"] is False
    assert payload["scope"]["promotion_effect"] == "NONE"
    assert payload["publication"]["exact_readback"] is False
