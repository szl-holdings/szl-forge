# SPDX-License-Identifier: Apache-2.0
"""Offline exact-main checks; all Git operations are synthetic."""
from types import SimpleNamespace
import subprocess

import pytest

from tools import publish_khipu_card as publisher

REVISION = 'a' * 40
OTHER = 'b' * 40
MAIN = 'refs/heads/main'


def install_git(monkeypatch, *, local=REVISION, remote=None, fail_at=None, error=None):
    calls = []
    remote = f'{REVISION}\t{MAIN}\n' if remote is None else remote

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs['cwd'] == publisher.ROOT
        assert kwargs['timeout'] == 30
        assert kwargs['check'] is False
        if len(calls) == fail_at:
            if error:
                raise error
            return SimpleNamespace(returncode=1, stdout='', stderr='synthetic failure')
        return SimpleNamespace(returncode=0, stdout=local+'\n' if len(calls)==1 else remote, stderr='')

    monkeypatch.setattr(publisher.subprocess,'run',run)
    return calls


def test_fresh_main_uses_exact_canonical_public_source(monkeypatch):
    calls = install_git(monkeypatch)
    publisher.assert_current_main(REVISION)
    assert calls == [
        ['git','rev-parse','--verify','HEAD'],
        ['git','ls-remote','--exit-code','https://github.com/szl-holdings/szl-forge.git',MAIN],
    ]


@pytest.mark.parametrize('expected', ['', 'main', 'a'*39, 'A'*40])
def test_missing_or_nonimmutable_expected_source_never_queries_git(monkeypatch, expected):
    calls = install_git(monkeypatch)
    with pytest.raises(publisher.PublicationError, match='exact source revision'):
        publisher.assert_current_main(expected)
    assert calls == []


@pytest.mark.parametrize('local', ['', OTHER, REVISION+'\n'+OTHER])
def test_wrong_or_missing_checkout_fails_before_remote_query(monkeypatch, local):
    calls = install_git(monkeypatch,local=local)
    with pytest.raises(publisher.PublicationError, match='checkout does not match'):
        publisher.assert_current_main(REVISION)
    assert len(calls)==1


@pytest.mark.parametrize('remote', ['', 'main', 'a'*39+'\t'+MAIN, REVISION+'\trefs/heads/dev',
                                    REVISION+'\t'+MAIN+'\n'+OTHER+'\t'+MAIN])
def test_missing_or_ambiguous_remote_main_fails_closed(monkeypatch, remote):
    install_git(monkeypatch,remote=remote)
    with pytest.raises(publisher.PublicationError, match='one exact revision'):
        publisher.assert_current_main(REVISION)


def test_stale_remote_main_fails_closed(monkeypatch):
    install_git(monkeypatch,remote=OTHER+'\t'+MAIN)
    with pytest.raises(publisher.PublicationError, match='no longer owns current main'):
        publisher.assert_current_main(REVISION)


@pytest.mark.parametrize('fail_at', [1,2])
@pytest.mark.parametrize('error', [None,OSError('synthetic missing git'),subprocess.TimeoutExpired('git',30)])
def test_git_lookup_failure_fails_closed_without_retry(monkeypatch,fail_at,error):
    calls = install_git(monkeypatch,fail_at=fail_at,error=error)
    with pytest.raises(publisher.PublicationError, match='lookup failed'):
        publisher.assert_current_main(REVISION)
    assert len(calls)==fail_at


def test_workflow_checks_main_before_acquiring_credentials():
    import yaml
    workflow = yaml.safe_load((publisher.ROOT/'.github/workflows/publish-khipu-card.yml').read_text())
    steps = workflow['jobs']['publish']['steps']
    guard_index = next(i for i,s in enumerate(steps) if s['name']=='Require exact fresh canonical main before credential acquisition')
    auth_index = next(i for i,s in enumerate(steps) if s['name']=='Acquire and actively validate model publisher credential')
    assert guard_index < auth_index
    assert 'assert_current_main(os.environ["SOURCE_REVISION"])' in steps[guard_index]['run']
    assert steps[guard_index]['env']['SOURCE_REVISION']=='${{ github.sha }}'
    events = workflow.get('on',workflow.get(True))
    for event in ('pull_request','push'):
        assert 'tests/test_card_publisher_concurrency.py' in events[event]['paths']
        assert 'tests/test_khipu_fresh_main.py' in events[event]['paths']
