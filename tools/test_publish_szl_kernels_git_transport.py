#!/usr/bin/env python3
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import publish_szl_kernels as publisher


class KernelGitTransportTests(unittest.TestCase):
    def test_fast_forward_push_is_credential_isolated_and_exact(self):
        parent = "a" * 40
        revision = "b" * 40
        token = "hf_test_secret"
        commands = []
        askpass_contents = []

        with tempfile.TemporaryDirectory() as source_dir:
            source = Path(source_dir) / "artifact.bin"
            source.write_bytes(b"kernel-bytes")
            operations = [
                type(
                    "Operation",
                    (),
                    {
                        "path_in_repo": "build/artifact.bin",
                        "path_or_fileobj": source,
                    },
                )(),
                type(
                    "Operation",
                    (),
                    {
                        "path_in_repo": publisher.KERNEL_BINDING_PATH,
                        "path_or_fileobj": io.BytesIO(b'{"source":"exact"}\n'),
                    },
                )(),
            ]

            def fake_run(command, **kwargs):
                commands.append(command)
                askpass_contents.append(
                    Path(kwargs["env"]["GIT_ASKPASS"]).read_text(encoding="utf-8")
                )
                if command[1:] == ["rev-parse", "FETCH_HEAD"]:
                    stdout = parent + "\n"
                    returncode = 0
                elif command[1:] == [
                    "diff",
                    "--cached",
                    "--quiet",
                    "--exit-code",
                ]:
                    stdout = ""
                    returncode = 1
                elif command[1:] == ["rev-parse", "HEAD"]:
                    stdout = revision + "\n"
                    returncode = 0
                elif command[1:] == [
                    "ls-remote",
                    "origin",
                    "refs/heads/main",
                ]:
                    stdout = revision + "\trefs/heads/main\n"
                    returncode = 0
                else:
                    stdout = ""
                    returncode = 0
                return subprocess.CompletedProcess(
                    command, returncode, stdout=stdout, stderr=""
                )

            observed = publisher.publish_kernel_branch_via_git(
                branch="main",
                parent_commit=parent,
                operations=operations,
                token=token,
                run_fn=fake_run,
            )

        self.assertEqual(observed, revision)
        self.assertIn(
            ["git", "push", "origin", "HEAD:refs/heads/main"], commands
        )
        staged = next(command for command in commands if command[1] == "add")
        self.assertEqual(
            staged,
            [
                "git",
                "add",
                "--",
                "build/artifact.bin",
                publisher.KERNEL_BINDING_PATH,
            ],
        )
        flattened = "\n".join(" ".join(command) for command in commands)
        self.assertNotIn(token, flattened)
        self.assertNotIn("--force", flattened)
        self.assertTrue(askpass_contents)
        self.assertTrue(all(token not in script for script in askpass_contents))
        self.assertTrue(all("HF_TOKEN" in script for script in askpass_contents))

    def test_noop_returns_exact_parent_without_push(self):
        parent = "c" * 40
        commands = []

        def fake_run(command, **kwargs):
            commands.append(command)
            if command[1:] == ["rev-parse", "FETCH_HEAD"]:
                stdout = parent + "\n"
            else:
                stdout = ""
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

        observed = publisher.publish_kernel_branch_via_git(
            branch="v1",
            parent_commit=parent,
            operations=[
                type(
                    "Operation",
                    (),
                    {
                        "path_in_repo": "source-binding.json",
                        "path_or_fileobj": io.BytesIO(b"same"),
                    },
                )()
            ],
            token="hf_test_secret",
            run_fn=fake_run,
        )

        self.assertEqual(observed, parent)
        self.assertFalse(any(command[1] == "push" for command in commands))


if __name__ == "__main__":
    unittest.main()
