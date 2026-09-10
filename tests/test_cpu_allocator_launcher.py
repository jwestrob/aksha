from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from aksha import _launcher as launcher


class AllocatorLauncherTests(unittest.TestCase):
    def test_linux_glibc_detection_is_conservative(self):
        with (
            mock.patch.object(sys, "platform", "darwin"),
            mock.patch.object(launcher.os, "confstr") as confstr,
        ):
            self.assertFalse(launcher._is_linux_glibc())
        confstr.assert_not_called()
        with (
            mock.patch.object(sys, "platform", "linux"),
            mock.patch.object(
                launcher.os, "confstr", return_value="glibc 2.36"
            ),
        ):
            self.assertTrue(launcher._is_linux_glibc())
        with (
            mock.patch.object(sys, "platform", "linux"),
            mock.patch.object(
                launcher.os, "confstr", return_value="musl 1.2"
            ),
        ):
            self.assertFalse(launcher._is_linux_glibc())
        with (
            mock.patch.object(sys, "platform", "linux"),
            mock.patch.object(
                launcher.os, "confstr", side_effect=ValueError
            ),
        ):
            self.assertFalse(launcher._is_linux_glibc())

    def test_requested_threads_preserves_parser_error_ownership(self):
        self.assertEqual(launcher._requested_threads(()), 1)
        self.assertEqual(launcher._requested_threads(("--threads", "64")), 64)
        self.assertEqual(launcher._requested_threads(("--threads=72",)), 72)
        self.assertEqual(
            launcher._requested_threads(
                ("--threads", "32", "--threads=64")
            ),
            64,
        )
        self.assertIsNone(launcher._requested_threads(("--threads",)))
        self.assertIsNone(launcher._requested_threads(("--threads=not-an-int",)))

    def test_default_policy_is_all_high_thread_search_routes(self):
        accepted = (
            ("search", "--prot_in", "proteins.faa", "--threads", "64"),
            ("search", "--threads", "64", "--gpu-manifest", "PFAM=x"),
            ("search", "--threads=64", "--gpu-manifest=PFAM=x"),
            ("search", "--threads=64", "--gpu-m=PFAM=x"),
            ("search", "--threads=64", "--g", "PFAM=x"),
        )
        rejected = (
            (),
            ("initialize", "--threads", "64"),
            ("search", "--threads", "63"),
            ("search", "--threads", "not-an-int"),
        )
        for arguments in accepted:
            with self.subTest(arguments=arguments):
                self.assertTrue(launcher._is_high_thread_search(arguments))
        for arguments in rejected:
            with self.subTest(arguments=arguments):
                self.assertFalse(launcher._is_high_thread_search(arguments))

    def test_default_reexec_environment_preserves_every_existing_value(self):
        source = {"PATH": "/bin", "TOKEN": "opaque"}
        with mock.patch.object(launcher, "_is_linux_glibc", return_value=True):
            result = launcher._reexec_environment(
                ("search", "--threads=64"), source
            )
        self.assertIsNot(result, source)
        self.assertEqual(source, {"PATH": "/bin", "TOKEN": "opaque"})
        self.assertEqual(result["PATH"], "/bin")
        self.assertEqual(result["TOKEN"], "opaque")
        self.assertEqual(result[launcher._GLIBC_ARENA_ENV], "24")
        self.assertEqual(result[launcher._REEXEC_SENTINEL_ENV], "1")

    def test_user_glibc_override_and_recursion_sentinel_always_win(self):
        arguments = ("search", "--threads", "64")
        with mock.patch.object(launcher, "_is_linux_glibc", return_value=True):
            for environment in (
                {launcher._GLIBC_ARENA_ENV: ""},
                {launcher._GLIBC_ARENA_ENV: "8"},
                {launcher._REEXEC_SENTINEL_ENV: "1"},
                {
                    launcher._GLIBC_ARENA_ENV: "8",
                    launcher._ASTRA_ARENA_ENV: "24",
                },
            ):
                with self.subTest(environment=environment):
                    self.assertIsNone(
                        launcher._reexec_environment(arguments, environment)
                    )

    def test_astra_override_accepts_optout_or_canonical_positive_value(self):
        arguments = ("search", "--threads", "64")
        with mock.patch.object(launcher, "_is_linux_glibc", return_value=True):
            self.assertIsNone(
                launcher._reexec_environment(
                    arguments, {launcher._ASTRA_ARENA_ENV: "0"}
                )
            )
            result = launcher._reexec_environment(
                arguments, {launcher._ASTRA_ARENA_ENV: "32"}
            )
        self.assertEqual(result[launcher._GLIBC_ARENA_ENV], "32")

    def test_astra_override_rejects_malformed_or_oversized_values(self):
        arguments = ("search", "--threads", "64")
        invalid = ("", "00", "01", "+1", " 1", "x", str(1 << 31))
        with mock.patch.object(launcher, "_is_linux_glibc", return_value=True):
            for value in invalid:
                with self.subTest(value=value):
                    with self.assertRaisesRegex(
                        ValueError, launcher._ASTRA_ARENA_ENV
                    ):
                        launcher._reexec_environment(
                            arguments,
                            {launcher._ASTRA_ARENA_ENV: value},
                        )

    def test_irrelevant_or_non_glibc_command_does_not_parse_override(self):
        malformed = {launcher._ASTRA_ARENA_ENV: "malformed"}
        with mock.patch.object(launcher, "_is_linux_glibc", return_value=True):
            self.assertIsNone(
                launcher._reexec_environment(("initialize",), malformed)
            )
            with self.assertRaisesRegex(ValueError, launcher._ASTRA_ARENA_ENV):
                launcher._reexec_environment(
                    ("search", "--threads", "64", "--gpu-manifest=x"),
                    malformed,
                )
        with mock.patch.object(launcher, "_is_linux_glibc", return_value=False):
            self.assertIsNone(
                launcher._reexec_environment(
                    ("search", "--threads", "64"), malformed
                )
            )

    def test_main_reexecs_with_exact_arguments_and_augmented_environment(self):
        arguments = [
            "search", "--threads", "64", "--bitscore", "5", "name with space"
        ]
        source = {"PATH": "/usr/bin", "KEEP": "exact"}
        expected_environment = {
            **source,
            launcher._GLIBC_ARENA_ENV: "24",
            launcher._REEXEC_SENTINEL_ENV: "1",
        }
        with (
            mock.patch.object(sys, "argv", ["/opt/aksha", *arguments]),
            mock.patch.object(sys, "executable", "/opt/python"),
            mock.patch.dict(os.environ, source, clear=True),
            mock.patch.object(
                launcher, "_is_linux_glibc", return_value=True
            ),
            mock.patch.object(
                launcher.os, "execvpe", side_effect=OSError("exec intercepted")
            ) as execute,
            mock.patch.object(launcher, "_dispatch") as dispatch,
        ):
            with self.assertRaisesRegex(OSError, "exec intercepted"):
                launcher.main()
        execute.assert_called_once_with(
            "/opt/python",
            ["/opt/python", "-m", "aksha.main", *arguments],
            expected_environment,
        )
        dispatch.assert_not_called()

    def test_main_bypass_preserves_direct_dispatch(self):
        with (
            mock.patch.object(sys, "argv", ["aksha", "search", "--threads", "1"]),
            mock.patch.object(launcher, "_dispatch", return_value=object()) as dispatch,
            mock.patch.object(launcher.os, "execvpe") as execute,
        ):
            expected = launcher.main()
        self.assertIs(expected, dispatch.return_value)
        dispatch.assert_called_once_with()
        execute.assert_not_called()

    def test_launcher_import_does_not_import_pyhmmer_or_runtime_cli(self):
        root = Path(__file__).resolve().parents[1]
        code = (
            "import aksha._launcher, sys; "
            "assert 'astra_pyhmmer' not in sys.modules; "
            "assert 'aksha.search' not in sys.modules; "
            "assert 'aksha.main' not in sys.modules"
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.fspath(root)
        completed = subprocess.run(
            [sys.executable, "-S", "-c", code],
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_reexec_sets_arena_environment_before_pyhmmer_import(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(
            prefix="aksha-allocator-launcher-"
        ) as temporary:
            temporary_path = Path(temporary)
            probe = temporary_path / "startup.tsv"
            (temporary_path / "sitecustomize.py").write_text(
                "import os, sys\n"
                "with open(os.environ['ASTRA_LAUNCHER_PROBE'], 'a') as stream:\n"
                "    stream.write('\\t'.join((\n"
                "        os.environ.get('MALLOC_ARENA_MAX', '-'),\n"
                "        os.environ.get('_ASTRA_CPU_ALLOCATOR_REEXEC', '-'),\n"
                "        str(int('astra_pyhmmer' in sys.modules)),\n"
                "    )) + '\\n')\n",
                encoding="ascii",
            )
            code = (
                "import sys; "
                "sys.argv = ['aksha', 'search', '--threads', '64', '--help']; "
                "from aksha._launcher import main; main()"
            )
            environment = os.environ.copy()
            for name in (
                launcher._GLIBC_ARENA_ENV,
                launcher._ASTRA_ARENA_ENV,
                launcher._REEXEC_SENTINEL_ENV,
            ):
                environment.pop(name, None)
            environment["ASTRA_LAUNCHER_PROBE"] = os.fspath(probe)
            environment["PYTHONPATH"] = os.pathsep.join(
                (os.fspath(temporary_path), os.fspath(root))
            )
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            records = [
                line.split("\t")
                for line in probe.read_text(encoding="ascii").splitlines()
            ]
        self.assertEqual(records, [["-", "-", "0"], ["24", "1", "0"]])

    def test_console_entry_points_to_lightweight_launcher(self):
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        self.assertIn(
            'aksha = "aksha._launcher:main"',
            pyproject.read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
