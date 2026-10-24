"""Synthetic tests for the pre-publication file check."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_release.py"
SPEC = importlib.util.spec_from_file_location("release_check", SCRIPT)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class ReleaseCheckTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def inspect(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return CHECK.check_file(self.root, name)

    def test_source_is_allowed(self):
        self.assertEqual(self.inspect("src/example.py", "answer = 7\n"), [])

    def test_forced_data_file_is_rejected(self):
        issues = self.inspect("data/records.json", "{}")
        self.assertIn("excluded data/cache directory", issues)

    def test_csv_is_rejected(self):
        self.assertIn("file type outside the code-only release", self.inspect("trials.csv", "x,y\n"))

    def test_private_path_is_rejected_without_echo(self):
        private = "/" + "home" + "/example/private-project/file"
        issues = self.inspect("config.json", private)
        self.assertIn("private filesystem path", issues)
        self.assertNotIn(private, str(issues))

    def test_token_pattern_is_rejected_without_echo(self):
        token = "gh" + "p_" + "a" * 36
        issues = self.inspect("config.json", token)
        self.assertIn("possible credential (value withheld)", issues)
        self.assertNotIn(token, str(issues))

    def test_symlink_is_rejected(self):
        target = self.root / "original.py"
        target.write_text("x = 1\n")
        (self.root / "linked.py").symlink_to(target)
        self.assertEqual(CHECK.check_file(self.root, "linked.py"), ["symlink"])

    def test_size_limit(self):
        with (self.root / "large.txt").open("wb") as handle:
            handle.truncate(CHECK.MAX_BYTES)
        self.assertIn("file is at least 50 MiB", CHECK.check_file(self.root, "large.txt"))

    def test_path_escape(self):
        self.assertEqual(CHECK.check_file(self.root, "../outside.py"), ["path outside repository"])


if __name__ == "__main__":
    unittest.main()
