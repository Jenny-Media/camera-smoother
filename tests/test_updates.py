import json
import subprocess
import unittest
from unittest.mock import patch

from smoother.jobs import SmootherError
from smoother.updates import check_update


class UpdateCheckTests(unittest.TestCase):
    @patch("smoother.updates.__version__", "0.1.9")
    @patch("smoother.updates.subprocess.run")
    def test_reports_newer_stable_release_numerically(self, run):
        run.return_value.stdout = json.dumps({"tag_name": "v0.1.10"})
        result = check_update()
        self.assertEqual(result["installed_version"], "0.1.9")
        self.assertEqual(result["latest_version"], "0.1.10")
        self.assertTrue(result["update_available"])
        self.assertTrue(result["release_url"].endswith("/v0.1.10"))

    @patch("smoother.updates.__version__", "0.1.9")
    @patch("smoother.updates.subprocess.run")
    def test_same_or_older_release_does_not_trigger_update(self, run):
        for tag in ("v0.1.9", "v0.1.8"):
            with self.subTest(tag=tag):
                run.return_value.stdout = json.dumps({"tag_name": tag})
                self.assertFalse(check_update()["update_available"])

    @patch("smoother.updates.subprocess.run")
    def test_rejects_unexpected_response(self, run):
        for value in ({}, {"tag_name": 7}, {"tag_name": "v0.1.10-rc1"}):
            with self.subTest(value=value):
                run.return_value.stdout = json.dumps(value)
                with self.assertRaisesRegex(SmootherError, "Update check failed"):
                    check_update()

    @patch("smoother.updates.subprocess.run")
    def test_reports_network_failure(self, run):
        run.side_effect = subprocess.CalledProcessError(28, ["curl"], stderr="Timed out")
        with self.assertRaisesRegex(SmootherError, "Timed out"):
            check_update()
