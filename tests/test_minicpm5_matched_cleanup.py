"""No native execution: prove owned process-group cleanup on success/timeout."""
import contextlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from inference import minicpm5_matched as m


class CleanupTests(unittest.TestCase):
    def test_normal_child_exit_still_cleans_native_group(self):
        with patch.object(m.subprocess, 'Popen') as start, patch.object(m.os, 'killpg') as kill:
            child = start.return_value
            child.pid = 12345
            child.wait.return_value = 2
            self.assertEqual(m.execute_child(['fixture'], 10), 2)
            self.assertTrue(start.call_args.kwargs['start_new_session'])
            self.assertEqual([c.args for c in kill.call_args_list], [(12345, m.signal.SIGTERM), (12345, m.signal.SIGKILL)])

    def test_timeout_cleans_entire_group_and_is_not_a_success(self):
        with patch.object(m.subprocess, 'Popen') as start, patch.object(m.os, 'killpg') as kill:
            child = start.return_value
            child.pid = 12345
            child.wait.side_effect = [subprocess.TimeoutExpired('fixture', 10), 1, 1]
            with self.assertRaises(subprocess.TimeoutExpired):
                m.execute_child(['fixture'], 10)
            self.assertEqual(kill.call_count, 2)

    def test_preflight_never_reaches_process_or_network_for_bad_source(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(m, 'execute_child') as execute:
            output = Path(directory) / 'record.json'
            output.write_text('{"status":"EXECUTED"}')
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(m.run('main', output), 1)
            execute.assert_not_called()
            self.assertIn('INCOMPLETE', output.read_text())


if __name__ == '__main__':
    unittest.main()
