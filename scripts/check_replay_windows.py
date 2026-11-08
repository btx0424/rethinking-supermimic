"""CPU replay regression checks. Run: PYTHONPATH=src python scripts/check_replay_windows.py."""

from types import SimpleNamespace
import unittest

import torch

from supermimic.supermimic.buffer import ReplayBuffer


class ReplayWindowTests(unittest.TestCase):
    def make_buffer(self, capacity=8):
        task = SimpleNamespace(
            has_object=True, state_dim=1, n_body=1, num_joints=1,
            xi_dim=1, n_contact=1, object_state_dim=1,
        )
        return ReplayBuffer(capacity, 1, "cpu", task)

    def add(self, buf, step, episode, terminal=False):
        value = torch.tensor([[float(step)]])
        successor = torch.tensor([[-100.0 if terminal else float(step + 1)]])
        buf.add({
            "s": value, "s1": successor,
            "body_pos": torch.zeros(1, 1, 3), "body_quat": torch.zeros(1, 1, 4),
            "a": value, "a_prev": value, "xi": value,
            "c": torch.zeros(1, 1), "c1": torch.full((1, 1), float(terminal)),
            "f": torch.tensor([step]), "ep": torch.tensor([episode]),
            "step": torch.tensor([step]), "obj": value, "obj1": successor,
        })

    def test_terminal_successor_is_supervised_without_crossing_reset(self):
        buf = self.make_buffer()
        for step in range(6):
            self.add(buf, step, step // 3, terminal=step in (2, 5))
        buf.prepare(3)
        self.assertEqual(buf._window_candidates.tolist(), [0, 3])
        batch = buf.sample_windows(16, 3)
        self.assertTrue(torch.all(batch["s1"][:, -1] == -100))
        self.assertTrue(torch.all(batch["obj1"][:, -1] == -100))
        self.assertTrue(torch.all(batch["c1"][:, -1] == 1))
        self.assertTrue(torch.all(batch["s"][..., 0].diff(dim=1) == 1))

    def test_one_step_includes_every_written_transition(self):
        buf = self.make_buffer()
        for step in range(3):
            self.add(buf, step, step, terminal=True)
        buf.prepare(1)
        self.assertEqual(buf._window_candidates.tolist(), [0, 1, 2])

    def test_exact_length_window_needs_no_extra_row(self):
        buf = self.make_buffer()
        for step in range(3):
            self.add(buf, step, 0, terminal=step == 2)
        buf.prepare(3)
        self.assertEqual(buf._window_candidates.tolist(), [0])
        buf.prepare(4)
        self.assertEqual(buf._window_candidates.numel(), 0)

    def test_wrapped_windows_preserve_chronology_and_terminal_successor(self):
        buf = self.make_buffer(capacity=5)
        for step in range(7):
            self.add(buf, step, int(step >= 6), terminal=step == 5)
        # Physical rows contain steps [5, 6, 2, 3, 4]. The terminal row at
        # index 0 belongs to the valid wrapped window [3, 4, 5].
        buf.prepare(3)
        self.assertEqual(buf._window_candidates.tolist(), [2, 3])
        buf.prepare(4)
        self.assertEqual(buf._window_candidates.tolist(), [2])
        batch = buf.sample_windows(1, 4)
        self.assertEqual(batch["s"][0, :, 0].tolist(), [2, 3, 4, 5])
        self.assertEqual(batch["s1"][0, -1, 0].item(), -100)

    def test_gaps_and_imagination_reference_bounds(self):
        buf = self.make_buffer()
        for step in (0, 1, 4):
            self.add(buf, step, 0)
        buf.prepare(2, start_horizon=2, clip_last=torch.full_like(buf.f, 4))
        self.assertEqual(buf._window_candidates.tolist(), [0])
        self.assertEqual(buf._start_candidates.tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
