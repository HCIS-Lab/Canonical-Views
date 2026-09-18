#!/usr/bin/env python3
"""CPU regression checks for legal targets and both trainer call sites."""
import ast
import importlib.util
from pathlib import Path
import unittest
import torch
from paper_td_target import next_action_value


class TargetTests(unittest.TestCase):
    def test_forbidden_and_already_selected(self):
        q = torch.tensor([[100., 50., -3., -4.]])
        selected = torch.tensor([[True, False, False, False]])
        keep = torch.tensor([[True, False, True, True]])
        self.assertEqual(next_action_value(q, selected, keep, torch.zeros(1)).item(), -3.)
        self.assertEqual(next_action_value(q, selected, keep, torch.zeros(1), 'legacy').item(), 100.)

    def test_exhausted_pool_uses_legal_repeats(self):
        q = torch.tensor([[2., 100., -3.]])
        selected = torch.tensor([[True, False, True]])
        keep = torch.tensor([[True, False, True]])
        self.assertEqual(next_action_value(q, selected, keep, torch.zeros(1)).item(), 2.)

    def test_terminal_target_is_zero(self):
        q = torch.tensor([[100., 200.]])
        selected = torch.ones_like(q).bool()
        self.assertEqual(next_action_value(q, selected, torch.zeros_like(selected), torch.ones(1)).item(), 0.)
        self.assertEqual(next_action_value(q, selected, None, torch.ones(1)).item(), 0.)

    def test_missing_legal_action_rejected(self):
        with self.assertRaises(ValueError):
            next_action_value(torch.ones(1, 3), torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1))

    def test_matches_selector_mask(self):
        root = Path(__file__).resolve().parent
        spec = importlib.util.spec_from_file_location('selector_check', root / 'src/models/mvselect.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        torch.manual_seed(42)
        for _ in range(20):
            feat = torch.randn(12, 8, 3, 1, 1)
            selected = torch.rand(12, 8) > .5
            keep = torch.rand(12, 8) > .5
            keep[:, 0] = True
            _, _, legal = module.setup_args(feat, selected, keep)
            q = torch.randn(12, 8)
            expected = q.masked_fill(~legal, -torch.inf).max(1).values
            torch.testing.assert_close(next_action_value(q, selected, keep, torch.zeros(12)), expected)

    def test_both_rollout_paths_use_helper(self):
        tree = ast.parse((Path(__file__).parent / 'src/trainer.py').read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'next_action_value']
        self.assertEqual(len(calls), 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
