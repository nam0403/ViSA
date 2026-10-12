import ast
import os
import pathlib
import sys
import tempfile
import types
from unittest.mock import MagicMock
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Ensure yacs mock is available for fastreid imports if yacs is not installed
if "yacs" not in sys.modules:
    try:
        import yacs
    except ImportError:
        yacs = types.ModuleType("yacs")
        yacs_config = types.ModuleType("yacs.config")
        yacs_config.CfgNode = MagicMock
        yacs.config = yacs_config
        sys.modules["yacs"] = yacs
        sys.modules["yacs.config"] = yacs_config

import torch
import torch.nn as nn


def load_log_accuracy():
    source_path = ROOT / "fastreid" / "modeling" / "losses" / "cross_entroy_loss.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "log_accuracy"
    )
    namespace = {"get_event_storage": lambda: _Storage()}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["log_accuracy"]


class _Storage:
    def put_scalar(self, name, value):
        pass


class RuntimeFixTest(unittest.TestCase):
    def test_accuracy_clamps_top_k_for_two_class_logits(self):
        calls = []
        logits = torch.tensor([[4.0, 1.0], [0.0, 2.0]])
        original_topk = torch.Tensor.topk

        def recording_topk(tensor, k, *args, **kwargs):
            calls.append(k)
            return original_topk(tensor, k, *args, **kwargs)

        torch.Tensor.topk = recording_topk
        try:
            load_log_accuracy()(logits, torch.tensor([0, 1]), topk=(1, 5))
        finally:
            torch.Tensor.topk = original_topk

        self.assertEqual(calls, [2])

    def test_accuracy_keeps_top_five_for_five_class_logits(self):
        calls = []
        logits = torch.arange(10, dtype=torch.float32).reshape(2, 5)
        original_topk = torch.Tensor.topk

        def recording_topk(tensor, k, *args, **kwargs):
            calls.append(k)
            return original_topk(tensor, k, *args, **kwargs)

        torch.Tensor.topk = recording_topk
        try:
            load_log_accuracy()(logits, torch.tensor([0, 1]), topk=(1, 5))
        finally:
            torch.Tensor.topk = original_topk

        self.assertEqual(calls, [5])

    def test_colab_enables_amp(self):
        config = (ROOT / "configs" / "AGReIDv2" / "visa_colab.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("AMP:\n    ENABLED: True", config)

    def test_amp_explicitly_unscales_before_step(self):
        source = (ROOT / "fastreid" / "engine" / "train_loop.py").read_text(
            encoding="utf-8"
        )
        self.assertLess(
            source.index("self.grad_scaler.unscale_(self.optimizer)"),
            source.index("self.grad_scaler.step(self.optimizer)"),
        )

    def test_vectorized_graph_norm_correctness(self):
        from fastreid.modeling.backbones.vision_transformer_multiview_ViSA import graph_norm

        B, N, P = 2, 4, 16
        A = torch.randn(B, N, P, P)
        A = torch.matmul(A, A.transpose(-1, -2))

        # Check against definition: D^(-1/2) * A * D^(-1/2)
        d = A.sum(-1)
        inv_sqrt_d = torch.pow(d.clamp_min(torch.finfo(A.dtype).eps), -0.5)
        expected = inv_sqrt_d.unsqueeze(-1) * A * inv_sqrt_d.unsqueeze(-2)

        actual = graph_norm(A)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_checkpointer_atomic_save_and_resume(self):
        from fastreid.utils.checkpoint import Checkpointer

        with tempfile.TemporaryDirectory() as tmp_dir:
            model = nn.Linear(10, 2)
            opt = torch.optim.SGD(model.parameters(), lr=0.01)
            scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

            checkpointables = {"optimizer": opt}
            if scaler is not None:
                checkpointables["grad_scaler"] = scaler

            checkpointer = Checkpointer(
                model,
                tmp_dir,
                save_to_disk=True,
                **checkpointables
            )

            checkpointer.save("model_test", epoch=3)

            # Verify files on disk
            save_path = os.path.join(tmp_dir, "model_test.pth")
            last_checkpoint_path = os.path.join(tmp_dir, "last_checkpoint")
            self.assertTrue(os.path.isfile(save_path))
            self.assertTrue(os.path.isfile(last_checkpoint_path))

            with open(last_checkpoint_path, "r") as f:
                self.assertEqual(f.read().strip(), "model_test.pth")

            # Check that no temporary files remain
            tmp_files = [f for f in os.listdir(tmp_dir) if f.startswith(".")]
            self.assertEqual(len(tmp_files), 0)

            # Test resume
            model2 = nn.Linear(10, 2)
            opt2 = torch.optim.SGD(model2.parameters(), lr=0.05)
            checkpointer2 = Checkpointer(model2, tmp_dir, save_to_disk=True, optimizer=opt2)

            resumed_data = checkpointer2.resume_or_load("", resume=True)
            self.assertEqual(resumed_data.get("epoch"), 3)
            # Parameter weights should match
            self.assertTrue(torch.equal(model.weight, model2.weight))
            self.assertTrue(torch.equal(model.bias, model2.bias))


if __name__ == "__main__":
    unittest.main()
