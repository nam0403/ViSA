import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


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
        import torch

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
        import torch

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

    def test_colab_disables_legacy_amp_path(self):
        config = (ROOT / "configs" / "AGReIDv2" / "visa_colab.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("AMP:\n    ENABLED: False", config)

    def test_amp_explicitly_unscales_before_step(self):
        source = (ROOT / "fastreid" / "engine" / "train_loop.py").read_text(
            encoding="utf-8"
        )
        self.assertLess(
            source.index("self.grad_scaler.unscale_(self.optimizer)"),
            source.index("self.grad_scaler.step(self.optimizer)"),
        )


if __name__ == "__main__":
    unittest.main()
