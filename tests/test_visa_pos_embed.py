import unittest

import torch

from fastreid.modeling.backbones.vision_transformer_multiview_ViSA import resize_pos_embed


class ViSAPositionalEmbeddingTest(unittest.TestCase):
    def test_nondistilled_deit_conversion_preserves_view_initialization(self):
        source = torch.arange(197 * 4, dtype=torch.float32).reshape(1, 197, 4)
        target = torch.full((1, 212, 4), -7.0)

        converted = resize_pos_embed(source, target, 21, 10)

        self.assertEqual(converted.shape, target.shape)
        self.assertTrue(torch.equal(converted[:, 0], source[:, 0]))
        self.assertTrue(torch.equal(converted[:, 1], target[:, 1]))
        expected_patch = torch.nn.functional.interpolate(
            source[:, 1:].reshape(1, 14, 14, 4).permute(0, 3, 1, 2),
            size=(21, 10),
            mode='bilinear',
            align_corners=False,
        ).permute(0, 2, 3, 1).reshape(1, 210, 4)
        self.assertTrue(torch.equal(converted[:, 2:], expected_patch))

    def test_distilled_layout_discards_distillation_position(self):
        source = torch.arange(198 * 2, dtype=torch.float32).reshape(1, 198, 2)
        target = torch.zeros((1, 212, 2))

        converted = resize_pos_embed(source, target, 21, 10)

        self.assertTrue(torch.equal(converted[:, 0], source[:, 0]))
        expected_patch = torch.nn.functional.interpolate(
            source[:, 2:].reshape(1, 14, 14, 2).permute(0, 3, 1, 2),
            size=(21, 10),
            mode='bilinear',
            align_corners=False,
        ).permute(0, 2, 3, 1).reshape(1, 210, 2)
        self.assertTrue(torch.equal(converted[:, 2:], expected_patch))

    def test_dimension_mismatch_is_reported(self):
        source = torch.zeros((1, 197, 3))
        target = torch.zeros((1, 212, 4))

        with self.assertRaisesRegex(ValueError, "dimensions do not match"):
            resize_pos_embed(source, target, 21, 10)


if __name__ == "__main__":
    unittest.main()
