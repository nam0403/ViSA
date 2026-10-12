# ViSA AG-ReID v2 Colab baseline

The Colab-specific configuration uses the dataset, pretrained checkpoint, and
output locations below:

- Dataset: `/content/datasets/AGReIDv2/AG-ReID.v2`
- DeiT-Base/16 checkpoint: `/content/drive/MyDrive/ViSA/pretrained/deit_base_patch16_224-b5f2ef4d.pth`
- Checkpoints and logs: `/content/drive/MyDrive/ViSA/checkpoints/AGReIDv2/ViSA-Colab`

These paths are expected to exist only in the Colab runtime after mounting
Google Drive and extracting the dataset. The original `visa.yml` remains the
baseline; `visa_colab.yml` keeps its architecture, evaluation protocols, input
size, stride, and 120-epoch schedule, while using an initial training batch
size of 8.

Run from the repository root in Colab:

```bash
python tools/train_net.py --config-file configs/AGReIDv2/visa_colab.yml
```
