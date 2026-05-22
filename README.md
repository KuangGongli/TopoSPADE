# TopoSPADE

Terrain grayscale image synthesis built on the full **pix2pixHD** framework (Wang et al., CVPR 2018). TopoSPADE extends the standard GAN pipeline—multi-scale discriminator, feature matching, and VGG perceptual loss—with a structure-aware encoder, SPADE conditioning, and height-consistency losses tailored for terrain generation.

---

## Experimental Environment

| Item | Configuration |
|------|---------------|
| OS | Windows 11 |
| CPU | 12th Gen Intel(R) Core(TM) i5-12400F @ 2.50 GHz |
| RAM | 32 GB |
| GPU | NVIDIA GeForce RTX 4060 (8 GB VRAM) |
| Language | Python 3.12 |
| Deep learning framework | PyTorch 2.0.1 |

```bash
pip install -r requirements.txt
```

---

## Data and Preprocessing

- Terrain samples are preprocessed to network-compatible sizes during data preparation; train/test splits are defined at that stage (**datasets are not included in this repository**).
- Dataset root: `datasets/TopoSPADE/`
- pix2pixHD aligned layout: `train_A`, `train_B`, `train_inst` (and `test_*` for evaluation)
- **Default input scaling**: resize to **1024 px width** while preserving aspect ratio (`--resize_or_crop scale_width --loadSize 1024`); no fixed square crop by default.

---

## Data Augmentation (Training Only)

To improve generalization and reduce overfitting, training applies random:

- Horizontal flip
- Vertical flip
- Small-angle rotation (default max ±5°, `--rotate_deg 5`)

Augmentation is disabled at test time. Use `--no_flip` to disable all flips and rotation during training.

---

## Training Configuration

| Item | Setting |
|------|---------|
| Optimizer | Adam |
| Initial learning rate | 0.0002 |
| β₁ | 0.5 |
| β₂ | 0.999 |
| Batch size | 1 (limited by 8 GB VRAM and resolution) |
| Total epochs | 200 (`--niter 100` + `--niter_decay 100`) |
| LR schedule | Fixed LR for first 100 epochs; linear decay to 0 over the next 100 |
| Feature matching & VGG perceptual | Shared weight **λ_feat = 10** |
| Height-consistency loss weight | Default **10** (pixel L₁; plus Sobel gradient consistency with input height when a height channel is present) |
| Backbone | Full pix2pixHD (GlobalGenerator + LocalEnhancer, multi-scale discriminator, instance maps, etc.) |

### Training

Defaults match the paper; specify the dataset path and enable structure-aware modules:

```bash
python train.py --name TopoSPADE --dataroot ./datasets/TopoSPADE --use_structure_aware
```

Equivalent explicit command:

```bash
python train.py --label_nc 0 --input_nc 4 --output_nc 1 --name TopoSPADE --dataroot ./datasets/TopoSPADE --resize_or_crop scale_width --loadSize 1024 --batchSize 1 --niter 100 --niter_decay 100 --lr 0.0002 --beta1 0.5 --beta2 0.999 --lambda_feat 10.0 --height_loss_weight 10.0 --use_structure_aware
```

Checkpoints are saved under `checkpoints/TopoSPADE/`.

---

## Testing / Inference

```bash
python test.py --name TopoSPADE --dataroot ./datasets/TopoSPADE --use_structure_aware --which_epoch latest
```

---

## Loss Functions

1. **GAN loss** + **multi-scale discriminator feature matching** (λ_feat = 10) + **VGG perceptual loss** (also scaled by λ_feat = 10; see `models/pix2pixHD_model.py`).
2. **Height-consistency loss** (`--height_loss_weight`, default 10):
   - Pixel **L₁** between generated and real images;
   - When the input label includes a 4th channel (encoded height), **Sobel gradient** consistency between the output and the input height channel (gradient term weight 0.3, summed with L₁, then multiplied by `height_loss_weight`).

---

## Repository Layout

```
TopoSPADE/
├── train.py
├── test.py
├── models/
├── options/
├── data/
└── util/
```

## Not Included

- Datasets, split scripts, checkpoints, or test outputs
- Ablation experiments, ArcGIS tools, or other auxiliary scripts
