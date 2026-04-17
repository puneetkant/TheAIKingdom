# 6.4.3 Text-to-Video

Text-to-video generation (Sora, Runway Gen-2, Pika) extends image diffusion to temporal sequences, requiring temporal consistency, motion modelling, and efficient 3D attention. Key challenges include maintaining subject identity and smooth motion across frames. This folder simulates temporal consistency metrics, linear frame interpolation, and SSIM proxy analysis.

## Files

| File | Description |
|------|-------------|
| `working_example2.py` | Temporal consistency MSE, linear frame interpolation, SSIM proxy across frame pairs |
| `working_example.ipynb` | Interactive notebook version |
| `output/` | `text_to_video.png` |

## Quick Start

```bash
python working_example2.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| Temporal consistency | Adjacent frames should be visually smooth |
| 3D attention | Self-attention across (T, H, W); expensive but powerful |
| Frame interpolation | Generate in-between frames for smooth video |
| SSIM | Structural Similarity Index; perceived image quality |
| Latent video diffusion | Compress video to latent space before diffusion |

## Learning Resources

- Ho et al. *Video Diffusion Models* (2022)
- OpenAI *Sora* technical report (2024)
- Brooks et al. *InstructPix2Pix* (2023)
