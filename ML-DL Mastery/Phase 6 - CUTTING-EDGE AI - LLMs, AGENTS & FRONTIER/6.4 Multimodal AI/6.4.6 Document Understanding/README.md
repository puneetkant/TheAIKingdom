# 6.4.6 Document Understanding

Document understanding models (LayoutLM, Donut, Nougat) parse complex documents by combining text, spatial layout, and visual features. Applications include invoice extraction, form parsing, and scientific paper digitisation. This folder implements bounding box IoU scoring, reading-order detection, and element-type classification from layout features.

## Files

| File | Description |
|------|-------------|
| `working_example2.py` | Bounding box layout extraction, IoU matrix, reading order algorithm, element area chart |
| `working_example.ipynb` | Interactive notebook version |
| `output/` | `document_understanding.png` |

## Quick Start

```bash
python working_example2.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| LayoutLM | Transformer with 2D positional embeddings for layout |
| Bounding box | (x, y, w, h) spatial location of document elements |
| IoU | Intersection over Union; overlap metric for boxes |
| Reading order | Top-to-bottom, left-to-right reconstruction |
| OCR | Optical Character Recognition; text extraction from images |

## Learning Resources

- Xu et al. *LayoutLM* (2020)
- Kim et al. *Donut* (2022)
- Blecher et al. *Nougat* (2023)
