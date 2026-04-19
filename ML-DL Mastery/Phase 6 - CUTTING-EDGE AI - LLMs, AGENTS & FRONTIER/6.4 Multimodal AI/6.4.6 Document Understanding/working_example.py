"""
Working Example: Document Understanding
Covers OCR, document AI, layout-aware models, and table/chart extraction.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_doc_understand")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. Document understanding overview ---------------------------------------
def doc_overview():
    print("=== Document Understanding ===")
    print()
    print("  Document AI: extract structured information from PDFs, scans, forms, tables")
    print("  Goes beyond OCR to semantic understanding of layout + content")
    print()
    models = [
        ("LayoutLM v3",  "Microsoft; text + image + layout; multi-task"),
        ("Donut",        "OCR-free; end-to-end; encoder-decoder ViT + BART"),
        ("DocFormer",    "Spatial, text, visual tokens; pre-trained"),
        ("Nougat",       "Meta; LaTeX from academic PDFs; equation aware"),
        ("GOT-OCR",      "General OCR Theory; 580M; open; markdown output"),
        ("Qwen-VL",      "Strong OCR + document; Chinese/English"),
        ("Claude 3.5",   "Excellent document QA; multi-page; tables"),
        ("GPT-4o",       "Strong with PDFs; API with vision"),
        ("Marker",       "Pipeline: detect -> OCR -> structure; open source"),
        ("Docling",      "IBM; PDF -> JSON/MD; tables/equations; open"),
    ]
    print(f"  {'Model':<16} {'Notes'}")
    for m, d in models:
        print(f"  {m:<16} {d}")


# -- 2. OCR pipeline -----------------------------------------------------------
def ocr_pipeline():
    print("\n=== OCR Pipeline ===")
    print()
    print("  Traditional OCR:")
    steps = [
        ("1. Pre-processing",  "Binarise, deskew, denoise"),
        ("2. Layout analysis", "Detect text regions, columns, tables"),
        ("3. Line/word detect","Segment text into lines and words"),
        ("4. Recognition",     "CNN + RNN/CTC -> character sequences"),
        ("5. Post-processing", "Spell check, language model correction"),
    ]
    for s, d in steps:
        print(f"  {s:<22} {d}")
    print()
    print("  OCR engines:")
    engines = [
        ("Tesseract",    "Open-source; 100+ languages; LSTM-based"),
        ("PaddleOCR",    "Baidu; fast; multi-lingual; rotated text"),
        ("EasyOCR",      "Python; 80+ langs; easy to use"),
        ("TrOCR",        "Microsoft; transformer OCR; handwriting"),
        ("GOT-OCR",      "End-to-end; markdown; formulas; tables"),
    ]
    for e, d in engines:
        print(f"  {e:<16} {d}")


# -- 3. Layout analysis --------------------------------------------------------
def layout_analysis():
    print("\n=== Layout Analysis ===")
    print()
    print("  Goal: segment document into semantic regions")
    print()
    region_types = ["Text", "Title", "List", "Figure", "Figure Caption",
                    "Table", "Table Caption", "Header", "Footer", "Reference"]
    print("  Region types:")
    for i, r in enumerate(region_types):
        print(f"  [{i}] {r}")
    print()
    print("  Layout models:")
    models = [
        ("Detectron2",      "Meta; region proposal; trained on DocLayNet"),
        ("LayoutParser",    "Unified toolkit; multiple backend models"),
        ("YOLO-Doc",        "Fast; real-time document segmentation"),
        ("DocBank",         "Token classification with layout features"),
    ]
    for m, d in models:
        print(f"  {m:<18} {d}")

    print()
    print("  Simulated bounding boxes for a PDF page:")
    rng = np.random.default_rng(0)
    regions = [
        ("Title",   [50,  20, 760, 60]),
        ("Text",    [50,  80, 760, 200]),
        ("Figure",  [50, 220, 380, 420]),
        ("Table",   [400, 220, 760, 420]),
        ("Footer",  [50, 750, 760, 780]),
    ]
    print(f"  {'Region':<10} {'x1':>4} {'y1':>4} {'x2':>4} {'y2':>4}")
    for r, (x1, y1, x2, y2) in regions:
        print(f"  {r:<10} {x1:>4} {y1:>4} {x2:>4} {y2:>4}")


# -- 4. Table extraction -------------------------------------------------------
def table_extraction():
    print("\n=== Table Extraction ===")
    print()
    print("  Hardest part of document understanding")
    print("  Steps: detect -> structure recognition -> cell content extraction")
    print()

    # Simulate a simple detected table
    headers = ["Quarter", "Revenue ($M)", "Growth (%)", "EBITDA ($M)"]
    rows = [
        ["Q1 2024", "1,240", "+12.3", "310"],
        ["Q2 2024", "1,380", "+14.7", "345"],
        ["Q3 2024", "1,520", "+10.1", "400"],
        ["Q4 2024", "1,650", "+8.6",  "412"],
    ]
    print("  Extracted table:")
    col_w = [10, 14, 12, 14]
    header_str = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_w))
    print(f"  {header_str}")
    print(f"  {'-'*len(header_str)}")
    for row in rows:
        row_str = " | ".join(f"{c:<{w}}" for c, w in zip(row, col_w))
        print(f"  {row_str}")
    print()
    print("  Table extraction approaches:")
    approaches = [
        ("Rule-based",      "Line detection -> cell inference; fragile for complex tables"),
        ("TableFormer",     "Microsoft; structure recognition + functional analysis"),
        ("TATR",            "Table Transformer; deformable DETR on PubTables-1M"),
        ("Img2Table",       "OpenCV + rules; no ML needed; fast"),
        ("Camelot/pdfplumber","PDF primitives; only works on digital PDFs"),
    ]
    for a, d in approaches:
        print(f"  {a:<18} {d}")


# -- 5. Document QA ------------------------------------------------------------
def document_qa():
    print("\n=== Document Question Answering ===")
    print()
    print("  End-to-end pipeline:")
    steps = [
        "1. Ingest PDF: extract text + layout (Docling/Marker/Nougat)",
        "2. Chunk preserving structure (section-aware chunking)",
        "3. Embed chunks -> vector DB (with page/section metadata)",
        "4. At query time: retrieve relevant chunks",
        "5. Pass chunks + question to VLM (include page images for figures)",
        "6. LLM generates cited answer",
    ]
    for s in steps:
        print(f"  {s}")
    print()
    print("  Benchmarks:")
    benchmarks = [
        ("DocVQA",     "Single-page document QA; 50K questions"),
        ("InfographicVQA","Infographic understanding"),
        ("MP-DocVQA",  "Multi-page document reasoning"),
        ("DUDE",       "Diverse document types; open-ended"),
        ("SCROLL",     "Scientific papers; long-document reasoning"),
    ]
    for b, d in benchmarks:
        print(f"  {b:<22} {d}")


if __name__ == "__main__":
    doc_overview()
    ocr_pipeline()
    layout_analysis()
    table_extraction()
    document_qa()
