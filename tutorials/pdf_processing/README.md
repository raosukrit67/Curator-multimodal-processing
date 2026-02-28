# Multimodal PDF Extraction and Curation Tutorial

This tutorial demonstrates a complete pipeline for extracting and curating multimodal content from PDF documents using NeMo Curator. It uses two vision-language models via local vLLM inference:

- **Nemotron Parse 1.1B** (`nvidia/NVIDIA-Nemotron-Parse-v1.1`) — OCR, layout detection, and text extraction in a single pass
- **Nemotron Nano 12B VL** (`nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16`) — Visual understanding for pictures, figures, and charts

## Architecture

### Why Two Models?

Nemotron Parse is a specialized encoder-decoder VLM (~885M params) that simultaneously detects document layout (13 content classes with bounding boxes) and extracts text as Markdown. However, for visual content (pictures, figures, charts), Parse can only detect the bounding box — it cannot describe or interpret visual content. Nemotron Nano VL (12B params) fills this gap by generating natural language descriptions of cropped image regions.

### Extraction Pipeline Stages

The extraction pipeline (`2_run_extraction.py`) has 7 stages:

| # | Stage | Description | Resources |
|---|-------|-------------|-----------|
| 1 | PDFReaderStage | Read PDF paths from JSONL manifest | 1 CPU |
| 2 | PDFToImageStage | Render PDF pages as 300 DPI images | 1 CPU |
| 3 | LayoutDetectionStage | Nemotron Parse: OCR + layout + text extraction | 1 CPU, 1 GPU (16GB) |
| 4 | ContentRoutingStage | Route text regions direct, visual regions to VL | 1 CPU |
| 5 | VisualAnalysisStage | Nemotron Nano VL: describe visual content | 1 CPU, 1 GPU (24GB) |
| 6 | TextAssemblyStage | Combine all modalities into structured JSON | 0.5 CPU |
| 7 | PDFWriterStage | Write results to JSONL | 1 CPU |

### Data Flow

```
PDF Page Image (300 DPI)
     │
     ▼
[Nemotron Parse 1.1B]  ──► Text/Title/Section/etc. ──────────────► ┐
     │                  ──► Tables (LaTeX) ───────────────────────► ├─► TextAssemblyStage ──► JSONL
     │                  ──► Picture/Figure/Chart (bbox only) ──► ┐ │
     │                                                          │ │
     ▼                                                          ▼ │
[Crop image region]  ──►  [Nemotron Nano 12B VL]  ──► description ─┘
```

### Content Routing

| Parse Class | Route | Processing |
|-------------|-------|------------|
| Text, Title, Section, List-Item, Caption, etc. | Direct | Already extracted as Markdown |
| Table | Direct (optionally VL) | LaTeX extracted by Parse; VL can add interpretation |
| Formula | Direct | LaTeX extracted by Parse |
| Picture, Figure, Chart | VL model | Cropped and sent to Nano VL for description |

## Tutorial Scripts

```
tutorials/pdf_processing/
├── datasets.json            # PDF source configs (URLs for download)
├── 0_download.py            # Download PDFs from URLs (optional, skip if you have local PDFs)
├── 1_prepare_data.py        # Create pdf_files.jsonl from a PDF directory
├── 2_run_extraction.py      # Main 7-stage extraction pipeline (GPU)
├── 3_remove_duplicates.py   # Fuzzy deduplication (MinHash + LSH)
├── 4_run_quality_filters.py # Q&A quality filtering
├── visualize_layout.py      # Visualize Parse layout detection with colored bboxes
├── visualize_extraction.py  # Side-by-side and heatmap visualization of extraction
└── data/raw/pdfs/           # Sample NVIDIA datasheets (5 PDFs included)
```

### Two Paths to Extraction

```
Path A (have URLs):                     Path B (have local PDFs):

  datasets.json                           /your/pdf/directory/
       │                                         │
       ▼                                         │
  0_download.py                                  │
       │                                         │
       ▼                                         ▼
  data/raw/pdfs/  ──────────────►  1_prepare_data.py
                                         │
                                         ▼
                                   pdf_files.jsonl
                                         │
                                         ▼
                                   2_run_extraction.py  ──►  extracted_data.jsonl
                                         │
                                         ▼
                                   3_remove_duplicates.py  ──►  deduplicated_data.jsonl
                                         │
                                         ▼
                                   4_run_quality_filters.py  ──►  filtered_data.jsonl
```

## Setup

### Requirements

```bash
# Install NeMo Curator with vLLM support
pip install nemo-curator[vllm]

# PDF rendering (PyMuPDF)
pip install pymupdf
```

### GPU Requirements

| Setup | Description |
|-------|-------------|
| **Minimum** | 1x GPU with 24GB VRAM (run Parse and VL sequentially) |
| **Recommended** | 2x GPUs (Parse on one, VL on another in parallel) |
| **Optimal** | 4+ GPUs (tensor parallelism + concurrent processing) |

GPU memory estimates:
- Nemotron Parse 1.1B: ~4 GB (BF16)
- Nemotron Nano 12B VL: ~24 GB (BF16) or ~12 GB (FP8)

## Usage

### Step 0: Download PDFs (optional)

Skip this step if you already have PDF files locally.

PDF URL sources are configured in `datasets.json`. Five NVIDIA hardware datasheets are included as sample data.

```bash
# List available datasets
python 0_download.py --list

# Download the default dataset
python 0_download.py --dataset NVIDIA_DATASHEETS

# Download to a custom directory
python 0_download.py --output-dir /data/my_pdfs

# Test with fewer files
python 0_download.py --max-files 2
```

To add a custom PDF source, edit `datasets.json`:
```json
{
  "MY_PAPERS": {
    "description": "My research papers",
    "source": "url",
    "urls": ["https://example.com/paper1.pdf", "https://example.com/paper2.pdf"]
  }
}
```

### Step 1: Prepare Data

Create a JSONL manifest from your PDF directory. Both paths (downloaded or local) converge here.

```bash
# From downloaded PDFs (default directory)
python 1_prepare_data.py

# From your own PDF directory
python 1_prepare_data.py --pdf-dir /path/to/your/pdfs
```

### Step 2: Extract Multimodal Content

```bash
python 2_run_extraction.py --input data/raw/pdf_files.jsonl --output data/extracted/extracted_data.jsonl
```

Options:
- `--parse-model` — Override Parse model (default: `nvidia/NVIDIA-Nemotron-Parse-v1.1`)
- `--vl-model` — Override VL model (default: `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16`)
- `--dpi` — PDF rendering resolution (default: 300)
- `--include-tables-for-vl` — Also send tables to VL model for richer interpretation
- `--model-cache-dir` — Custom cache directory for model weights

### Step 3: Deduplicate

```bash
python 3_remove_duplicates.py
```

### Step 4: Quality Filter for Q&A Curation

Filters are designed for building Q&A datasets from PDF content:

```bash
python 4_run_quality_filters.py --input data/extracted/extracted_data.jsonl --output data/filtered/filtered_data.jsonl
```

The pipeline applies three filters:

| Filter | What it does | Why |
|--------|-------------|-----|
| **ExtractionCompletenessFilter** | Drop docs where >50% of pages have no content | Parse failed — corrupt/scanned/encrypted PDF |
| **BoilerplateFilter** | Strip pages that are only headers/footers/ToC | Cannot produce Q&A pairs |
| **QAReadinessFilter** | Keep docs with answerable content (text >= 80 chars, tables, or described figures) | Short fragments and empty pages can't form Q&A pairs |

Parameters: `--min-extraction-ratio`, `--min-answer-length`, `--min-qa-pages`

## Output Format

The extraction pipeline produces JSONL with one JSON object per PDF:

```json
{
  "pdf_path": "/path/to/document.pdf",
  "pages": [
    {
      "page_number": 0,
      "text_blocks": [
        {"class_name": "Title", "bbox": [100, 50, 500, 90], "text": "Introduction"}
      ],
      "tables": [
        {"bbox": [50, 200, 550, 400], "latex": "\\begin{tabular}...\\end{tabular}"}
      ],
      "figures": [
        {"bbox": [100, 450, 400, 700], "class_name": "Picture", "description": "Bar chart showing..."}
      ],
      "full_text": "Introduction\n\nThe extracted body text...\n\n[Picture: Bar chart showing...]"
    }
  ]
}
```

Bounding boxes are `[left, top, right, bottom]` in pixel coordinates (at the rendered DPI).

## Visualization

```bash
# Color-coded layout detection overlay
python visualize_layout.py

# Side-by-side extraction view (numbered regions + extracted text)
python visualize_extraction.py --mode side_by_side

# Heatmap (green=text ok, red=empty, yellow=VL description)
python visualize_extraction.py --mode heatmap
```

## References

- [NeMo Curator](https://github.com/NVIDIA/NeMo-Curator)
- [Nemotron Parse 1.1B](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1)
- [Nemotron Nano 12B v2 VL](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16)
- [vLLM](https://docs.vllm.ai/)
