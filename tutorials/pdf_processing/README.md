# PDF Multimodal Extraction Tutorial

This tutorial demonstrates how to build a complete multimodal pretraining data curation pipeline using NeMo Curator with local vLLM inference. The pipeline extracts and analyzes text, tables, and images from PDF documents using vision-language models.

## Overview

The tutorial consists of 4 scripts that process PDFs through a complete data curation workflow:

1. **Download PDFs** - Download PDF files from URLs
2. **Extract Multimodal Content** - Extract text, tables, and images using vision-language models
3. **Remove Duplicates** - Apply fuzzy and semantic deduplication
4. **Quality Filtering** - Filter low-quality content

## Architecture

### Pipeline Stages

The main extraction pipeline (`1_run_data_extraction.py`) implements **10 stages**:

| Stage | Description | Resources | Model |
|-------|-------------|-----------|-------|
| JSONLReaderStage | Read PDF paths from JSONL | 2 CPUs | - |
| PDFToImageStage | Convert PDF pages to images (300 DPI) | 4 CPUs | - |
| LayoutDetectionStage | Detect document layout with bounding boxes | 4 CPUs, 16GB GPU | nvidia/nemoretriever-parse |
| BoundingBoxExtractionStage | Crop regions from images | 4 CPUs | - |
| ContentTypeClassificationStage | Classify content types (text/table/image) | 2 CPUs | - |
| TableExtractionStage | Extract tables to HTML format | 4 CPUs | - |
| TextExtractionStage | Extract text from regions | 4 CPUs | - |
| ImageExtractionStage | Extract and save image regions | 2 CPUs | - |
| DeepAnalysisStage | Deep content analysis with VLM | 4 CPUs, 12GB GPU | llama-3.1-nemotron-nano-vl-8b-v1 |
| JSONLWriterStage | Write organized results to JSONL | 2 CPUs | - |

### Data Flow

```
source/pdf_urls.jsonl
    ↓ (0_download.py)
extraction_results/downloaded_pdfs.jsonl + source/pdfs/
    ↓ (1_run_data_extraction.py)
extraction_results/extracted_multimodal_data.jsonl
    ↓ (2_remove_duplicates.py)
dedup_results/deduplicated_data.jsonl
    ↓ (3_run_quality_filters.py)
quality_results/filtered_data.jsonl (final curated dataset)
```

## Setup

### Requirements

```bash
# Install NeMo Curator with vLLM support
pip install nemo-curator[vllm]

# Install PDF processing dependencies
pip install pdf2image
sudo apt-get install poppler-utils  # Linux
# or
brew install poppler  # macOS

# Optional: For OCR support
pip install pytesseract
```

### GPU Requirements

- **Minimum**: 1x 24GB GPU (e.g., RTX 3090, A5000)
  - Can run both GPU stages sequentially
- **Recommended**: 2x 24GB GPUs
  - Run layout detection and analysis in parallel for higher throughput
- **Optimal**: 4+ GPUs
  - Use tensor parallelism for larger models or process more PDFs concurrently

## Usage

### Step 0: Prepare PDF URLs

Create `source/pdf_urls.jsonl` with PDF URLs to download:

```jsonl
{"url": "https://arxiv.org/pdf/2301.00001.pdf", "filename": "sample1.pdf"}
{"url": "https://arxiv.org/pdf/2302.00001.pdf", "filename": "sample2.pdf"}
```

### Step 1: Download PDFs

```bash
python 0_download.py
```

This downloads PDFs to `source/pdfs/` and creates a manifest at `extraction_results/downloaded_pdfs.jsonl`.

### Step 2: Extract Multimodal Content

```bash
python 1_run_data_extraction.py
```

This runs the complete extraction pipeline:
- Converts PDFs to images
- Detects layout with vision-language model
- Extracts tables, text, and images
- Performs deep content analysis
- Writes organized results to `extraction_results/extracted_multimodal_data.jsonl`

**Expected runtime**: ~2-5 minutes per PDF (depends on GPU, page count, and model size)

### Step 3: Remove Duplicates

```bash
python 2_remove_duplicates.py
```

Applies fuzzy and semantic deduplication to remove near-duplicate and semantically duplicate content.

Output: `dedup_results/deduplicated_data.jsonl`

### Step 4: Apply Quality Filters

```bash
python 3_run_quality_filters.py
```

Filters low-quality content based on text quality metrics and custom thresholds.

Output: `quality_results/filtered_data.jsonl` (final curated dataset)

## Output Format

The final output (`extracted_multimodal_data.jsonl`) contains one JSON object per PDF:

```json
{
  "pdf_path": "/path/to/document.pdf",
  "pages": [
    {
      "page_number": 0,
      "text": [
        {
          "bbox": {"x": 0.1, "y": 0.2, "width": 0.8, "height": 0.1},
          "text": "Extracted text content..."
        }
      ],
      "tables": [
        {
          "bbox": {"x": 0.1, "y": 0.4, "width": 0.8, "height": 0.3},
          "html": "<table>...</table>"
        }
      ],
      "images": [
        {
          "bbox": {"x": 0.2, "y": 0.7, "width": 0.6, "height": 0.2},
          "type": "figure",
          "image_base64": "iVBORw0KGgoAAAANS..."
        }
      ],
      "analyses": [
        {
          "type": "table",
          "bbox": {"x": 0.1, "y": 0.4, "width": 0.8, "height": 0.3},
          "analysis": "This table shows experimental results..."
        }
      ]
    }
  ]
}
```

## Configuration

### Model Selection

You can change the models used for layout detection and deep analysis:

```python
# In 1_run_data_extraction.py

# Layout detection model
LayoutDetectionStage(
    model_identifier="nvidia/nemoretriever-parse",  # Change here
    ...
)

# Deep analysis model
DeepAnalysisStage(
    model_identifier="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",  # Change here
    ...
)
```

### GPU Memory Configuration

Adjust GPU memory allocation per stage:

```python
# In stage definitions
self.resources = Resources(
    cpus=4.0,
    gpus=1.0,
    gpu_mem_gb=16.0  # Adjust based on available GPU memory
)
```

### Processing Parameters

Customize processing parameters:

```python
# PDF to Image conversion
PDFToImageStage(
    dpi=300,  # Increase for higher quality (slower)
)

# Layout detection
LayoutDetectionStage(
    max_tokens=3500,  # Increase for longer outputs
    temperature=0.0,  # Adjust for more/less randomness
)

# Deep analysis
DeepAnalysisStage(
    max_tokens=1024,
    temperature=0.2,
    top_p=0.7,
)
```

## Performance Optimization

### Scaling Strategies

1. **Single GPU**: Process PDFs sequentially
   - Layout detection and analysis run on the same GPU
   - Throughput: ~2-5 PDFs/hour (depends on page count)

2. **Multi-GPU**: Parallel processing
   - Distribute PDFs across GPUs
   - Throughput scales linearly with GPU count

3. **Ray Distributed Execution**:
   - Automatic GPU scheduling
   - Worker pooling for efficient GPU utilization
   - Stage-level parallelism

### Batch Size Strategy

All stages use `batch_size = 1` (process one PDF at a time) because:
- Ray handles parallelism automatically
- Workers process multiple PDFs concurrently based on available resources
- Better failure isolation (one PDF failure doesn't affect others)

## Troubleshooting

### Out of Memory (OOM) Errors

If you encounter GPU OOM errors:

1. **Reduce GPU memory allocation**:
   ```python
   # In stage definitions
   self.resources = Resources(gpu_mem_gb=12.0)  # Reduce from 16GB
   ```

2. **Process fewer PDFs concurrently**:
   - Ray will automatically limit concurrency based on available GPU memory

3. **Use smaller models**:
   - Switch to smaller vision-language models if available

### PDF Conversion Issues

If PDF to image conversion fails:

1. **Install poppler**:
   ```bash
   # Linux
   sudo apt-get install poppler-utils

   # macOS
   brew install poppler
   ```

2. **Check PDF file integrity**:
   - Ensure PDFs are not corrupted
   - Try opening PDFs in a viewer first

### vLLM Installation Issues

If vLLM fails to load:

1. **Check CUDA version compatibility**:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Reinstall vLLM**:
   ```bash
   pip uninstall vllm
   pip install vllm --no-cache-dir
   ```

## Advanced Usage

### Custom Stages

You can add custom processing stages to the pipeline:

```python
from nemo_curator.stages.base import ProcessingStage

class CustomStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    def __init__(self):
        self.name = "custom_stage"
        self.resources = Resources(cpus=2.0)

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        # Your custom processing logic here
        return batch

# Add to pipeline
pipeline.add_stage(CustomStage())
```

### Filtering by Content Type

You can filter stages to process only specific content types:

```python
# Only analyze tables and images
DeepAnalysisStage(
    analyze_types=["table", "image"],  # Exclude text
)
```

### Enable OCR Fallback

For PDFs with images of text:

```python
TextExtractionStage(
    use_ocr=True,  # Enable OCR for text extraction
)
```

## References

- [NeMo Curator Documentation](https://github.com/NVIDIA/NeMo-Curator)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Nemotron Parse Model](https://huggingface.co/nvidia/nemoretriever-parse)
- [Llama 3.1 Nemotron Nano VL](https://huggingface.co/nvidia/llama-3.1-nemotron-nano-vl-8b-v1)

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{nemo_curator,
  title = {NeMo Curator: Scalable Data Curation for Large Language Models},
  author = {NVIDIA Corporation},
  year = {2025},
  url = {https://github.com/NVIDIA/NeMo-Curator}
}
```
