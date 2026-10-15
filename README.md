# DCT Compression Demo (Python)

This repository contains a simple demo of block-based DCT processing (similar to JPEG): splitting an image into N×N blocks, applying 2D DCT, quantization, selecting coefficients by a diagonal mask, reconstructing via IDCT and comparing results.

Files:
- `dct_pipeline.py` — main script.
- `requirements.txt` — Python dependencies.
- `output/` — generated images are saved here by default (created at runtime).

Quick start:
1. Create a virtual environment and install dependencies:
   python -m venv venv
   source venv/bin/activate  # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
2. Run the script (example using your 17.bmp image):
   python dct_pipeline.py --input 17.bmp

Command-line options are available (see `-h`).
