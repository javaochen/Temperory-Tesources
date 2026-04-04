"""
pdf_crop.py – image-based PDF crop pipeline

Features
--------
* Renders every PDF page to an RGB image (no get_text()).
* Crops surrounding whitespace (margins) via a grayscale threshold.
* Removes blank rows (line-spacing / empty lines) via a colour-depth
  threshold.
* When a page is entirely blank, keeps a 1×1 white placeholder so the
  output page count never changes and pages are never zero-sized.
* Copies the source bookmarks (table of contents) to the output unchanged
  (page count is preserved 1-to-1, so page numbers stay correct).
* Saves the output with JPEG-compressed images and PDF deflate for a
  compact file size.

Bug fixed vs. the original conversation prototype
--------------------------------------------------
The original ``remove_blank_rows`` implementation did a *two-pass*
approach: it found the kept-row index range [top, bottom], sliced the
image to that range (a no-op after ``crop_whitespace`` had already done
a tight bounding-box crop), then re-computed depths on the already-sliced
image.  The re-indexed depths matched, so intermediate blank rows were
retained instead of being removed, making the row filter effectively a
no-op.  The fix is to use NumPy fancy-indexing (``arr[keep_indices]``)
which directly selects only the non-blank rows in a single pass.
"""

import io
import fitz  # PyMuPDF
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _gray(img: Image.Image) -> np.ndarray:
    """Return a float32 (H, W) grayscale array for *img* (RGB input).

    Uses the ITU-R BT.601 luma coefficients (0.299 R + 0.587 G + 0.114 B).
    """
    arr = np.asarray(img, dtype=np.float32)
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


# ---------------------------------------------------------------------------
# Crop steps
# ---------------------------------------------------------------------------

def crop_whitespace(img: Image.Image, white_threshold: int = 245) -> Image.Image:
    """Trim surrounding whitespace from *img*.

    Finds the tight bounding box of all pixels whose grayscale value is
    strictly below *white_threshold* (i.e. non-background pixels) and
    returns the cropped image.

    When the entire page is blank (all pixels ≥ *white_threshold*), a
    1×1 white placeholder is returned so that downstream code never
    receives a zero-size image.
    """
    gray = _gray(img)
    mask = gray < white_threshold

    if not mask.any():
        # Blank page – return a 1×1 white placeholder.
        return Image.fromarray(np.full((1, 1, 3), 255, dtype=np.uint8))

    ys, xs = np.where(mask)
    return img.crop((int(xs.min()), int(ys.min()),
                     int(xs.max()) + 1, int(ys.max()) + 1))


def filter_blank_rows(img: Image.Image, depth_threshold: int = 20) -> Image.Image:
    """Remove rows whose maximum colour depth is below *depth_threshold*.

    Colour depth per pixel is defined as ``255 – grayscale``, so pure
    white pixels have depth 0 and pure black pixels have depth 255.  A
    row is considered blank when every pixel in that row has a depth
    below *depth_threshold*.

    Non-blank rows are collected with NumPy fancy indexing so that
    intermediate blank rows (e.g. empty lines between paragraphs) are
    genuinely removed in a single pass.

    When every row is blank, a 1-pixel-tall centre slice is returned as
    a non-zero-size placeholder.
    """
    gray = _gray(img)
    depth_per_row = (255.0 - gray).max(axis=1)  # shape: (H,)
    keep = np.where(depth_per_row >= depth_threshold)[0]

    if keep.size == 0:
        # All rows blank – keep a single centre row as placeholder.
        mid = max(img.height // 2, 0)
        return img.crop((0, mid, img.width, mid + 1))

    arr = np.asarray(img, dtype=np.uint8)
    kept = arr[keep]          # fancy-index: one pass, no double-crop
    return Image.fromarray(kept)


def process_page_image(
    img: Image.Image,
    white_threshold: int = 245,
    depth_threshold: int = 20,
) -> Image.Image:
    """Run the full single-page crop pipeline on *img*.

    Steps:
    1. :func:`crop_whitespace` – remove surrounding blank margins.
    2. :func:`filter_blank_rows` – remove blank rows within the content.
    """
    img = crop_whitespace(img, white_threshold=white_threshold)
    img = filter_blank_rows(img, depth_threshold=depth_threshold)
    return img


# ---------------------------------------------------------------------------
# PDF I/O helpers
# ---------------------------------------------------------------------------

def render_page_to_image(page: fitz.Page, zoom: float = 1.5) -> Image.Image:
    """Render a PDF *page* as an RGB :class:`PIL.Image.Image`.

    *zoom* controls the render resolution.  The default of 1.5 avoids
    unnecessary upscaling while still producing a legible image for most
    documents.
    """
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def image_to_pdf_page(
    doc: fitz.Document,
    img: Image.Image,
    jpeg_quality: int = 75,
) -> fitz.Page:
    """Insert *img* as a JPEG-compressed image into a new page of *doc*.

    JPEG is used (instead of PNG) to keep the output file size small.
    """
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    img_bytes = buf.getvalue()

    page = doc.new_page(width=img.width, height=img.height)
    rect = fitz.Rect(0, 0, img.width, img.height)
    page.insert_image(rect, stream=img_bytes)
    return page


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_pdf(
    input_pdf: str,
    output_pdf: str,
    zoom: float = 1.5,
    white_threshold: int = 245,
    depth_threshold: int = 20,
    jpeg_quality: int = 75,
) -> None:
    """Convert *input_pdf* to a cropped image-based PDF at *output_pdf*.

    One output page is produced per input page (the page count is
    preserved), so bookmarks copied from the source document point to the
    correct pages without any mapping step.

    Parameters
    ----------
    input_pdf:
        Path to the source PDF.
    output_pdf:
        Path for the resulting PDF.
    zoom:
        Render resolution multiplier (default 1.5 – good quality without
        excessive upscaling).
    white_threshold:
        Grayscale cut-off for whitespace detection (0–255).  Pixels with
        grayscale ≥ this value are treated as background/white.
    depth_threshold:
        Minimum per-row colour depth to keep a row.  Rows where every
        pixel has depth < this value are removed.
    jpeg_quality:
        JPEG quality for the embedded page images (1–95).
    """
    src = fitz.open(input_pdf)
    dst = fitz.open()

    toc = src.get_toc()

    for page_index, page in enumerate(src):
        print(f"Processing page {page_index + 1}/{len(src)} …")
        img = render_page_to_image(page, zoom=zoom)
        processed = process_page_image(
            img,
            white_threshold=white_threshold,
            depth_threshold=depth_threshold,
        )
        image_to_pdf_page(dst, processed, jpeg_quality=jpeg_quality)

    if toc:
        dst.set_toc(toc)

    dst.save(output_pdf, garbage=4, deflate=True)
    dst.close()
    src.close()
    print(f"Done. Saved to: {output_pdf}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Crop PDF pages (image-based pipeline) and preserve bookmarks."
    )
    parser.add_argument("input_pdf", help="Path to the input PDF.")
    parser.add_argument("output_pdf", help="Path for the output PDF.")
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.5,
        help="Render zoom factor (default: 1.5).",
    )
    parser.add_argument(
        "--white-threshold",
        type=int,
        default=245,
        help="Grayscale threshold for background detection (default: 245).",
    )
    parser.add_argument(
        "--depth-threshold",
        type=int,
        default=20,
        help="Minimum per-row colour depth to keep a row (default: 20).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=75,
        help="JPEG quality for embedded images (default: 75).",
    )
    args = parser.parse_args()

    process_pdf(
        input_pdf=args.input_pdf,
        output_pdf=args.output_pdf,
        zoom=args.zoom,
        white_threshold=args.white_threshold,
        depth_threshold=args.depth_threshold,
        jpeg_quality=args.jpeg_quality,
    )
