"""
test_pdf_crop.py – unit tests for pdf_crop.py

Run with:
    pytest test_pdf_crop.py -v
"""

import io
import pytest
import numpy as np
from PIL import Image
import fitz  # PyMuPDF

from pdf_crop import (
    _gray,
    crop_whitespace,
    filter_blank_rows,
    process_page_image,
    render_page_to_image,
    image_to_pdf_page,
    process_pdf,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic test images
# ---------------------------------------------------------------------------

def white_image(w: int = 100, h: int = 100) -> Image.Image:
    """Return a fully white RGB image."""
    return Image.fromarray(np.full((h, w, 3), 255, dtype=np.uint8))


def black_rect_image(
    w: int = 100,
    h: int = 100,
    rect_x0: int = 20,
    rect_y0: int = 20,
    rect_x1: int = 80,
    rect_y1: int = 80,
) -> Image.Image:
    """Return a white image with a black rectangle drawn in the centre."""
    arr = np.full((h, w, 3), 255, dtype=np.uint8)
    arr[rect_y0:rect_y1, rect_x0:rect_x1] = 0
    return Image.fromarray(arr)


def image_with_blank_rows(
    content_rows: int = 10,
    blank_rows: int = 5,
    width: int = 50,
) -> Image.Image:
    """
    Build an image that alternates between black content rows and white
    blank rows:

        [blank_rows white rows]
        [content_rows black rows]
        [blank_rows white rows]
        [content_rows black rows]
        [blank_rows white rows]
    """
    white_block = np.full((blank_rows, width, 3), 255, dtype=np.uint8)
    black_block = np.full((content_rows, width, 3), 0, dtype=np.uint8)
    arr = np.vstack([white_block, black_block, white_block, black_block, white_block])
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# _gray
# ---------------------------------------------------------------------------

class TestGray:
    def test_pure_white_gives_255(self):
        img = Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8))
        g = _gray(img)
        np.testing.assert_allclose(g, 255.0)

    def test_pure_black_gives_0(self):
        img = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
        g = _gray(img)
        np.testing.assert_allclose(g, 0.0)

    def test_shape_matches_input(self):
        img = Image.fromarray(np.zeros((7, 13, 3), dtype=np.uint8))
        g = _gray(img)
        assert g.shape == (7, 13)


# ---------------------------------------------------------------------------
# crop_whitespace
# ---------------------------------------------------------------------------

class TestCropWhitespace:
    def test_blank_page_returns_1x1_placeholder(self):
        img = white_image(100, 100)
        result = crop_whitespace(img)
        assert result.size == (1, 1), (
            "Blank page must return a 1×1 placeholder, not a full image."
        )

    def test_content_trimmed_to_tight_bounding_box(self):
        # Black rectangle from (20,20) to (80,80) on a 100×100 white image.
        img = black_rect_image(100, 100, 20, 20, 80, 80)
        result = crop_whitespace(img, white_threshold=245)
        # The result must be the 60×60 black block, not the full image.
        assert result.size == (60, 60), (
            f"Expected 60×60 crop, got {result.size}."
        )

    def test_full_content_image_unchanged_size(self):
        # Image entirely filled with black – no whitespace to trim.
        arr = np.zeros((50, 50, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        result = crop_whitespace(img, white_threshold=245)
        assert result.size == (50, 50)

    def test_single_dark_pixel_top_left(self):
        arr = np.full((20, 20, 3), 255, dtype=np.uint8)
        arr[0, 0] = [0, 0, 0]
        img = Image.fromarray(arr)
        result = crop_whitespace(img, white_threshold=245)
        # Tight bbox of a single (0,0) black pixel → 1×1
        assert result.size == (1, 1)

    def test_result_is_never_zero_size(self):
        img = white_image(200, 300)
        result = crop_whitespace(img)
        w, h = result.size
        assert w > 0 and h > 0, "Result must never be zero-sized."


# ---------------------------------------------------------------------------
# filter_blank_rows
# ---------------------------------------------------------------------------

class TestFilterBlankRows:
    def test_blank_page_returns_placeholder_not_full_image(self):
        img = white_image(60, 60)
        result = filter_blank_rows(img, depth_threshold=20)
        # Must NOT return the full 60-row image.
        assert result.height <= 1, (
            "An all-blank image must yield a 1-row placeholder."
        )

    def test_blank_rows_are_removed(self):
        # alternating: 5 white / 10 black / 5 white / 10 black / 5 white
        img = image_with_blank_rows(content_rows=10, blank_rows=5, width=50)
        original_h = img.height  # 5+10+5+10+5 = 35
        result = filter_blank_rows(img, depth_threshold=20)
        # All white (blank) rows must be gone; only 2×10 = 20 black rows kept.
        assert result.height == 20, (
            f"Expected 20 rows after removing blanks, got {result.height}."
        )
        assert result.height < original_h

    def test_no_blank_rows_image_unchanged(self):
        arr = np.zeros((30, 40, 3), dtype=np.uint8)  # all black
        img = Image.fromarray(arr)
        result = filter_blank_rows(img, depth_threshold=20)
        assert result.height == 30

    def test_result_is_never_zero_size(self):
        img = white_image(100, 100)
        result = filter_blank_rows(img)
        w, h = result.size
        assert w > 0 and h > 0


# ---------------------------------------------------------------------------
# process_page_image (combined pipeline)
# ---------------------------------------------------------------------------

class TestProcessPageImage:
    def test_blank_page_produces_tiny_placeholder(self):
        img = white_image(200, 200)
        result = process_page_image(img)
        w, h = result.size
        assert w > 0 and h > 0
        assert w <= 1 and h <= 1, (
            "Blank page must yield a tiny (≤1×1) placeholder."
        )

    def test_content_surrounded_by_whitespace_is_cropped(self):
        # 100×100 image with a 10×10 black block in the centre.
        arr = np.full((100, 100, 3), 255, dtype=np.uint8)
        arr[45:55, 45:55] = 0
        img = Image.fromarray(arr)
        result = process_page_image(img, white_threshold=245, depth_threshold=20)
        # Width and height must be much smaller than the original 100×100.
        assert result.width <= 15
        assert result.height <= 15

    def test_result_size_never_zero(self):
        for _ in range(3):
            img = white_image(50, 50)
            result = process_page_image(img)
            w, h = result.size
            assert w > 0 and h > 0

    def test_intermediate_blank_rows_removed(self):
        img = image_with_blank_rows(content_rows=8, blank_rows=4, width=40)
        result = process_page_image(img, depth_threshold=20)
        # After removing leading/trailing whitespace AND blank rows,
        # only 2×8 = 16 rows of content remain.
        assert result.height == 16, (
            f"Expected 16 content rows, got {result.height}."
        )


# ---------------------------------------------------------------------------
# PDF round-trip: bookmarks preserved
# ---------------------------------------------------------------------------

def _make_test_pdf_with_toc(n_pages: int = 3) -> bytes:
    """Create an in-memory PDF with *n_pages* pages and a simple ToC."""
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page(width=200, height=200)
        # Draw a black rectangle so pages are not blank.
        page.draw_rect(fitz.Rect(40, 40, 160, 160), color=(0, 0, 0), fill=(0, 0, 0))

    toc = [[1, f"Chapter {i+1}", i + 1] for i in range(n_pages)]
    doc.set_toc(toc)

    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


class TestBookmarkPreservation:
    def test_bookmarks_copied_to_output(self, tmp_path):
        pdf_bytes = _make_test_pdf_with_toc(n_pages=3)
        input_path = tmp_path / "input.pdf"
        output_path = tmp_path / "output.pdf"
        input_path.write_bytes(pdf_bytes)

        process_pdf(str(input_path), str(output_path))

        out_doc = fitz.open(str(output_path))
        toc = out_doc.get_toc()
        out_doc.close()

        assert len(toc) == 3, f"Expected 3 bookmark entries, got {len(toc)}."
        titles = [entry[1] for entry in toc]
        assert titles == ["Chapter 1", "Chapter 2", "Chapter 3"]

    def test_page_count_preserved(self, tmp_path):
        pdf_bytes = _make_test_pdf_with_toc(n_pages=4)
        input_path = tmp_path / "input.pdf"
        output_path = tmp_path / "output.pdf"
        input_path.write_bytes(pdf_bytes)

        process_pdf(str(input_path), str(output_path))

        src = fitz.open(str(input_path))
        dst = fitz.open(str(output_path))
        assert len(dst) == len(src), (
            "Output must have the same page count as the input."
        )
        src.close()
        dst.close()

    def test_no_bookmarks_pdf_works(self, tmp_path):
        """process_pdf must not crash when the source has no bookmarks."""
        doc = fitz.open()
        page = doc.new_page(width=200, height=200)
        page.draw_rect(fitz.Rect(40, 40, 160, 160), color=(0, 0, 0), fill=(0, 0, 0))
        buf = io.BytesIO()
        doc.save(buf)
        doc.close()

        input_path = tmp_path / "input.pdf"
        output_path = tmp_path / "output.pdf"
        input_path.write_bytes(buf.getvalue())

        process_pdf(str(input_path), str(output_path))

        out_doc = fitz.open(str(output_path))
        assert len(out_doc) == 1
        out_doc.close()


# ---------------------------------------------------------------------------
# Crop effectiveness – regression test for the double-pass bug
# ---------------------------------------------------------------------------

class TestCropEffectiveness:
    """
    Regression tests ensuring that the crop is actually effective (i.e., the
    original double-pass bug in the conversation prototype does not regress).
    """

    def test_crop_removes_large_white_margins(self):
        # 200×200 image, black 10×10 block in the bottom-right corner.
        arr = np.full((200, 200, 3), 255, dtype=np.uint8)
        arr[180:190, 180:190] = 0
        img = Image.fromarray(arr)
        result = process_page_image(img)
        # The result should be much smaller than 200×200.
        assert result.width < 50, (
            f"crop_whitespace should have trimmed wide margins; got width={result.width}"
        )
        assert result.height < 50, (
            f"crop_whitespace should have trimmed tall margins; got height={result.height}"
        )

    def test_filter_blank_rows_is_not_a_noop(self):
        """
        After crop_whitespace the image starts from row 0, so if
        filter_blank_rows uses range-crop only it would be a no-op.
        This test verifies that intermediate blank rows ARE removed.
        """
        # Build: 5 black rows, 10 white rows, 5 black rows → height 20
        black = np.zeros((5, 40, 3), dtype=np.uint8)
        white = np.full((10, 40, 3), 255, dtype=np.uint8)
        arr = np.vstack([black, white, black])
        img = Image.fromarray(arr)

        result = filter_blank_rows(img, depth_threshold=20)
        # Must be 10 (5+5) rows, not 20 (the full range).
        assert result.height == 10, (
            f"Expected 10 rows (blanks removed), got {result.height}. "
            "The filter is a no-op – the double-pass bug may have regressed."
        )
