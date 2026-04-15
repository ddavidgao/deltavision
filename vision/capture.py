"""
Screenshot capture via Playwright.
"""

from PIL import Image
from io import BytesIO


async def capture_screenshot(page) -> Image.Image:
    """Capture current page as PIL Image."""
    png_bytes = await page.screenshot(type="png")
    return Image.open(BytesIO(png_bytes))


def get_current_url(page) -> str:
    """Get current page URL."""
    return page.url
