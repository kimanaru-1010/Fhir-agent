"""Tiện ích tải ảnh — tải ảnh và mã hóa thành data-URI base64."""

import base64
import io
from pathlib import Path
from PIL import Image


def resolve_image_path(image_path: str) -> str:
    """Giải quyết đường dẫn ảnh: nếu là thư mục, tự chọn tệp ảnh hợp lệ đầu tiên."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp hoặc thư mục tại: {image_path}")

    if path.is_dir():
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        images = [f for f in path.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
        if not images:
            raise FileNotFoundError(f"Thư mục '{image_path}' không chứa tệp ảnh hợp lệ (.jpg, .png, .webp).")
        return str(images[0])

    return str(path)


# Maximum compressed byte size before base64 encoding.
# Ollama/Gemma4 vision tiles cost ~256 tokens per 560x560 patch.
# Keeping the image under 50 KB keeps vision tokens well within a 32K context.
MAX_IMAGE_BYTES = 50_000


def load_image(image_path: str, max_dimension: int = 1024) -> bytes:
    """Tải tệp ảnh, tự động resize + nén để giữ dưới MAX_IMAGE_BYTES."""
    actual_path = resolve_image_path(image_path)

    try:
        with Image.open(actual_path) as img:
            # Convert RGBA/P to RGB for JPEG compatibility
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            # Adaptive compression: try decreasing dimension then quality
            dim_steps = [max_dimension, 768, 640, 512, 384]
            quality_steps = [80, 65, 50, 38]

            for dim in dim_steps:
                resized = img.copy()
                resized.thumbnail((dim, dim), Image.Resampling.LANCZOS)

                for quality in quality_steps:
                    buf = io.BytesIO()
                    resized.save(buf, format="JPEG", quality=quality, optimize=True)
                    data = buf.getvalue()
                    if len(data) <= MAX_IMAGE_BYTES:
                        return data

            # Last resort: smallest possible
            resized = img.copy()
            resized.thumbnail((256, 256), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            resized.save(buf, format="JPEG", quality=35, optimize=True)
            return buf.getvalue()

    except Exception:
        # Fallback: read raw file bytes (may still be large)
        with open(actual_path, "rb") as f:
            return f.read()


def image_to_base64_uri(image_path: str) -> str:
    """Tải ảnh và trả về chuỗi data-URI base64.

    Tự động resize + nén về dưới MAX_IMAGE_BYTES để tránh lỗi context overflow.
    """
    actual_path = resolve_image_path(image_path)
    content = load_image(actual_path)
    # load_image always outputs JPEG; use image/jpeg unconditionally
    b64 = base64.b64encode(content).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

