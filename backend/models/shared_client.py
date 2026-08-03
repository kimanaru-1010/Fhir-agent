"""Generic LLM client cho hệ thống — dùng chung cho orchestrator và tool agents."""

import time

import requests

from config.settings import DEFAULT_MAX_RETRIES, DEFAULT_MAX_TOKENS
from utils.image_loader import image_to_base64_uri


def call_llm(
    prompt: str,
    base_url: str,
    model: str,
    image_path: str | None = None,
    timeout: int = 300,
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    system_prompt: str | None = None,
) -> str:
    """Gửi yêu cầu text hoặc multi-modal (image + text) đến OpenAI-compatible LLM server.

    Thử lại tối đa max_retries lần khi bị timeout hoặc lỗi kết nối.
    """
    url = f"{base_url.rstrip('/')}/v1/chat/completions"

    content_parts = [{"type": "text", "text": prompt}]
    if image_path:
        image_uri = image_to_base64_uri(image_path)
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": image_uri},
        })

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content_parts})

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                url,
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": 1.0,
                    "seed": 42,
                },
                timeout=timeout,
            )
            if not response.ok:
                error_detail = response.text
                try:
                    err_json = response.json()
                    if "error" in err_json:
                        if isinstance(err_json["error"], dict) and "message" in err_json["error"]:
                            error_detail = err_json["error"]["message"]
                        else:
                            error_detail = str(err_json["error"])
                except Exception:
                    pass
                raise requests.exceptions.HTTPError(
                    f"LLM API Error {response.status_code} ({response.url}): {error_detail}",
                    response=response,
                )
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.ReadTimeout as e:
            if attempt < max_retries:
                wait = (attempt + 1) * 2
                print(f"  LLM: Hết giờ (lần {attempt + 1}/{max_retries + 1}). Thử lại sau {wait}s...")
                time.sleep(wait)
            else:
                raise e
        except requests.exceptions.ConnectionError as e:
            if attempt < max_retries:
                wait = (attempt + 1) * 2
                print(f"  LLM: Lỗi kết nối (lần {attempt + 1}/{max_retries + 1}). Thử lại sau {wait}s...")
                time.sleep(wait)
            else:
                raise e


def embed_text(
    texts: list[str],
    base_url: str,
    model: str,
    timeout: int = 60,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> list[list[float]]:
    """Gửi danh sách văn bản đến OpenAI-compatible embeddings endpoint
    (Ollama hỗ trợ /v1/embeddings với model embedding, ví dụ 'bge-m3').

    Trả về danh sách vector, cùng thứ tự với `texts`.
    """
    clean_base = base_url.rstrip("/")
    if clean_base.endswith("/docs"):
        clean_base = clean_base[:-5].rstrip("/")
    url = f"{clean_base}/v1/embeddings"

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                url,
                json={"model": model, "input": texts},
                timeout=timeout,
            )
            if not response.ok:
                raise requests.exceptions.HTTPError(
                    f"Embedding API Error {response.status_code} ({response.url}): {response.text}",
                    response=response,
                )
            data = response.json()["data"]
            # Preserve input order via the "index" field OpenAI-style responses include
            data.sort(key=lambda d: d.get("index", 0))
            return [d["embedding"] for d in data]
        except requests.exceptions.ReadTimeout as e:
            if attempt < max_retries:
                time.sleep((attempt + 1) * 2)
            else:
                raise e
        except requests.exceptions.ConnectionError as e:
            if attempt < max_retries:
                time.sleep((attempt + 1) * 2)
            else:
                raise e