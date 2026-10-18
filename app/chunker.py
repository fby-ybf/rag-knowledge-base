"""文本分块：段落优先，超长段落按定长滑窗切分并保留重叠。"""
from __future__ import annotations

CHUNK_SIZE = 300      # 目标块长度（字符）
CHUNK_OVERLAP = 50    # 相邻块重叠字符数


def split_paragraphs(text: str) -> list[str]:
    """按空行/换行切段，过滤空白段。"""
    parts = [p.strip() for p in text.replace("\r\n", "\n").split("\n") if p.strip()]
    return parts


def sliding_window(paragraph: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """把超长段落切成带重叠的滑窗块。"""
    if len(paragraph) <= size:
        return [paragraph]
    step = max(1, size - overlap)
    return [paragraph[i : i + size] for i in range(0, len(paragraph), step)]


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """整篇文档 -> 分块列表（保持原文顺序）。"""
    chunks: list[str] = []
    for para in split_paragraphs(text):
        chunks.extend(sliding_window(para, size, overlap))
    return [c for c in (c.strip() for c in chunks) if c]
