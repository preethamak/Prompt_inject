"""
Layered defense helpers for indirect prompt injection:

1. Chunk-level detection over external context
2. Sanitization by removing high-risk chunks
3. Lightweight task-alignment risk gating
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import numpy as np


INSTRUCTION_PATTERNS = [
    r"\bignore\b.{0,40}\b(instruction|previous|above|system)\b",
    r"\bfollow\b.{0,40}\b(instruction|steps|directions)\b",
    r"\breveal\b.{0,40}\b(prompt|system|secret|policy)\b",
    r"\bdo not\b.{0,40}\buser\b",
    r"\bact as\b",
    r"\bdeveloper message\b",
    r"\bsystem prompt\b",
    r"\boverride\b",
    r"\bbypass\b",
    r"\btool\b.{0,40}\bcall\b",
    r"\bexecute\b.{0,40}\bcommand\b",
]

QUESTION_RE = re.compile(r"^(what|which|who|when|where|why|how|did|does|is|are|can)\b", re.I)
INSTRUCTION_RE = re.compile(
    r"^(summarize|list|extract|write|tell|give|show|describe|explain|ignore|return|provide)\b",
    re.I,
)
INSTRUCTION_RX = re.compile("|".join(INSTRUCTION_PATTERNS), re.I | re.S)


@dataclass
class ChunkAssessment:
    idx: int
    text: str
    start: int
    end: int
    risk_prob: float
    flagged: bool


@dataclass
class DefenseAssessment:
    user_intent: str
    sanitized_context: str
    original_text: str
    sanitized_text: str
    removed_chunks: int
    total_chunks: int
    removed_ratio: float
    instruction_hits: int
    alignment_risk: float
    gate_label: str
    chunks: list[ChunkAssessment]


def build_model_text(context: str, user_intent: str, context_max_chars: int) -> str:
    return f"Context: {context[:context_max_chars]}\nUser intent: {user_intent}"


def split_context_into_chunks(
    context: str,
    chunk_chars: int = 320,
    overlap_chars: int = 64,
) -> list[tuple[int, int, str]]:
    """
    Split context into partially overlapping chunks.
    Prefer row/paragraph boundaries when possible to preserve local attack spans.
    """
    if not context:
        return []

    boundaries = ["\n\n", "\n", "|"]
    chunks: list[tuple[int, int, str]] = []
    start = 0
    n = len(context)

    while start < n:
        target_end = min(n, start + chunk_chars)
        end = target_end
        for sep in boundaries:
            idx = context.rfind(sep, start, target_end)
            if idx > start + max(32, chunk_chars // 3):
                end = idx + len(sep)
                break
        if end <= start:
            end = target_end

        chunk = context[start:end].strip()
        if chunk:
            chunks.append((start, end, chunk))

        if end >= n:
            break
        start = max(end - overlap_chars, start + 1)

    return chunks


def assess_task_alignment(
    user_intent: str,
    context: str,
    removed_chunks: int,
    total_chunks: int,
) -> tuple[float, str, int]:
    """
    Heuristic alignment gate for practical defense layering.
    """
    instruction_hits = len(INSTRUCTION_RX.findall(context))
    intent = user_intent.strip()
    is_question = ("?" in intent) or bool(QUESTION_RE.match(intent))
    is_instruction = bool(INSTRUCTION_RE.match(intent))
    removed_ratio = removed_chunks / max(total_chunks, 1)

    risk = 0.0
    risk += min(0.5, 0.12 * instruction_hits)
    risk += min(0.35, 0.50 * removed_ratio)
    if is_question and instruction_hits > 0:
        risk += 0.15
    if is_instruction and instruction_hits > 1:
        risk += 0.10
    risk = min(1.0, risk)

    if risk >= 0.75:
        label = "block"
    elif risk >= 0.35:
        label = "review"
    else:
        label = "allow"
    return risk, label, instruction_hits


class LayeredDefensePipeline:
    """
    Scores context chunks with the trained fused detector, removes high-risk chunks,
    then computes a lightweight task-alignment gate.
    """

    def __init__(
        self,
        tfidf_branch,
        xgb_branch,
        fusion_model,
        context_max_chars: int,
        chunk_threshold: float,
        chunk_chars: int = 320,
        overlap_chars: int = 64,
        replacement_text: str = "[UNTRUSTED INSTRUCTION REMOVED]",
    ):
        self.tfidf_branch = tfidf_branch
        self.xgb_branch = xgb_branch
        self.fusion_model = fusion_model
        self.context_max_chars = context_max_chars
        self.chunk_threshold = chunk_threshold
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars
        self.replacement_text = replacement_text

    def _score_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([], dtype=float)
        prob_a = self.tfidf_branch.predict_proba(texts)
        prob_b = self.xgb_branch.predict_proba(texts)
        meta = []
        for text, a, b in zip(texts, prob_a, prob_b):
            instruction_hits = len(INSTRUCTION_RX.findall(text))
            pipe_count = text.count("|")
            newline_count = text.count("\n")
            meta.append([
                float(a),
                float(b),
                float(abs(a - b)),
                float(a),
                float(a),
                float(b),
                float(b),
                float(np.log1p(len(text))),
                0.0,
                0.0,
                float(np.log1p(instruction_hits)),
                float(np.log1p(pipe_count)),
                float(np.log1p(newline_count)),
            ])
        return self.fusion_model.predict_proba(np.asarray(meta, dtype=float))

    def _score_chunks(self, chunks: list[tuple[int, int, str]], user_intent: str) -> np.ndarray:
        texts = [
            build_model_text(chunk_text, user_intent, self.context_max_chars)
            for _, _, chunk_text in chunks
        ]
        return self._score_texts(texts)

    def _build_assessment(
        self,
        context: str,
        user_intent: str,
        chunks: list[tuple[int, int, str]],
        chunk_probs: np.ndarray,
    ) -> DefenseAssessment:
        assessments: list[ChunkAssessment] = []
        sanitized_parts: list[str] = []
        last_end = 0
        removed_chunks = 0

        for idx, ((start, end, chunk_text), prob) in enumerate(zip(chunks, chunk_probs)):
            flagged = bool(prob >= self.chunk_threshold)
            assessments.append(ChunkAssessment(
                idx=idx,
                text=chunk_text,
                start=start,
                end=end,
                risk_prob=float(prob),
                flagged=flagged,
            ))
            sanitized_parts.append(context[last_end:start])
            if flagged:
                sanitized_parts.append(self.replacement_text)
                removed_chunks += 1
            else:
                sanitized_parts.append(context[start:end])
            last_end = end

        sanitized_parts.append(context[last_end:])
        sanitized_context = "".join(sanitized_parts) if chunks else context
        sanitized_text = build_model_text(sanitized_context, user_intent, self.context_max_chars)
        original_text = build_model_text(context, user_intent, self.context_max_chars)

        risk, gate_label, instruction_hits = assess_task_alignment(
            user_intent=user_intent,
            context=context,
            removed_chunks=removed_chunks,
            total_chunks=len(chunks),
        )

        return DefenseAssessment(
            user_intent=user_intent,
            sanitized_context=sanitized_context,
            original_text=original_text,
            sanitized_text=sanitized_text,
            removed_chunks=removed_chunks,
            total_chunks=len(chunks),
            removed_ratio=(removed_chunks / len(chunks)) if chunks else 0.0,
            instruction_hits=instruction_hits,
            alignment_risk=risk,
            gate_label=gate_label,
            chunks=assessments,
        )

    def analyze_row(self, row: dict) -> DefenseAssessment:
        context = str(row.get("context", ""))
        user_intent = str(row.get("user_intent", ""))
        chunks = split_context_into_chunks(
            context,
            chunk_chars=self.chunk_chars,
            overlap_chars=self.overlap_chars,
        )
        chunk_probs = self._score_chunks(chunks, user_intent)
        return self._build_assessment(context, user_intent, chunks, chunk_probs)

    def analyze_rows(self, rows: Iterable[dict]) -> list[DefenseAssessment]:
        rows_list = list(rows)
        chunk_sets: list[list[tuple[int, int, str]]] = []
        contexts: list[str] = []
        intents: list[str] = []
        flat_texts: list[str] = []

        for row in rows_list:
            context = str(row.get("context", ""))
            user_intent = str(row.get("user_intent", ""))
            chunks = split_context_into_chunks(
                context,
                chunk_chars=self.chunk_chars,
                overlap_chars=self.overlap_chars,
            )
            contexts.append(context)
            intents.append(user_intent)
            chunk_sets.append(chunks)
            flat_texts.extend([
                build_model_text(chunk_text, user_intent, self.context_max_chars)
                for _, _, chunk_text in chunks
            ])

        flat_probs = self._score_texts(flat_texts)
        assessments: list[DefenseAssessment] = []
        cursor = 0
        for context, user_intent, chunks in zip(contexts, intents, chunk_sets):
            n_chunks = len(chunks)
            chunk_probs = flat_probs[cursor:cursor + n_chunks]
            cursor += n_chunks
            assessments.append(self._build_assessment(
                context=context,
                user_intent=user_intent,
                chunks=chunks,
                chunk_probs=chunk_probs,
            ))
        return assessments


def summarize_defense_assessments(
    assessments: list[DefenseAssessment],
    prefix: str = "",
) -> dict:
    p = f"{prefix}_" if prefix else ""
    if not assessments:
        return {
            f"{p}avg_removed_chunks": 0.0,
            f"{p}avg_removed_ratio": 0.0,
            f"{p}avg_alignment_risk": 0.0,
            f"{p}blocked_fraction": 0.0,
            f"{p}review_fraction": 0.0,
            f"{p}avg_instruction_hits": 0.0,
        }

    removed_chunks = np.array([a.removed_chunks for a in assessments], dtype=float)
    removed_ratio = np.array([a.removed_ratio for a in assessments], dtype=float)
    alignment_risk = np.array([a.alignment_risk for a in assessments], dtype=float)
    instruction_hits = np.array([a.instruction_hits for a in assessments], dtype=float)
    gate_labels = [a.gate_label for a in assessments]

    return {
        f"{p}avg_removed_chunks": round(float(removed_chunks.mean()), 6),
        f"{p}avg_removed_ratio": round(float(removed_ratio.mean()), 6),
        f"{p}avg_alignment_risk": round(float(alignment_risk.mean()), 6),
        f"{p}blocked_fraction": round(float(np.mean([g == "block" for g in gate_labels])), 6),
        f"{p}review_fraction": round(float(np.mean([g == "review" for g in gate_labels])), 6),
        f"{p}avg_instruction_hits": round(float(instruction_hits.mean()), 6),
    }
