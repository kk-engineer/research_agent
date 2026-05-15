from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Protocol

from src.models import ClaimChunk, ContradictionRecord


class ClaimStore(Protocol):
    def add_claims(self, claims: list[ClaimChunk]) -> None: ...
    def get_all_claims(self) -> list[ClaimChunk]: ...
    def get_claims_by_sub_question(self, sub_question_id: str) -> list[ClaimChunk]: ...
    def deduplicate(self) -> int: ...
    def get_unique_sources(self) -> list[str]: ...
    def clear(self) -> None: ...


class InMemoryClaimStore:
    def __init__(self) -> None:
        self._claims: Dict[str, ClaimChunk] = {}
        self._by_sub_question: Dict[str, list[str]] = defaultdict(list)

    def add_claims(self, claims: list[ClaimChunk]) -> None:
        for claim in claims:
            if claim.claim_id not in self._claims:
                self._claims[claim.claim_id] = claim
                self._by_sub_question[claim.sub_question_id].append(claim.claim_id)

    def get_all_claims(self) -> list[ClaimChunk]:
        return list(self._claims.values())

    def get_claims_by_sub_question(self, sub_question_id: str) -> list[ClaimChunk]:
        ids = self._by_sub_question.get(sub_question_id, [])
        return [self._claims[cid] for cid in ids if cid in self._claims]

    def deduplicate(self) -> int:
        seen_texts: set[str] = set()
        before = len(self._claims)
        to_remove: list[str] = []
        for cid, claim in self._claims.items():
            normalized = claim.text.strip().lower()
            if normalized in seen_texts:
                to_remove.append(cid)
            else:
                seen_texts.add(normalized)
        for cid in to_remove:
            del self._claims[cid]
            for sq_id, ids in self._by_sub_question.items():
                if cid in ids:
                    ids.remove(cid)
        return before - len(self._claims)

    def get_unique_sources(self) -> list[str]:
        return sorted({c.source_url for c in self._claims.values()})

    def clear(self) -> None:
        self._claims.clear()
        self._by_sub_question.clear()
