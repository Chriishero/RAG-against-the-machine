from pydantic import BaseModel, Field
from typing import List
from langchain_core.documents import Document
import bm25s
from bm25s.tokenization import Tokenized
from ..models import MinimalSource


class Searcher(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True
    }
    retriever: bm25s.BM25
    corpus: List[Document] = Field(min_length=1)

    def retrieve(self, query: str, n_chunk: int) -> List[MinimalSource]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        query_tokens = self._tokenize(query)
        indices, scores = self.retriever.retrieve(
            query_tokens=query_tokens, k=n_chunk
        )
        if len(indices) == 0:
            raise IndexError("No documents found for the given query.")

        docs = [self.corpus[i] for i in indices[0]]
        sources: List[MinimalSource] = []
        for doc in docs:
            file_path, first_idx, last_idx = (
                doc.metadata["file_path"],
                int(doc.metadata["first_character_index"]),
                int(doc.metadata["last_character_index"])
            )
            sources.append(
                MinimalSource(
                    file_path=file_path,
                    first_character_index=first_idx,
                    last_character_index=last_idx
                )
            )
        return sources

    def _tokenize(self, query: str) -> List[List[str]] | Tokenized:
        query_tokens = bm25s.tokenize(query)
        return query_tokens
