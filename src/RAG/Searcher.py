from pydantic import BaseModel, Field, PrivateAttr, model_validator
from typing import List
from langchain_core.documents import Document
import bm25s
from bm25s.tokenization import Tokenized
import pickle
from ..models import MinimalSource


class Searcher(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True
    }
    index_path: str = Field(min_length=1)
    chunks_path: str = Field(min_length=1)
    _retriever: bm25s.BM25 = PrivateAttr()
    _corpus: List[Document] = PrivateAttr()

    @property
    def retriever(self) -> bm25s.BM25:
        return self._retriever

    @property
    def corpus(self) -> List[Document]:
        return self._corpus

    @model_validator(mode="after")
    def validator(self) -> "Searcher":
        try:
            self._retriever = bm25s.BM25.load(
                self.index_path, load_corpus=False
            )
        except FileNotFoundError as e:
            raise ValueError(
                f"Index not found: {e}.\n"
                "You must run the 'index' command beforehand."
            )

        try:
            with open(self.chunks_path, "rb") as f:
                self._corpus = pickle.load(f)
        except FileNotFoundError as e:
            raise ValueError(
                f"Corpus (chunks) not found: {e}.\n"
                "You must run the 'index' command beforehand."
            )

        return self

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
