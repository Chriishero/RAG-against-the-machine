from pydantic import BaseModel, Field, PrivateAttr, model_validator
from typing import List, Tuple, Optional
from langchain_core.documents import Document
import bm25s
from bm25s.tokenization import Tokenized
from typing import Union
import pickle
from ..models import MinimalSource, MinimalSearchResults, UnansweredQuestion


class Searcher(BaseModel):
    """
    Search query-related chunks using BM25 retrieval.

    Attributs:
        index_path (str): path of the corpus indices.
        chunks_path (str): path of the corpus.
    """
    model_config = {
        "arbitrary_types_allowed": True
    }
    index_path: str = Field(min_length=1)
    chunks_path: str = Field(min_length=1)
    _retriever: bm25s.BM25 = PrivateAttr()
    _corpus: List[Document] = PrivateAttr()

    @property
    def retriever(self) -> bm25s.BM25:
        """Getter of private attribute '_retriever'."""
        return self._retriever

    @property
    def corpus(self) -> List[Document]:
        """Getter of private attribute '_corpus'."""
        return self._corpus

    @model_validator(mode="after")
    def validator(self) -> "Searcher":
        """
        Validate the model after object creation.

        Load a BM25 retriever from 'index_path' into '_retriever', then load
        the corpus from 'chunks_path' into '_corpus'.
        """
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

    def retrieve(
            self, query_id: Optional[str], query: str, n_chunk: int = 10
            ) -> Tuple[MinimalSearchResults, List[Document]]:
        """
        Retrieve 'n_chunk' related chunks for a query using BM25 retrieval.

        Creates an 'UnansweredQuestion' model containing a query and its ID
        (provided or generated). Retrieves the indices of the relevant chunks
        and stores the corresponding chunks in 'docs'. Each chunk is converted
        into a 'MinimalSource' model. Returns a 'MinimalSearchResults' object
        containing the question and the sources, along with the retrieved
        chunks.

        Args:
            query_id (Optional[str]): ID of the query. Generated if 'None'
            query (str): Query using to retrieve chunks.
            n_chunk (int): Number of chunks to retrieve. Defaults to 10.

        Returns:
            Tuple[MinimalSearchResults, List[Document]]: Search results and
                retrieved chunks.
        """
        if query_id is None:
            question = UnansweredQuestion(question_str=query)
        else:
            question = UnansweredQuestion(
                question_id=query_id, question_str=query
            )
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
        search_results = MinimalSearchResults(
            **question.model_dump(),
            retrieved_sources=sources
        )
        return search_results, docs

    def _tokenize(self, query: str) -> Union[List[List[str]], Tokenized]:
        """
        Tokenize the query for BM25 retrieval.

        Args:
            query (str): Query to tokenize.

        Returns:
            Union[List[List[str]], Tokenized]: Tokenize vector representation
                of the query.
        """
        query_tokens = bm25s.tokenize(query)

        return query_tokens
