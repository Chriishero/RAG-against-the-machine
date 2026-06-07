from pydantic import BaseModel, Field
from typing import Tuple, List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    DirectoryLoader, TextLoader, PythonLoader
    )
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter, PythonCodeTextSplitter
    )
import bm25s
from bm25s.tokenization import Tokenized
import pickle
import os


class Indexer(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True
    }
    directory_path: str = Field(default="data/raw/vllm-0.10.1/", min_length=1)
    max_chunk_size: int = Field(ge=1, le=2000)
    text_chunk_overlap: int = Field(default=50, ge=0)
    code_chunk_overlap: int = Field(default=50, ge=0)

    def indexing(self) -> None:
        docs = self._load()
        chunks = self._chunking(docs)
        corpus = [chunk.page_content for chunk in chunks]
        print(corpus[0])

        self._create_directory()
        retriever = bm25s.BM25()
        tokenized_corpus = self._tokenize(corpus)
        retriever.index(tokenized_corpus)
        retriever.save("data/processed/bm25_index/")
        with open("data/processed/chunks/chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)

    def _load(self) -> Tuple[List[Document], List[Document]]:
        extensions = [
            ".py", ".md", ".txt", ".rst"
            ]
        text_docs: List[Document] = []
        python_docs: List[Document] = []
        for ext in extensions:
            if ext in [".md", ".txt", ".rst"]:
                loader = DirectoryLoader(
                    self.directory_path,
                    glob=f"**/*{ext}",
                    loader_cls=TextLoader
                )
                text_docs.extend(loader.load())
            else:
                loader = DirectoryLoader(
                    self.directory_path,
                    glob=f"**/*{ext}",
                    loader_cls=PythonLoader
                )
                python_docs.extend(loader.load())
        return (text_docs, python_docs)

    def _chunking(
            self, docs: Tuple[List[Document], List[Document]]
            ) -> List[Document]:
        chunks: List[Document] = []
        text_splitter = RecursiveCharacterTextSplitter(
            separators=[".", ",", ";", ")", "}", "]", "\n", "\n\n"],
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.text_chunk_overlap,
            add_start_index=True,
        )
        code_splitter = PythonCodeTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.code_chunk_overlap,
            add_start_index=True,
        )
        chunks += text_splitter.split_documents(docs[0])
        chunks += code_splitter.split_documents(docs[1])
        chunks = self._add_chunk_metadata(chunks)
        return chunks

    def _tokenize(self, corpus: List[str]) -> List[List[str]] | Tokenized:
        query_tokens = bm25s.tokenize(corpus)
        return query_tokens

    def _add_chunk_metadata(self, chunks: List[Document]) -> List[Document]:
        for i in range(0, len(chunks)):
            chunks[i].metadata[
                "file_path"] = chunks[i].metadata["source"]
            chunks[i].metadata[
                "first_character_index"] = chunks[i].metadata["start_index"]
            last_character_index = (
                chunks[i].metadata["start_index"] + len(chunks[i].page_content)
            )
            chunks[i].metadata["last_character_index"] = last_character_index

            chunks[i].metadata.pop("source")
            chunks[i].metadata.pop("start_index")
        return chunks

    def _create_directory(self) -> None:
        os.makedirs("data/processed/bm25_index/", exist_ok=True)
        os.makedirs("data/processed/chunks/", exist_ok=True)
