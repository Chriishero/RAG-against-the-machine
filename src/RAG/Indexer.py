from pydantic import BaseModel, Field, model_validator
from typing import Tuple, List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    DirectoryLoader, TextLoader, PythonLoader
    )
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter, PythonCodeTextSplitter
    )
import bm25s
from bm25s.tokenization import Tokenized
from typing import Union
import pickle
import os


class Indexer(BaseModel):
    """
    Index the contents of a directory for BM25 retrieval.

    Indexing steps:
        1. Loading documents based on their file extensions.
        2. Splitting the loaded content into chunks of fixed size.
        3. Tokenizing the corpus and converting it into vector representations.
    The resulting BM25 index and chunks are saved to disk.

    Attributes:
        directory_path (str): Path to the directory to index.
        max_chunk_size (int): Maximum size of each chunk (in character).
        text_chunk_overlap (Optional[int]): Number of overlapping characters
            between text chunks. If None, set to 15% of 'max_chunk_size'.
        code_chunk_overlap (Optional[int]): Number of overlapping characters
            between Python code chunks. If None, set to 15% of
            'max_chunk_size'.
    """
    model_config = {
        "arbitrary_types_allowed": True
    }
    directory_path: str = Field(default="data/raw/vllm-0.10.1/", min_length=1)
    max_chunk_size: int = Field(ge=1, le=2000)
    text_chunk_overlap: Optional[int] = Field(default=None, ge=0)
    code_chunk_overlap: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validator(self) -> "Indexer":
        """
        Validate the model after the object creation.

        If the chunk overlap is None, it is set to 15% of 'max_chunk_size' and
        converted to integer.
        If the chunk overlap is between 0 and 1 (exclusive), it is interpreted
        as a percentage of 'max_chunk_size' and multipilied accordingly.
        """
        if self.text_chunk_overlap is None:
            self.text_chunk_overlap = int(self.max_chunk_size * 0.15)
        if self.code_chunk_overlap is None:
            self.code_chunk_overlap = int(self.max_chunk_size * 0.15)

        if self.text_chunk_overlap > 0 and self.text_chunk_overlap < 1:
            self.text_chunk_overlap *= self.max_chunk_size
        if self.code_chunk_overlap > 0 and self.code_chunk_overlap < 1:
            self.code_chunk_overlap *= self.max_chunk_size

        return self

    def indexing(self) -> None:
        """
        Index the content of the directory and store it to disk.

        The documents are loaded, then split into chunks. The chunk contents
        are stored in a list 'corpus' and tokenized into a vector
        representation. This vector representation and the corpus are stored to
        disk.
        """
        docs = self._load()
        if not docs:
            raise ValueError("No documents has been retrieved.")

        chunks = self._chunking(docs)
        corpus = [chunk.page_content for chunk in chunks]

        self._create_directory()
        retriever = bm25s.BM25()
        tokenized_corpus = self._tokenize(corpus)
        retriever.index(tokenized_corpus)

        try:
            retriever.save("data/processed/bm25_index/")
        except Exception as e:
            print(f"Cannot save the bm25 indices: {e}")

        try:
            with open("data/processed/chunks/chunks.pkl", "wb") as f:
                pickle.dump(chunks, f)
        except Exception as e:
            print(f"Cannot save the chunks: {e}")

    def _load(self) -> Tuple[List[Document], List[Document]]:
        """
        Load the document based on their file extension.

        The function loops over a list of extensions and loads each file using
        either 'TextLoader' or 'PythonLoader'. The file content are stored in
        two separate lists ('text_docs' and 'python_docs'). A tuple of these
        two lists is returned.

        Returns:
            Tuple[List[Document], List[Document]]: Text and Python files
                content.
        """
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
        """
        Split the file contents into chunks of fixed size.

        Two different splitters are used: one for text and one for code. The
        split documents are stored in a list of 'Document'. Metadata is then
        edite to fit specific models.

        Args:
            docs (Tuple[List[Document]], List[Document]): Text and Python
                documents.

        Returns:
            List[Document]: The created chunks.
        """
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
        chunks = self._edit_chunk_metadata(chunks)
        return chunks

    def _tokenize(
            self, corpus: List[str]
    ) -> Union[List[List[str]], Tokenized]:
        """
        Tokenize the corpus for BM25 retrieval.

        Args:
            corpus (List[str]): Corpus containing the chunk contents.

        Returns:
            Union[List[List[str]], Tokenized]: Tokenize vector representation
                of the corpus.
        """
        tokens = bm25s.tokenize(corpus)

        return tokens

    def _edit_chunk_metadata(self, chunks: List[Document]) -> List[Document]:
        """
        Edit chunk metadata to fit specific models.

        Rename the metadata fied 'source' to 'file_path', and 'start_index' to
        'first_character_index'. Adds a new field 'last_character_index'
        computed as the sum of 'first_character_index' and the length of the
        chunk.

        Args:
            chunks (List[Document]): Chunks whose metadata will be edited.

        Returns:
            List[Document]: Chunks with updated metadata.
        """
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
        """Create the directories to store the BM25 index and the corpus."""
        os.makedirs("data/processed/bm25_index/", exist_ok=True)
        os.makedirs("data/processed/chunks/", exist_ok=True)
