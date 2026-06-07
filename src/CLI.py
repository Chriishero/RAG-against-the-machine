from pydantic import BaseModel
from .RAG import Indexer, Searcher
import bm25s
import pickle


class CLI(BaseModel):
    def index(self, max_chunk_size: int = 2000) -> None:
        indexer = Indexer(max_chunk_size=max_chunk_size)
        indexer.indexing()
        print("Ingestion complete! Indices saved under data/processed/")

    def search(self, query: str, k: int) -> None:
        try:
            retriever = bm25s.BM25.load(
                "data/processed/bm25_index/", load_corpus=False
            )
        except FileNotFoundError as e:
            print(f"Index not found: {e}.\n"
                  "You must run the 'index' command beforehand.")
            return

        try:
            with open("data/processed/chunks/chunks.pkl", "rb") as f:
                corpus = pickle.load(f)
        except FileNotFoundError as e:
            print(f"Corpus (chunks) not found: {e}.\n"
                  "You must run the 'index' command beforehand.")
            return

        searcher = Searcher(retriever=retriever, corpus=corpus)
        srcs = searcher.retrieve(query=query, n_chunk=k)
        for src in srcs:
            print("--------------------")
            print("File path:", src.file_path)
            print("First character index:", src.first_character_index)
            print("Last character index:", src.last_character_index)

    def search_dataset(
            self, dataset_path: str, k: int, save_directory: str
            ) -> None:
        print(f"Search dataset: {dataset_path} for k={k} in {save_directory}")

    def answer(self, query: str, k: int) -> None:
        print(f"Answer: {query} for k={k}")

    def answer_dataset(
            self, student_search_results_path: str, save_directory: str
            ) -> None:
        print(f"Answer dataset: {student_search_results_path} \
              in {save_directory}")
