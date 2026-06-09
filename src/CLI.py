from pydantic import BaseModel
from typing import Dict, Any
from pathlib import Path
import os
import json
from .RAG import Indexer, Searcher
from .Parser import Parser


class CLI(BaseModel):
    def index(self, max_chunk_size: int = 2000) -> None:
        indexer = Indexer(max_chunk_size=max_chunk_size)
        indexer.indexing()
        print("Ingestion complete! Indices saved under data/processed/")

    def search(self, query: str, k: int) -> None:
        searcher = Searcher(
            index_path="data/processed/bm25_index/",
            chunks_path="data/processed/chunks/chunks.pkl"
        )
        srcs = searcher.retrieve(query=query, n_chunk=k)
        for src in srcs:
            print("--------------------")
            print("File path:", src.file_path)
            print("First character index:", src.first_character_index)
            print("Last character index:", src.last_character_index)

    def search_dataset(
            self, dataset_path: str, k: int, save_directory: str
            ) -> None:
        searcher = Searcher(
            index_path="data/processed/bm25_index/",
            chunks_path="data/processed/chunks/chunks.pkl"
        )
        parser = Parser(dataset_path=dataset_path)
        dataset = parser.dataset
        questions = dataset["rag_questions"]
        res: Dict[str, Any] = {"search_results": [], "k": k}
        for q in questions:
            search: Dict[str, Any] = {
                "question_id": q["question_id"], "retrieved_sources": []
            }
            srcs = searcher.retrieve(query=q["question"], n_chunk=k)
            for src in srcs:
                s_dict = src.model_dump()
                search["retrieved_sources"].append(s_dict)
            res["search_results"].append(search)
        file_name = Path(dataset_path).name
        self.__save_search_result(res, save_directory, file_name)

    def answer(self, query: str, k: int) -> None:
        print(f"Answer: {query} for k={k}")

    def answer_dataset(
            self, student_search_results_path: str, save_directory: str
            ) -> None:
        print(f"Answer dataset: {student_search_results_path} \
              in {save_directory}")

    def __save_search_result(
            self, result: Dict[str, Any], save_directory: str, file_name: str
            ) -> None:
        os.makedirs(save_directory, exist_ok=True)

        try:
            with open(f"{save_directory}/{file_name}", "w") as f:
                json.dump(result, f, indent=4)
        except Exception as e:
            print("Error while saving 'search_dataset' command results: ", e)
