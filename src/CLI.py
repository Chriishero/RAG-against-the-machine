from pydantic import BaseModel
from pathlib import Path
import os
import json
from tqdm import tqdm
from .RAG import Indexer, Searcher, LLM
from .Parser import Parser
from .models import (
    StudentSearchResults, StudentSearchResultsAndAnswer,
    StudentSearchResultsBase
)


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
        search_res, docs = searcher.retrieve(
            query_id=None, query=query, n_chunk=k
        )
        srcs = search_res.retrieved_sources
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
        res = StudentSearchResults(
            k=k,
            search_results=[]
        )
        for q in questions:
            search_res, docs = searcher.retrieve(
                query_id=q["question_id"], query=q["question"], n_chunk=k
            )
            res.search_results.append(search_res)

        file_name = Path(dataset_path).name
        self.__save_dataset_results(res, save_directory, file_name)
        print(
            f"Saved student_search_results to {save_directory}{file_name}"
        )

    def answer(self, query: str, k: int) -> None:
        searcher = Searcher(
            index_path="data/processed/bm25_index/",
            chunks_path="data/processed/chunks/chunks.pkl"
        )
        search_res, docs = searcher.retrieve(None, query, k)
        llm = LLM(
            model_name="qwen3:0.6b",
            host="http://localhost:11434",
            chunks_path=searcher.chunks_path
        )
        answer = llm.answer(search_res)
        print(answer.answer)

    def answer_dataset(
            self, student_search_results_path: str, save_directory: str
            ) -> None:
        try:
            with open(student_search_results_path, "r") as f:
                data = json.load(f)
                search_results = StudentSearchResults.model_validate(data)
        except Exception as e:
            raise ValueError(
                "Failed to load the student search results at "
                f"'{student_search_results_path}': {e}"
            )
        llm = LLM(
            model_name="qwen3:0.6b",
            host="http://localhost:11434",
            chunks_path="data/processed/chunks/chunks.pkl"
        )
        res = StudentSearchResultsAndAnswer(
            k=search_results.k,
            search_results=[]
        )
        for i in tqdm(
            range(0, len(search_results.search_results)),
            desc="Processing dataset answer"
        ):
            answer = llm.answer(search_results.search_results[i])
            res.search_results.append(answer)

        file_name = Path(student_search_results_path).name
        self.__save_dataset_results(res, save_directory, file_name)
        print(
            "Saved student_search_results_and_answer to "
            f"{save_directory}{file_name}"
        )

    def __save_dataset_results(
            self, result: StudentSearchResultsBase,
            save_directory: str, file_name: str
            ) -> None:
        os.makedirs(save_directory, exist_ok=True)

        try:
            with open(f"{save_directory}/{file_name}", "w") as f:
                f.write(result.model_dump_json(indent=4))
        except Exception as e:
            print("Error while saving 'search_dataset' command results: ", e)
