from pydantic import BaseModel
from pathlib import Path
import os
from typing import Any, Dict
from tqdm import tqdm
from .RAG import Indexer, Searcher, LLM, Evaluator
from .Parser import Parser
from .models import (
    StudentSearchResults, StudentSearchResultsAndAnswer,
    StudentSearchResultsBase
)


class CLI(BaseModel):
    """
    Interface used by the Command Line Interface created with 'fire'.

    Provides commands: index, search, search_dataset, answer, answer_dataset,
    evaluate.
    """
    def index(
            self, directory_path: str = "data/raw/vllm-0.10.1",
            max_chunk_size: int = 2000, text_chunk_overlap: int = 50,
            code_chunk_overlap: int = 50
    ) -> None:
        """
        Ingest the directory content from 'directory_path'.

        Args:
            directory_path (str): Path of the directory.
            max_chunk_size (int): Maximum chunk size.
            text_chunk_overlap (int): Number of overlapping characters for
                the text chunks.
            code_chunk_overlap (int): Number of overlapping characters for
                the code chunks.
        """
        indexer = Indexer(
            directory_path=directory_path,
            max_chunk_size=max_chunk_size,
            text_chunk_overlap=text_chunk_overlap,
            code_chunk_overlap=code_chunk_overlap
        )
        indexer.indexing()
        print("Ingestion complete! Indices saved under data/processed/")

    def search(self, query: str, k: int = 10) -> None:
        """
        Retrieve k query-related chunks.

        Args:
            query (str): Used query
            k (int): Number of chunks to retrieve.
        """
        searcher = Searcher(
            index_path="data/processed/bm25_index/",
            chunks_path="data/processed/chunks/chunks.pkl"
        )
        search_res, docs = searcher.retrieve(
            query_id=None, query=query, n_chunk=k
        )
        srcs = search_res.retrieved_sources
        print(f"Search result(s) for '{query}':")
        for src in srcs:
            print("--------------------")
            print("File path:", src.file_path)
            print("First character index:", src.first_character_index)
            print("Last character index:", src.last_character_index)

    def search_dataset(
            self, dataset_path: str, k: int, save_directory: str
            ) -> None:
        """
        Retrieve k query-related chunk from a dataset.

        Opens the dataset from 'dataset_path' with a 'Parser' model. Loops on
        each query in the dataset and retrieves k relevant chunks, then stores
        them in a 'StudentSearchResults' model and saves it to disk.

        Args:
            dataset_path (str): Path to the dataset containing the queries.
            k (int): Number of chunk to retrieve.
            save_directory (str): Path to the save directory.
        """
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
        for q in tqdm(questions, desc="Processing queries"):
            search_res, docs = searcher.retrieve(
                query_id=q["question_id"], query=q["question"], n_chunk=k
            )
            res.search_results.append(search_res)

        file_name = Path(dataset_path).name
        self.__save_dataset_results(res, save_directory, file_name)
        print(
            f"Saved student_search_results to {save_directory}{file_name}"
        )

    def answer(self, query: str, k: int = 10) -> None:
        """
        Answer to a query using the 'LLM' interface.

        Retrieves k query-related chunks then generates an answer using them
        as context for the LLM.

        Args:
            query (str): Query for the LLM.
            k (int): Number of query-related chunks to retrieve.
        """
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
        """
        Answer to each query in a dataset using the student search results.

        Loads the student search results using the 'Parser' model. Loops on
        each query and their associated search results and generate and answer
        from the 'LLM' interface. Stores the answers to disk along with the
        queries and search results.

        Args:
            student_search_results_path (str): Path to the search results.
            save_directory (str): Path to the save directory.
        """
        parser = Parser(dataset_path=student_search_results_path)
        data = parser.dataset
        search_results = StudentSearchResults.model_validate(data)
        n_question = len(search_results.search_results)
        print(f"Loaded {n_question} questions from "
              f"{student_search_results_path}")

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

        print(f"Processed {len(res.search_results)} of {n_question}")

        file_name = Path(student_search_results_path).name
        self.__save_dataset_results(res, save_directory, file_name)
        print(
            "Saved student_search_results_and_answer to "
            f"{save_directory}{file_name}"
        )

    def evaluate(
            self, student_search_results_path: str, dataset_path: str) -> None:
        """
        Evaluate search results using the Recall@k metric.

        Args:
            student_search_results_path (str): Path to the search results.
            dataset_path (str): Path to the ground truth reference dataset.
        """
        def load_dataset(path: str) -> Dict[str, Any]:
            parser = Parser(dataset_path=path)
            return parser.dataset

        sr = load_dataset(student_search_results_path)
        d = load_dataset(dataset_path)
        evaluator = Evaluator(
            student_search_results=sr,
            dataset=d
        )
        res = evaluator.evaluate()
        print("Evaluation results:")
        print("  Questions Evaluated:", len(evaluator.list_retrieve_srcs))
        for k, v in res.items():
            print(f"  - Recall@{k}: {v:.3f} ({v * 100:.1f}%)")

    def __save_dataset_results(
            self, result: StudentSearchResultsBase,
            save_directory: str, file_name: str
            ) -> None:
        """
        Save a model to a JSON format file.

        Args:
            results (StudentSearchResultsBase): Model to save.
            save_directory (str): Path to the save directory.
            file_name (str): File name of the model
        """
        os.makedirs(save_directory, exist_ok=True)

        try:
            with open(f"{save_directory}/{file_name}", "w") as f:
                f.write(result.model_dump_json(indent=4))
        except Exception as e:
            print("Error while saving 'search_dataset' command results: ", e)
