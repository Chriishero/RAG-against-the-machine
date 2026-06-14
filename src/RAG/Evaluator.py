from pydantic import BaseModel, Field, PrivateAttr, model_validator
from typing import List, Dict, Any
import json
from ..models import MinimalSource


class Evaluator(BaseModel):
    """
    Interface to evaluate the student search results using the Recall@k metric.

    Attributes:
        student_search_results_path (str): path to the student search results.
        dataset_path (str): path to the dataset containing the ground truth.
    """
    model_config = {
        "arbitrary_types_allowed": True
    }
    student_search_results_path: str = Field(min_length=1)
    dataset_path: str = Field(min_length=1)

    _list_retrieve_srcs: List[List[MinimalSource]] = PrivateAttr()
    _list_true_srcs: List[List[MinimalSource]] = PrivateAttr()

    @property
    def list_retrieve_srcs(self) -> List[List[MinimalSource]]:
        """Getter of the private attribute '_list_retrieve_srcs'."""
        return self._list_retrieve_srcs

    @property
    def list_true_srcs(self) -> List[List[MinimalSource]]:
        """Getter of the privat attrbute '_list_true_srcs'."""
        return self._list_true_srcs

    @model_validator(mode="after")
    def validator(self) -> "Evaluator":
        """
        Validate the model after the object creation.

        Load the student search results and the dataset containing the ground
        truth. Then, create from it the private attributes
        '_list_retrieve_srcs' and '_list_true_srcs'. And finally check if they
        are of the same length.
        """
        def load_dataset(path: str) -> Any:
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception as e:
                raise ValueError(
                    f"Cannot open '{path}': {e}"
                )

        search_res = load_dataset(self.student_search_results_path)
        answer_dataset = load_dataset(self.dataset_path)

        self._list_retrieve_srcs = [
            [MinimalSource.model_validate(src)
             for src in res["retrieved_sources"]]
            for res in search_res["search_results"]
        ]
        self._list_true_srcs = [
            [MinimalSource.model_validate(src)
             for src in res["sources"]]
            for res in answer_dataset["rag_questions"]
        ]
        if len(self._list_retrieve_srcs) != len(self._list_true_srcs):
            raise ValueError(
                "The student_search_results and the ground truth dataset"
                "must be of the same length."
            )

        return self

    def evaluate(self) -> Dict[int, float]:
        """
        Evaluate the student search results by calculate the average recall@k.

        Maps each k value to a list of recall scores for each query, then
        returns a mapping from each k to the average recall@k.

        Returns:
            Dict[int, float]: A mapping from k value to the average recall@k.
        """
        recall_res: Dict[int, List[float]] = {
            1: [],
            3: [],
            5: [],
            10: []
        }

        for i in range(0, len(self._list_retrieve_srcs)):
            retrieve_srcs = self._list_retrieve_srcs[i]
            true_srcs = self._list_true_srcs[i]
            for k in recall_res.keys():
                recall = self.__recall(
                    retrieve_srcs,
                    true_srcs,
                    k
                )
                recall_res[k].append(recall)

        return {
            k: sum(v) / len(v) for k, v in recall_res.items()
        }

    def __recall(
            self, retrieve_srcs: List[MinimalSource],
            true_srcs: List[MinimalSource], k: int
            ) -> float:
        """
        Calculate the recall@k between retrieved and ground truth source.

        The recall@k is computed as:
            recall = true_positives / (true_positives + false_negatives)
        where:
            - true_positives: number of ground truth sources found in the first
                'k' retrieved sources.
            - false_negatives: number of ground truth sources not found in the
                first 'k'retrieved sources.

        Args:
            retrieve_srcs (List[MinimalSource]): Sources retrieved for a query.
            true_srcs (List[MinimalSource]): Ground truth source for the same
                query.
            k (int): Number of retrieved sources to consider.

        Returns:
            float: Recall@k value (between 0.0 and 1.0)
        """
        true_positives = 0
        false_negatives = 0

        k_srcs = retrieve_srcs[:k]
        for true_src in true_srcs:
            if self.__validate(true_src, k_srcs):
                true_positives += 1
            else:
                false_negatives += 1

        return true_positives / (true_positives + false_negatives)

    def __validate(
            self, true_src: MinimalSource, k_retrieve_srcs: List[MinimalSource]
            ) -> bool:
        """
        Check if a ground truth source is found among the k retrieve sources.

        Conditions used for checking:
            - at least one of the k retrieve source must have the same file
              path of the ground truth source.
            - one of these source must have an minimum of 5% overlap with the
              ground truth.

        Args:
            true_src (MinimalSource): One of the ground truth sources for a
                query.
            k_retrieve_srcs (List[MinimalSource]): The k first retrieved
                sources for the same query.

        Returns:
            bool: True if at least one source is correct, else False
        """
        if not any(true_src.file_path == s.file_path for s in k_retrieve_srcs):
            return False

        candidate_srcs = [
            src for src in k_retrieve_srcs
            if src.file_path == true_src.file_path
        ]
        for src in candidate_srcs:
            if self.__overlap(src, true_src) >= 0.05:
                return True

        return False

    def __overlap(
            self, a: MinimalSource, b: MinimalSource
            ) -> float:
        """
        Calculate the overlap between two sources.

        The overlap is computed as:
            overlap = intersection_len / total_len
        where:
            intersection_len: length of the part of 'a' intersecting with 'b'.
            total_len: length of 'b'
        The denominator and numerator are computed used the first and last
        character indices of 'a' and 'b'.

        Args:
            a (MinimalSource): Source whose overlapping with 'b' is calculated.
            b (MinimalSource): The reference source.

        Returns:
            float: Overlap value (between 0.0 and 1.0)
        """
        inter_start = max(a.first_character_index, b.first_character_index)
        inter_end = min(a.last_character_index, b.last_character_index)

        if inter_start >= inter_end:
            return 0.0

        inter_len = inter_end - inter_start
        total_len = b.last_character_index - b.first_character_index

        return inter_len / total_len if total_len > 0 else 0.0
