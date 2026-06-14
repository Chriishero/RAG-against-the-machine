from pydantic import BaseModel, Field, PrivateAttr, model_validator
from typing import Dict, Any
import json


class Parser(BaseModel):
    """
    Check if a file is in a valid JSON format and store its content.

    Attributes:
        dataset_path (str): Path of the dataset.
    """
    dataset_path: str = Field(min_length=1)
    _dataset_str: str = PrivateAttr(default="")
    _dataset: Dict[str, Any] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def validator(self) -> "Parser":
        """
        Validate the model after the object creation.

        Open the dataset from 'dataset_path' and parses it as JSON, storing the
        result as a dictionnary in '_dataset'.
        """
        try:
            with open(self.dataset_path, "r") as f:
                self._dataset_str = f.read()
        except FileNotFoundError as e:
            raise ValueError(f"{self.dataset_path} does not exist: {e}")

        try:
            self._dataset = json.loads(self._dataset_str)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"{self.dataset_path} is not a valid json file: {e}"
            )
        return self

    @property
    def dataset(self) -> Dict[str, Any]:
        """Getter of the private attribute '_dataset'."""
        return self._dataset
