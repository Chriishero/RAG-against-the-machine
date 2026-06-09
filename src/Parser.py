from pydantic import BaseModel, Field, PrivateAttr, model_validator
from typing import Dict, Any
import json


class Parser(BaseModel):
    dataset_path: str = Field(min_length=1)
    _dataset_str: str = PrivateAttr(default="")
    _dataset: Dict[str, Any] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def validator(self) -> "Parser":
        try:
            with open(self.dataset_path, "r") as f:
                self._dataset_str = f.read()
        except FileExistsError as e:
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
        return self._dataset
