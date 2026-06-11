from pydantic import BaseModel, Field, PrivateAttr, model_validator
from ollama import Client
from langchain_core.documents import Document
import pickle
from typing import List, Dict, Any
from ..models import MinimalSearchResults, MinimalAnswer, MinimalSource


class LLM(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True
    }
    model_name: str = Field(min_length=1, default="qwen3:0.6b")
    host: str = Field(min_length=1, default="http://localhost:11434")
    chunks_path: str = Field(min_length=1)
    _client: Client = PrivateAttr()
    _corpus: list[Document] = PrivateAttr()
    _chunks_map: Dict[Any, str] = PrivateAttr()

    @model_validator(mode="after")
    def validator(self) -> "LLM":
        try:
            self._client = Client(host=self.host, headers={})
        except Exception as e:
            raise ValueError(
                f"Failed to create the ollama client: {e}"
            )
        try:
            with open(self.chunks_path, "rb") as f:
                self._corpus = pickle.load(f)
        except Exception as e:
            raise ValueError(
                f"Failed to open the corpus file '{self.chunks_path}': {e}"
            )
        self._chunks_map = {}
        for doc in self._corpus:
            key = (
                doc.metadata.get("file_path"),
                doc.metadata.get("first_character_index"),
                doc.metadata.get("last_character_index")
            )
            self._chunks_map[key] = doc.page_content
        return self

    def answer(
            self, search_result: MinimalSearchResults) -> MinimalAnswer:
        res = MinimalAnswer(
                **search_result.model_dump(),
                answer="I cannot find this information "
                       "in the provided documents."
        )
        chunks: List[str] = []
        srcs = search_result.retrieved_sources
        for src in srcs:
            key = (
                src.file_path,
                src.first_character_index,
                src.last_character_index
            )
            content = self._chunks_map[key]
            if content:
                chunks.append(content)
        if not chunks:
            return res

        messages: List[Dict[str, str]] = []
        system_prompt = (
            "Answer only based on the provided documents. "
            "Be concise and precise. Ignore off-topic or "
            "contradictory documents. "
            "If the query is unrelated to the provided documents, say exactly:"
            " 'I cannot find this information in the provided documents.' "
            "No markdown, no code bocks. Plain text only."
        )
        messages.append({"role": "system", "content": system_prompt})

        examples = [
            (
                "Where is the Eiffel Tower?",
                "The Eiffel Tower is located in Paris."
            ),
            (
                "What is vLLM?",
                "vLLM is a fast and easy-to-use library for LLM inference and"
                "serving."
            )
        ]
        for q, a in examples:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

        context = "\n\n".join(
            f"Context:\n[Document {i}]\n{chunk}"
            for i, chunk in enumerate(chunks)
        )
        user_prompt = context + "\n" + "Query:\n" + search_result.question_str
        messages.append({"role": "user", "content": user_prompt})

        try:
            response = self._client.chat(
                model=self.model_name,
                messages=messages,
                stream=False,
                options={
                    "temperature": 0.0,
                    "num_gpu": -1,
                    "enable_thinking": False
                }
            )
        except Exception as e:
            raise ValueError(
                f"Failed to get answer from {self.model_name}: {e}"
            )
        if response.message.content is None:
            raise ValueError(
                f"Failed to get answer from {self.model_name}."
            )
        res.answer = response.message.content
        return res
