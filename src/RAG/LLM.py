from pydantic import BaseModel, Field, PrivateAttr, model_validator
from ollama import Client
from langchain_core.documents import Document
import pickle
from typing import List, Dict, Optional
from ..models import MinimalSearchResults, MinimalAnswer


class LLM(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True
    }
    model_name: str = Field(min_length=1, default="qwen3:0.6b")
    host: str = Field(min_length=1, default="http://localhost:11434")
    chunks_path: str = Field(min_length=1)
    _client: Client = PrivateAttr()
    _corpus: list[Document] = PrivateAttr()

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
        return self

    def answer(
            self, search_result: MinimalSearchResults,
            docs: Optional[list[Document]]) -> MinimalAnswer:
        if docs is None:
            docs = self._corpus
        res = MinimalAnswer(
                **search_result.model_dump(),
                answer="I cannot find this information "
                       "in the provided documents."
        )
        chunks: list[Document] = []
        srcs = search_result.retrieved_sources
        for src in srcs:
            for doc in docs:
                data = src.model_dump()
                if data == doc.metadata:
                    chunks.append(doc)
        if not chunks:
            return res

        messages: List[Dict[str, str]] = []
        system_prompt = (
            "You are a specialized assistant in documentation based answer.\n"
            "You must answer only based on the documents provided "
            "in the context.\n"
            "Your answer must be concise and precise.\n"
            "Do not answer in a general manner.\n"
            "If the query is completely unrelated to the provided context, "
            "or if the context does not contain any relevant information, "
            "you must answer exactly: "
            "'I cannot find this information in the provided documents.'"
            "Do not invent anything.\n"
            "Do not answer with markdown."
            "Answer only in plain text, "
            "no asterisks, no code blocks, no headings."
        )
        messages.append({"role": "system", "content": system_prompt})

        examples = [
            (
                "Context:\nThe Eiffel Tower is located in Paris.\n"
                "Query: Where is the Eiffel Tower?",
                "The Eiffel Tower is located in Paris."
            ),
            (
                "Context:\nvLLM is a fast and easy-to-use library for LLM"
                "inference and serving. Originally developed in the Sky"
                "Computing Lab at UC Berkeley, vLLM has evolved into a"
                "community-driven project with contributions from both"
                "academia and industry.\n"
                "Query: What is vLLM?",
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
                messages=messages
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
