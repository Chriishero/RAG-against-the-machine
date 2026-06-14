
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from ollama import Client
from langchain_core.documents import Document
import pickle
from typing import List, Dict, Any
from ..models import MinimalSearchResults, MinimalAnswer, MinimalSource


class LLM(BaseModel):
    """
    Interface to interact with a local Large Language Model.

    Use an Ollama server to interact with any specified LLM (if available),
    and can generate an optimal answer by using a system prompt,
    few-shot examples and a large context from many documents.

    Attributes:
        model_name (str): name of the model.
        host (str): URL of the Ollama server.
        chunks_path (str): path to the chunks pickle file.
    """
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
        """
        Validate the model after the object creation.

        Initialize private attributes:
        - Ollama client for chatting with the model.
        - Chunks loaded from the pickle file (created by the 'index' command).
        - A mapping from chunk metadata to page content.
        """
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
        """
        Generate an answer using the LLM based on the provided search result.

        Get the chunks via '__get_chunks and the message list via
        '__get_messages'. Calls the LLM via '_client.chat()', and returns a
        'MinimalAnswer' enriched with the answer or a default message if the
        context is unrelated to the query.

        Args:
            search_result (MinimalSearchResults): contains the query and the
                retrieved sources.

        Returns:
            MinimalAnswer: A copy of 'search_results' with the 'answer' field
                set.
        """
        res = MinimalAnswer(
                **search_result.model_dump(),
                answer="I cannot find this information "
                       "in the provided documents."
        )
        srcs = search_result.retrieved_sources
        chunks = self.__get_chunks(srcs)
        if not chunks:
            return res

        messages = self.__get_messages(
            search_result.question_str,
            chunks
        )

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

    def __get_chunks(self, srcs: List[MinimalSource]) -> List[str]:
        """
        Get the query related chunks from a list of source.

        Loops over each source and creates a key using their attributes, then
        use it in a mapping between sources attributes and chunks to retrieve
        the corresponding chunks.

        Args:
            srcs (List[MinimalSource]): Sources contained in the corpus.

        Returns:
            List[str]: Query-related chunks.
        """
        chunks: List[str] = []

        for src in srcs:
            key = (
                src.file_path,
                src.first_character_index,
                src.last_character_index
            )
            content = self._chunks_map[key]
            if content:
                chunks.append(content)

        return chunks

    def __get_messages(
            self, query: str, chunks: List[str]
    ) -> List[Dict[str, str]]:
        """
        Build the message list for the LLM.

        Constructs a prompt containing a system prompt, few-shot examples,
        a context built from the provided chunks, and the user query.

        Args:
            query (str): User query.
            chunks (List[str]): Context chunks retrieved from the knowledge
                base.

        Returns:
            List[Dict[str, str]]: Complete message list for the LLM
        """
        messages: List[Dict[str, str]] = []
        system_prompt = (
            "Answer only based on the provided documents. "
            "Be concise and precise. Ignore off-topic or "
            "contradictory documents. "
            "If the query is unrelated to the provided documents, say exactly:"
            " 'I cannot find this information in the provided documents.' "
            "No markdown, no code blocks. Plain text only."
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
        user_prompt = context + "\n" + "Query:\n" + query
        messages.append({"role": "user", "content": user_prompt})

        return messages
