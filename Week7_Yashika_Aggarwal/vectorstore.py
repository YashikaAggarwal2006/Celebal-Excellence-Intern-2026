import os
import logging
from typing import List, Tuple

from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:

    def __init__(
        self,
        openai_api_key: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):

        self.embeddings = OpenAIEmbeddings(
            api_key=openai_api_key,
            model="text-embedding-3-small",
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.vectorstore = None

    def extract_text_from_pdf(self, pdf_path):

        text = ""

        reader = PdfReader(pdf_path)

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        return text

    def extract_text_from_file(self, file_path):

        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def create_documents(self, text, source):

        docs = [
            Document(
                page_content=text,
                metadata={"source": source},
            )
        ]

        return self.text_splitter.split_documents(docs)

    def process_and_index_pdf(self, pdf_path):

        text = self.extract_text_from_pdf(pdf_path)

        docs = self.create_documents(
            text,
            os.path.basename(pdf_path),
        )

        self.index_documents(docs)

        return True

    def process_and_index_text(self, file_path):

        text = self.extract_text_from_file(file_path)

        docs = self.create_documents(
            text,
            os.path.basename(file_path),
        )

        self.index_documents(docs)

        return True

    def index_documents(self, documents):

        if self.vectorstore is None:

            self.vectorstore = FAISS.from_documents(
                documents,
                self.embeddings,
            )

        else:

            self.vectorstore.add_documents(documents)

    def retrieve(
        self,
        query,
        top_k=4,
    ) -> List[Tuple[str, float]]:

        if self.vectorstore is None:
            return []

        docs = self.vectorstore.similarity_search_with_score(
            query,
            k=top_k,
        )

        return [
            (doc.page_content, score)
            for doc, score in docs
        ]

    def get_relevant_context(
        self,
        query,
        top_k=4,
    ):

        docs = self.retrieve(query, top_k)

        if not docs:
            return ""

        return "\n\n".join(
            [doc for doc, _ in docs]
        )

    def is_ready(self):

        return self.vectorstore is not None