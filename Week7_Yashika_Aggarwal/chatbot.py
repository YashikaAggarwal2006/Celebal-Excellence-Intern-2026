"""
Chatbot module for handling user interaction and generating
responses using OpenAI's language models.
"""

import logging
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGChatbot:
    """
    Chatbot that uses Retrieval-Augmented Generation to answer questions
    based on retrieved context from documents.
    """
    
    def __init__(
        self,
        openai_api_key: str,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        """
        Initialize the RAG Chatbot.
        
        Args:
            openai_api_key: API key for OpenAI
            model: LLM model to use (gpt-4, gpt-4-turbo, gpt-3.5-turbo)
            temperature: Creativity level (0=deterministic, 1=creative)
            max_tokens: Maximum tokens in response
        """
        self.openai_api_key = openai_api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=openai_api_key
        )
        
        # Initialize the chain with prompt and parser
        self._setup_chain()
        
        logger.info(f"RAGChatbot initialized with model: {model}")
    
    def _setup_chain(self):
        """Set up the prompt template and chain."""
        
        system_prompt = """You are a helpful assistant that answers questions based on provided context.

Instructions:
1. Use ONLY the provided context to answer questions
2. Be accurate and factual
3. If the answer is not in the context, clearly state "I cannot find this information in the provided documents"
4. Provide clear and concise answers
5. When relevant, mention which part of the document the information comes from
6. Be professional and helpful in tone"""
        
        self.prompt_template = ChatPromptTemplate.from_template(
            """System: {system_prompt}

Context from Documents:
{context}

User Question: {question}

Answer:"""
        )
        
        # Create the chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()
    
    def generate_answer(
        self,
        question: str,
        context: str
    ) -> str:
        """
        Generate an answer based on the question and context.
        
        Args:
            question: The user's question
            context: Retrieved context from documents
        
        Returns:
            The generated answer
        """
        try:
            logger.info(f"Generating answer for question: {question}")
            
            answer = self.chain.invoke({
                "system_prompt": self._get_system_prompt(),
                "context": context,
                "question": question
            })
            
            logger.info("Answer generated successfully")
            return answer
        
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"Error generating answer: {str(e)}"
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt."""
        return """You are a helpful assistant that answers questions based on provided context.

Instructions:
1. Use ONLY the provided context to answer questions
2. Be accurate and factual
3. If the answer is not in the context, clearly state "I cannot find this information in the provided documents"
4. Provide clear and concise answers
5. When relevant, mention which part of the document the information comes from
6. Be professional and helpful in tone"""
    
    def generate_answer_with_sources(
        self,
        question: str,
        context: str,
        sources: list = None
    ) -> dict:
        """
        Generate an answer and include source information.
        
        Args:
            question: The user's question
            context: Retrieved context from documents
            sources: List of source document names
        
        Returns:
            Dictionary with answer and source information
        """
        answer = self.generate_answer(question, context)
        
        return {
            "answer": answer,
            "sources": sources or [],
            "question": question
        }
    
    def update_model(self, model: str):
        """
        Update the LLM model.
        
        Args:
            model: New model name
        """
        self.model = model
        self.llm = ChatOpenAI(
            model=model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.openai_api_key
        )
        self._setup_chain()
        logger.info(f"Model updated to: {model}")
    
    def update_temperature(self, temperature: float):
        """
        Update the temperature parameter.
        
        Args:
            temperature: New temperature value (0-1)
        """
        if 0 <= temperature <= 1:
            self.temperature = temperature
            self.llm = ChatOpenAI(
                model=self.model,
                temperature=temperature,
                max_tokens=self.max_tokens,
                api_key=self.openai_api_key
            )
            self._setup_chain()
            logger.info(f"Temperature updated to: {temperature}")
        else:
            logger.warning(f"Temperature must be between 0 and 1, got {temperature}")
    
    def update_max_tokens(self, max_tokens: int):
        """
        Update the max tokens parameter.
        
        Args:
            max_tokens: New max tokens value
        """
        if max_tokens > 0:
            self.max_tokens = max_tokens
            self.llm = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                max_tokens=max_tokens,
                api_key=self.openai_api_key
            )
            self._setup_chain()
            logger.info(f"Max tokens updated to: {max_tokens}")
        else:
            logger.warning(f"Max tokens must be positive, got {max_tokens}")
    
    def get_current_config(self) -> dict:
        """Get current chatbot configuration."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }


class ConversationManager:
    """
    Manages conversation history for multi-turn interactions.
    """
    
    def __init__(self, max_history: int = 10):
        """
        Initialize the conversation manager.
        
        Args:
            max_history: Maximum number of messages to keep in history
        """
        self.max_history = max_history
        self.history = []
    
    def add_exchange(self, question: str, answer: str, context: str = ""):
        """
        Add a question-answer exchange to history.
        
        Args:
            question: User's question
            answer: Generated answer
            context: Retrieved context (optional)
        """
        exchange = {
            "question": question,
            "answer": answer,
            "context": context
        }
        self.history.append(exchange)
        
        # Keep only recent history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        logger.info(f"Added exchange to history. Total: {len(self.history)}")
    
    def get_history(self) -> list:
        """Get conversation history."""
        return self.history.copy()
    
    def get_last_context(self) -> Optional[str]:
        """Get context from the last exchange."""
        if self.history:
            return self.history[-1].get("context", "")
        return None
    
    def clear_history(self):
        """Clear conversation history."""
        self.history = []
        logger.info("Conversation history cleared")
    
    def format_history_for_display(self) -> str:
        """Format history as a readable string."""
        if not self.history:
            return "No conversation history yet."
        
        formatted = "Conversation History:\n" + "="*60 + "\n"
        for i, exchange in enumerate(self.history, 1):
            formatted += f"\n[Exchange {i}]\n"
            formatted += f"Q: {exchange['question']}\n"
            formatted += f"A: {exchange['answer'][:200]}...\n" if len(exchange['answer']) > 200 else f"A: {exchange['answer']}\n"
            formatted += "-"*60 + "\n"
        
        return formatted
    
    def get_summary(self) -> dict:
        """Get a summary of the conversation."""
        return {
            "total_exchanges": len(self.history),
            "questions": [ex["question"] for ex in self.history],
            "answers": [ex["answer"][:100] + "..." for ex in self.history]
        }


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Initialize chatbot
    chatbot = RAGChatbot(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4",
        temperature=0.7
    )
    
    # Example context (would come from vectorstore retrieval)
    context = """
    Machine learning is a subset of artificial intelligence that enables 
    systems to learn and improve from experience without being explicitly programmed. 
    It focuses on the development of computer programs that can access data and use 
    it to learn for themselves.
    """
    
    # Generate answer
    question = "What is machine learning?"
    answer = chatbot.generate_answer(question, context)
    print(f"Q: {question}")
    print(f"A: {answer}")
    
    # Initialize conversation manager
    conversation = ConversationManager()
    conversation.add_exchange(question, answer, context)
    
    # Print history
    print("\n" + conversation.format_history_for_display())
