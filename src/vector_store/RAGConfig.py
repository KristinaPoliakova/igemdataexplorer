from transformers import AutoTokenizer, AutoModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGConfig:
    def __init__(self, chunk_size: int, embedding_model_name: str):
        self.chunk_size = chunk_size
        self.embedding_model_name = embedding_model_name

        # Initialize dataset path using an environment variable
        self.dataset = 'src/data/processed_data.xlsx'

        # Load model and tokenizer with error handling
        try:
            self.embedding_model = AutoModel.from_pretrained(self.embedding_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
            self.vector_dimension = self.embedding_model.config.hidden_size
            logger.info("Model and tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model/tokenizer: {str(e)}")
            raise

    def __str__(self):
        return (f"RAGConfig("
                f"dataset={self.dataset}, "
                f"chunk_size={self.chunk_size}, "
                f"embedding_model_name={self.embedding_model_name}"
                )
