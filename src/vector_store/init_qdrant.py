import logging
import os
import time

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct

from src.data.process_data import load_and_process_data
from src.evaluation.best_configuration import find_best_configuration
from src.vector_store.RAGConfig import RAGConfig
from src.vector_store.data_embeddings import get_embeddings_from_chunked_text, compute_embeddings

# Initialize the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_collection(client: QdrantClient, collection_name: str, vector_dimension: int) -> None:
    """Create a new collection in Qdrant with the specified configuration."""
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_dimension,
                distance=models.Distance.COSINE
            )
        )
    except Exception as e:
        logger.exception("Failed to create collection '%s': %s", collection_name, e)
        raise


def upload_data_to_collection(
    client: QdrantClient,
    collection_name: str,
    embeddings,
    data,
) -> None:
    """Upload data points to the Qdrant collection in batches."""
    points = [
        PointStruct(
            id=i,
            vector=emb,
            payload={
                'Wiki_content': chunk['Wiki_content'],
                **chunk['Metadata']
            }
        )
        for i, (emb, chunk) in enumerate(zip(embeddings, data))
    ]

    batch_size = 30
    for start in range(0, len(points), batch_size):
        batch_points = points[start:start + batch_size]
        try:
            client.upload_points(collection_name=collection_name, points=batch_points, max_retries=3)
        except Exception as e:
            logger.exception("Failed to upload points to collection '%s': %s", collection_name, e)
            raise
    logger.info("Stored %d embeddings in collection '%s'.", len(points), collection_name)


def check_collection_status(client: QdrantClient, collection_name: str) -> None:
    max_retries = 30
    retry_count = 0
    while retry_count < max_retries:
        try:
            status = client.get_collection(collection_name=collection_name).status
            if status == 'green':
                logger.info("🟢 Collection is ready and operational.")
                break
            elif status == 'yellow':
                logger.info("🟡 Collection is optimizing. Please wait...")
            elif status == 'grey':
                logger.info("⚫ Collection is pending optimization.")
            elif status == 'red':
                logger.error("🔴 An unrecoverable error occurred in the engine.")
                break
            time.sleep(1)
            retry_count += 1
        except Exception as e:
            logger.exception("Failed to get collection status: %s", e)
            break
    else:
        logger.error("Collection did not become ready after %d retries.", max_retries)


def init_qdrant(config: RAGConfig, mode: str, prepared_data=None) -> None:
    try:
        client = QdrantClient(host='qdrant', port=6333)
    except Exception as e:
        logger.exception("Failed to connect to Qdrant client: %s", e)
        return

    collection_name = 'igem_eval' if mode == 'evaluation' else 'igem'

    if mode == 'production':
        # In production, always compute embeddings
        logger.info("Production mode: Computing data embeddings...")
        if not client.collection_exists(collection_name):
            try:
                create_collection(client, collection_name, config.vector_dimension)
            except Exception as e:
                logger.exception("Failed to create collection '%s': %s", collection_name, e)
                return

        try:
            embeddings, data = get_embeddings_from_chunked_text(
                config.dataset, config.chunk_size, config.embedding_model, config.tokenizer
            )
        except Exception as e:
            logger.exception("Failed to get embeddings from chunked text: %s", e)
            return
        try:
            upload_data_to_collection(client, collection_name, embeddings, data)
            logger.info("Stored %d embeddings in collection '%s'.", len(data), collection_name)
        except Exception as e:
            logger.exception("Failed to create collection or upload data: %s", e)
            return
        check_collection_status(client, collection_name)

    elif mode == 'evaluation':
        if client.collection_exists(collection_name):
            try:
                client.delete_collection(collection_name)
                logger.info(f"Collection '{collection_name}' deleted successfully.")
            except Exception as e:
                logger.exception("Failed to delete existing collection '%s': %s", collection_name, e)
                return
        try:
            create_collection(client, collection_name, config.vector_dimension)
        except Exception as e:
            logger.exception("Failed to create collection '%s': %s", collection_name, e)
            return

        # In evaluation, expect precomputed summarised chunks to be provided
        if prepared_data is None:
            logger.error("Evaluation mode expects precomputed summarised chunks.")
            return
        elif prepared_data is not None:
            embeddings = [compute_embeddings(doc, config.tokenizer, config.embedding_model) for doc in prepared_data]
            try:
                upload_data_to_collection(client, collection_name, embeddings, prepared_data)
                logger.info("Stored %d embeddings in collection '%s'.", len(prepared_data), collection_name)
            except Exception as e:
                logger.exception("Failed to create collection or upload data: %s", e)
                return


def main() -> None:
    try:
        if os.path.isfile('src/data/processed_data.xlsx'):
            logger.info("Data has already been processed and will be used to populate vector store.")
        else:
            logger.info("Data has not been processed yet; preparing data for vector store population.")
            load_and_process_data()
            logger.info("Data has been processed successfully and will be loaded into vector store next.")

        best_config_name = find_best_configuration()

        if best_config_name:
            # Extract parameters from the best config name
            dataset, best_chunk_size, best_embedding_model, best_score = best_config_name
            if dataset and best_chunk_size and best_embedding_model:
                config = RAGConfig(embedding_model_name=best_embedding_model,
                                   chunk_size=best_chunk_size)
                init_qdrant(config, mode='production')
                logger.info(
                    f"Initialized Qdrant with best configuration: Chunk Size={best_chunk_size},"
                    f" Embedding Model={best_embedding_model}")

        else:
            logger.info("No valid configuration found based on previous evaluations;"
                        " running with default configuration.")
            config = RAGConfig(
                embedding_model_name='sentence-transformers/all-MiniLM-L6-v2',
                chunk_size=250
            )
            init_qdrant(config=config, mode='production')

    except Exception as e:
        logger.exception("An error occurred during the main execution: %s", e)


if __name__ == '__main__':
    main()
