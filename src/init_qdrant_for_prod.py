import logging
from qdrant_client import QdrantClient
from src.vector_store.init_qdrant import main as init_qdrant_main
from src.vector_store.init_qdrant import check_collection_status


# Initialize the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def is_qdrant_populated_for_production(client: QdrantClient) -> bool:
    """Check if the Qdrant vector store already has data in the production collection named 'igem'."""
    if client.collection_exists('igem'):
        info = client.get_collection(collection_name='igem')
        if info.points_count != 0:
            return True
        else:
            return False
    else:
        return False


def initialize_database():
    client = QdrantClient(host='qdrant', port=6333)
    logger.info("Connected to Qdrant at qdrant:6333")

    if is_qdrant_populated_for_production(client):
        logger.info("Vector store is populated, checking its availability...")
        check_collection_status(client=client, collection_name='igem')

    else:
        logger.info("The database is not populated with data. Populating now, this can last for long...")
        init_qdrant_main()
        logger.info("Database initialized successfully.")


if __name__ == '__main__':
    initialize_database()
