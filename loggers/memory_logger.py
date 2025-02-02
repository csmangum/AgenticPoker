import logging

logger = logging.getLogger(__name__)


class MemoryLogger:
    """Handles all logging operations for memory store operations."""

    @staticmethod
    def log_init_error(error: Exception, persist_dir: str) -> None:
        """Log directory creation errors."""
        logger.error(f"Failed to create directory {persist_dir}: {error}")

    @staticmethod
    def log_cleanup_warning(path: str) -> None:
        """Log cleanup warnings."""
        logger.warning(f"Could not remove existing directory: {path}")

    @staticmethod
    def log_cleanup_item_error(path: str, error: Exception) -> None:
        """Log errors during item cleanup."""
        logger.warning(f"Could not remove {path}: {error}")

    @staticmethod
    def log_collection_status(name: str, is_new: bool = False) -> None:
        """Log collection initialization status."""
        action = "Created new" if is_new else "Using existing"
        logger.info(f"{action} collection: {name}")

    @staticmethod
    def log_collection_warning(msg: str) -> None:
        """Log collection-related warnings."""
        logger.warning(msg)

    @staticmethod
    def log_max_id_error(error: Exception) -> None:
        """Log errors during max ID determination."""
        logger.warning(f"Could not determine max ID: {error}")

    @staticmethod
    def log_chroma_init_error(error: Exception) -> None:
        """Log ChromaDB initialization errors."""
        logger.error(f"Failed to initialize ChromaDB: {error}")

    @staticmethod
    def log_memory_add_error(error: Exception, attempts: int) -> None:
        """Log memory addition errors."""
        logger.error(f"Failed to add memory after {attempts} attempts: {error}")

    @staticmethod
    def log_collection_reinit(
        msg: str = "Collection is None, reinitializing...",
    ) -> None:
        """Log collection reinitialization."""
        logger.warning(msg)

    @staticmethod
    def log_query_warning(query: str) -> None:
        """Log query-related warnings."""
        logger.warning(f"No results found for query: {query}")

    @staticmethod
    def log_collection_error(error: Exception, attempt: int) -> None:
        """Log collection errors during query."""
        logger.warning(
            f"Collection error during query (attempt {attempt + 1}): {error}"
        )

    @staticmethod
    def log_collection_recovery_error() -> None:
        """Log collection recovery failures."""
        logger.error("Failed to recover collection after multiple attempts")

    @staticmethod
    def log_memory_retrieval_error(error: Exception) -> None:
        """Log memory retrieval errors."""
        logger.error(f"Error retrieving memories: {error}")

    @staticmethod
    def log_clear_error(error: Exception) -> None:
        """Log memory clearing errors."""
        if "no such table" not in str(error):  # Ignore if table already gone
            logger.error(f"Failed to clear memories: {error}")

    @staticmethod
    def log_connection_close(name: str) -> None:
        """Log successful connection closure."""
        logger.info(f"Closed ChromaDB connection for {name}")

    @staticmethod
    def log_close_error(error: Exception) -> None:
        """Log connection closure errors."""
        if "Python is likely shutting down" not in str(error):
            logger.error(f"Error closing ChromaDB connection: {error}")

    @staticmethod
    def log_client_warning(error: Exception) -> None:
        """Log client-related warnings."""
        if "Python is likely shutting down" not in str(error):
            logger.warning(f"Error closing client: {error}")

    @staticmethod
    def log_info(message: str) -> None:
        """Log an informational message."""
        logger.info(f"[Memory] {message}")

    @staticmethod
    def log_warning(message: str) -> None:
        """Log a warning message."""
        logger.warning(f"[Memory] {message}")

    @staticmethod
    def log_error(message: str) -> None:
        """Log an error message."""
        logger.error(f"[Memory] {message}")
