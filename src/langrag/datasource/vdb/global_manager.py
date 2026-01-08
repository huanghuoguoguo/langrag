from langrag.datasource.vdb.vector_manager import VectorManager, DefaultVectorManager

_GLOBAL_MANAGER: VectorManager = DefaultVectorManager()

def get_vector_manager() -> VectorManager:
    """Access the global vector manager instance."""
    return _GLOBAL_MANAGER

def set_vector_manager(manager: VectorManager):
    """
    Set the global vector manager.
    To be called by External/Web layer during startup.
    """
    global _GLOBAL_MANAGER
    _GLOBAL_MANAGER = manager
