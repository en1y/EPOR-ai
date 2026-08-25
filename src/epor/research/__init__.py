"""Public API for EPOR's provenance-aware research archive."""

from .catalog import DEFAULT_CATALOG_PATH, load_catalog, select_sources
from .models import (
    CatalogError,
    EvidenceClass,
    IndexResult,
    IndexStatus,
    IntegrityStatus,
    ResearchCatalog,
    ResearchEntry,
    SyncResult,
    SyncStatus,
)
from .service import index_catalog, sync_catalog, verify_catalog

__all__ = [
    "DEFAULT_CATALOG_PATH",
    "CatalogError",
    "EvidenceClass",
    "IndexResult",
    "IndexStatus",
    "IntegrityStatus",
    "ResearchCatalog",
    "ResearchEntry",
    "SyncResult",
    "SyncStatus",
    "index_catalog",
    "load_catalog",
    "select_sources",
    "sync_catalog",
    "verify_catalog",
]
