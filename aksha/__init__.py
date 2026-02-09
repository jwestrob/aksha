"""Aksha: Scalable HMM-based sequence search and retrieval.

Example usage:
    from aksha import search, scan, install_database
    
    # Install a database
    install_database("PFAM")
    
    # Search sequences
    result = search("proteins.faa", "PFAM", thresholds=ThresholdOptions(cut_ga=True))
    
    # Get results as DataFrame
    df = result.to_dataframe()
"""

from aksha.types import (
    ThresholdOptions,
    SearchResult,
    SequenceHit,
    DomainHit,
    MoleculeType,
    BitscoreCutoff,
    DatabaseInfo,
)
from aksha.search import search
from aksha.scan import scan
from aksha.phmmer import phmmer
from aksha.jackhmmer import jackhmmer
from aksha.nhmmer import nhmmer
from aksha.databases import (
    install_database,
    uninstall_database,
    list_databases,
    register_custom_database,
)
from aksha.config import get_config, get_registry

__version__ = "0.1.0"
__all__ = [
    # Core functions
    "search",
    "scan",
    "phmmer",
    "jackhmmer",
    "nhmmer",
    # Database management
    "install_database",
    "uninstall_database",
    "list_databases",
    "register_custom_database",
    # Types
    "ThresholdOptions",
    "SearchResult",
    "SequenceHit",
    "DomainHit",
    "MoleculeType",
    "BitscoreCutoff",
    "DatabaseInfo",
    # Config
    "get_config",
    "get_registry",
]
