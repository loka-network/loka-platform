"""loka_adapters — concrete implementations of the MemoryAdapter contract.

The contract itself lives in loka_schemas.adapter; this package provides implementations.
"""

from .memory import InMemoryAdapter

__all__ = ["InMemoryAdapter"]
