"""The complex systems a breeder experiment can search.

Mirrors `cax.cs`: one package per system, each binding its own genotype space and
development to its config. `core/` never imports from here — a config's `build` hands
the search a `ComplexSystem`, and that carrier is the only interface.
"""
