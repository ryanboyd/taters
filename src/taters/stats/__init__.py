"""
Statistical engines shared across feature extractors.

The text and audio steps *produce* feature tables; the modules here *reduce
and model* them, and are deliberately generic over any Taters feature CSV --
the same PCA that powers the MEM topic model can reduce a wall of dictionary
scores before a regression.
"""
