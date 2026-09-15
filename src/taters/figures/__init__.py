"""
Figures drawn from finished results.

Everything here reads tables an analysis has already written and draws
pictures of them; nothing here computes a statistic. That split is
deliberate: a figure can be redrawn with different settings without a
refit, and a table stands on its own without the figure.

:mod:`taters.figures.render` is the word-cloud renderer (layout, sizes,
colors, the PNG). :mod:`taters.figures.wordclouds` decides which words go
into which cloud for each kind of result.
"""
