"""
The console setup wizard, and the layers it is built from.

Run ``taters`` to use it. See :mod:`taters.ui.wizard` for the flow.

Four modules, each usable without the one above it::

    wizard      the questions, in order
    prompts     how a question reaches a person
    compose     answers -> a runnable preset
    recipes     the declared data flow between steps
    introspect  a function -> renderable field descriptions

Only :mod:`taters.ui.prompts` knows what a terminal is. Everything below it is
plain data, which is what would let a graphical front end reuse the same logic
rather than reimplementing it.
"""
