# Setup Wizard

The interactive console wizard behind the `taters` command, and the layers it
is built from. See the [wizard guide](../guides/wizard.md) for what it looks
like to use.

The modules are deliberately separate. `introspect`, `recipes` and `compose`
have no terminal dependency at all, so they can back a different front end — a
GUI, a web page — without the logic moving. `prompts` defines the seam, and
`live` is one renderer on the far side of it.

## The front door

The opening menu, and the registry of things a user can choose to do. Each task
is a self-contained flow, so adding one is a new module and a registry entry
rather than a change to anything that already works.

::: taters.ui.hub
    options:
      members_order: source
      show_source: true
      show_root_heading: true

::: taters.ui.tasks
    options:
      members_order: source
      show_source: true
      show_root_heading: true

## Building a pipeline

::: taters.ui.wizard
    options:
      members_order: source
      show_source: true
      show_root_heading: true

## Describing a function to a UI

Reads a function's signature and numpydoc docstring into renderable field
descriptions. This is what lets the wizard offer a module's options without
anyone hand-writing a list of them.

::: taters.ui.introspect
    options:
      members_order: source
      show_source: true
      show_root_heading: true

## The recipe catalog

The declared wiring: which steps exist, what each one needs, and what it
produces. Signatures cannot tell you that the transcription step's output feeds
the acoustics step's `transcript_csv`, so that part is written down here.

::: taters.ui.recipes
    options:
      members_order: source
      show_source: true
      show_root_heading: true

## Composing a preset

Turns a set of chosen features into a runnable preset: resolves prerequisites,
orders the steps, and writes the metadata. Pure — no I/O, no analysis imports.

::: taters.ui.compose
    options:
      members_order: source
      show_source: true
      show_root_heading: true

## Prompting

The interface the wizard asks its questions through, and the implementations:
one backed by `questionary`, one that reads canned answers for the test suite.

::: taters.ui.prompts
    options:
      members_order: source
      show_source: true
      show_root_heading: true

## Choosing a file

A filesystem browser built from the ordinary selection prompt, so it needs
nothing from the renderer that the scripted prompter cannot also do — which is
what makes it testable without a terminal.

::: taters.ui.browse
    options:
      members_order: source
      show_source: true
      show_root_heading: true

## Showing progress

Turns the pipeline runner's event stream into stacked progress bars: one for the
run, one for the current step, and one for each file being worked on.

::: taters.ui.run_display
    options:
      members_order: source
      show_source: true
      show_root_heading: true

## The live-region renderer

The default look of the `taters` command: a progress rail pinned above whatever
question is open, drawn inline so the terminal's scrollback survives the run.

::: taters.ui.live
    options:
      members_order: source
      show_source: true
      show_root_heading: true

## Terminal capability

What the terminal can draw, as opposed to what it says it can. A client
connecting over SSH announces a terminal type that understates its colors and
says nothing at all about its font, so these two modules hold the defaults that
make the screen look right anyway.

::: taters.ui.console
    options:
      members_order: source
      show_source: true
      show_root_heading: true

::: taters.ui.glyphs
    options:
      members_order: source
      show_source: true
      show_root_heading: true
