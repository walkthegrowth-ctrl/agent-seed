"""Tool reservoir: ready-made tools an agent composer can attach.

Each ``*.py`` file should define one or more plain tool functions (a name, a one-line
docstring, and type-annotated parameters) that are **self-contained** — they must not
depend on sibling helpers or module-level state. ``agent compose --list-tools`` scans
this folder (ast-only, never imported) and offers them; a chosen tool's source is
inlined into the generated agent folder, keeping that agent self-contained.

Override the folder with ``$TOOLS_ROOT``. This is the seed set — add your own files.
"""
