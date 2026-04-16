"""Compatibility shim for the old Daydream CLI package name.

This repository now exposes the speech CLI under the ``dwhisper`` package
and ``dwhisper`` command to avoid colliding with the original Daydream CLI.
"""

LEGACY_MESSAGE = (
    "This repository no longer exposes the speech CLI as 'daydream'. "
    "Use 'dwhisper' or 'python -m dwhisper' instead."
)
