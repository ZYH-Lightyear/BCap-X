"""Errors intentionally returned as M1.3 function results."""


class ContextFunctionError(RuntimeError):
    """A dispatch, reference or backend failure visible to the policy."""


__all__ = ["ContextFunctionError"]
