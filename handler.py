"""Runpod Serverless entry point.

The Hub looks for ``handler.py`` at the repository root, so this file exists
where the platform expects it while the implementation stays in ``app/``.
"""

from app.handler import handler, main

__all__ = ["handler", "main"]

if __name__ == "__main__":
    main()
