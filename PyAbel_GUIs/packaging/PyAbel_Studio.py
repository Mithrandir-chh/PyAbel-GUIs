"""Entry script for the packaged PyAbel Studio applications.

From a source checkout, start the same application from the repository root
with ``python -m PyAbel_GUIs.PyAbel_Application``.
"""

from PyAbel_GUIs.PyAbel_Application import main


if __name__ == "__main__":
    raise SystemExit(main())
