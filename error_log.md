# Error Log

This file tracks errors and issues encountered during the development of the 360 VR Person Segmentation Pipeline.

| Date | Error/Issue | Resolution |
|------|-------------|------------|
| 2025-04-26 | Import error when running the pipeline via run.sh: `ImportError: attempted relative import with no known parent package` | Fixed by modifying import structure to work both when imported as a module and when run directly as a script. Updated run.sh to use `python -m src.main` to treat main as a module. |
