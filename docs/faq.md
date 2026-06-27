# FAQ

**Why replay instead of restore?** High-confidence exact matches save tokens and improve consistency.

**When is verify triggered?** Facts, workflows, tool outputs, or memories flagged `requires_verification=True`.

**How do I debug decisions?** Call `decision.explain()` for score breakdown and reason tags.
