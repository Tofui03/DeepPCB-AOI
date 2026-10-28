"""Loaded before test modules. Lets the API import the mock fixture in pytest only."""

from __future__ import annotations

import os

os.environ["DEEPPCB_ALLOW_MOCK"] = "1"
