"""The conductor: one subscription chat owns an order from request to answer.

ADR-0043. Pure building blocks, wired into the mission path behind
``missions.conductor`` (off | shadow | on):

* :mod:`.lanes`   - which executor does a part of an order, cheapest first.
* :mod:`.leads`   - the four roles an order's parts are handed to.
* :mod:`.ledger`  - task and progress records, stall and loop detection.
* :mod:`.lessons` - rules learned from failures, injected into later plans.
"""
from __future__ import annotations
