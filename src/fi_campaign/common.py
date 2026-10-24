"""Canonical serialization used to identify prompt templates.

The campaign's path management and scheduling utilities are not included in
this public subset. The serializer below is unchanged from the executed code.
"""
import json


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
