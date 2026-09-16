"""Regression-test home for stable adaptation outputs.

Use this module only after a behaviour has been deliberately established and
needs protection against accidental change, for example:
- a fixed synthetic stream's update count
- a known classifier-version sequence
- a documented CSW/CSV-to-update mapping
- a published or repository-accepted benchmark value

Do not use regression assertions to force uncertain paper values to match.
"""
