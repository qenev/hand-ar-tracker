"""Pytest configuration and global fixtures.

In Windows Python 3.13 environment, MediaPipe does not include the legacy
solutions submodule. This conftest dynamically mocks these missing submodules
