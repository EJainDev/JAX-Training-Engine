from typing import Any


class FakeCheckpointer:
    def __init__(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass

    def restore(self, *args, **kwargs) -> Any:
        pass
