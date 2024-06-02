from abc import ABC, abstractmethod

from pyfts.system.system import System


class FieldUpdater(ABC):
    @abstractmethod
    def step(self, system: System) -> None:
        raise NotImplementedError
