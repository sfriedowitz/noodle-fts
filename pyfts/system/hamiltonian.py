from pyfts.system.system import System


class Hamiltonian:
    def __init__(self):
        # Flory-Huggins chi-parameters
        self.interactions: dict[tuple[str, str], float] = {}

    def set_interaction(self, a: str, b: str, chi: float) -> None:
        if a != b:
            key = tuple(sorted((a, b)))
            self.interactions[key] = chi

    def interaction_energy(self, system: System) -> float:
        pass

    def interaction_potentials(self, system: System) -> float:
        pass

    def free_energy(self, system: System) -> float:
        pass

    def free_energy_bulk(self, system: System) -> float:
        pass
