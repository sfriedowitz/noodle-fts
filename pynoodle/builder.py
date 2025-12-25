from pynoodle import (
    Block,
    CubicCell,
    Hexagonal2DCell,
    LamellarCell,
    Mesh,
    Monomer,
    Point,
    Polymer,
    SquareCell,
    System,
)
from pynoodle.configs import (
    BlockConfig,
    CubicCellConfig,
    Hexagonal2DCellConfig,
    LamellarCellConfig,
    MeshConfig,
    MonomerConfig,
    PointConfig,
    PolymerConfig,
    SquareCellConfig,
    SystemConfig,
)


class SystemBuilder:
    """Builds System instances from configuration objects."""

    @staticmethod
    def build_monomer(config: MonomerConfig) -> Monomer:
        """Build a Monomer from configuration.

        Args:
            config: Monomer configuration

        Returns:
            Monomer instance
        """
        return Monomer(id=config.id, size=config.size)

    @staticmethod
    def build_block(config: BlockConfig, monomers: dict[int, Monomer]) -> Block:
        """Build a Block from configuration.

        Args:
            config: Block configuration
            monomers: Dictionary mapping monomer IDs to Monomer instances

        Returns:
            Block instance
        """
        return Block(
            monomer=monomers[config.monomer_id],
            repeat_units=config.repeat_units,
            segment_length=config.segment_length,
        )

    @staticmethod
    def build_point(config: PointConfig, monomers: dict[int, Monomer]) -> Point:
        """Build a Point species from configuration.

        Args:
            config: Point configuration
            monomers: Dictionary mapping monomer IDs to Monomer instances

        Returns:
            Point instance
        """
        return Point(monomer=monomers[config.monomer_id], phi=config.phi)

    @staticmethod
    def build_polymer(config: PolymerConfig, monomers: dict[int, Monomer]) -> Polymer:
        """Build a Polymer species from configuration.

        Args:
            config: Polymer configuration
            monomers: Dictionary mapping monomer IDs to Monomer instances

        Returns:
            Polymer instance
        """
        blocks = [SystemBuilder.build_block(block, monomers) for block in config.blocks]
        return Polymer(blocks=blocks, contour_steps=config.contour_steps, phi=config.phi)

    @staticmethod
    def build_mesh(config: MeshConfig) -> Mesh:
        """Build a Mesh from configuration.

        Args:
            config: Mesh configuration

        Returns:
            Mesh instance
        """
        return Mesh(*config.dimensions)

    @staticmethod
    def build_cell(
        config: LamellarCellConfig | SquareCellConfig | Hexagonal2DCellConfig | CubicCellConfig,
    ) -> LamellarCell | SquareCell | Hexagonal2DCell | CubicCell:
        """Build a UnitCell from configuration.

        Args:
            config: Unit cell configuration

        Returns:
            UnitCell instance
        """
        if isinstance(config, LamellarCellConfig):
            return LamellarCell(a=config.a)
        elif isinstance(config, SquareCellConfig):
            return SquareCell(a=config.a)
        elif isinstance(config, Hexagonal2DCellConfig):
            return Hexagonal2DCell(a=config.a)
        elif isinstance(config, CubicCellConfig):
            return CubicCell(a=config.a)
        else:
            msg = f"Unknown cell type: {type(config)}"
            raise ValueError(msg)

    @staticmethod
    def build_system(config: SystemConfig) -> System:
        """Build a System from configuration.

        Args:
            config: System configuration

        Returns:
            System instance with all interactions set
        """
        # Build monomers
        monomers = {m.id: SystemBuilder.build_monomer(m) for m in config.monomers}

        # Build domain
        mesh = SystemBuilder.build_mesh(config.mesh)
        cell = SystemBuilder.build_cell(config.cell)

        # Build species
        species = []
        for spec in config.species:
            if isinstance(spec, PointConfig):
                species.append(SystemBuilder.build_point(spec, monomers))
            elif isinstance(spec, PolymerConfig):
                species.append(SystemBuilder.build_polymer(spec, monomers))

        # Create system
        system = System(mesh=mesh, cell=cell, species=species)

        # Set interactions
        for interaction in config.interactions:
            system.set_interaction(interaction.i, interaction.j, interaction.chi)

        return system
