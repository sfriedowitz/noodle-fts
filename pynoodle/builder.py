from pynoodle import (
    Block,
    CubicCell,
    Hexagonal2DCell,
    Hexagonal3DCell,
    LamellarCell,
    Mesh,
    MonoclinicCell,
    Monomer,
    ObliqueCell,
    OrthorhombicCell,
    Point,
    Polymer,
    RectangularCell,
    RhombohedralCell,
    SquareCell,
    System,
    TetragonalCell,
    TriclinicCell,
    UnitCell,
)
from pynoodle.configs import (
    BlockConfig,
    CubicCellConfig,
    Hexagonal2DCellConfig,
    Hexagonal3DCellConfig,
    LamellarCellConfig,
    MeshConfig,
    MonoclinicCellConfig,
    MonomerConfig,
    ObliqueCellConfig,
    OrthorhombicCellConfig,
    PointConfig,
    PolymerConfig,
    RectangularCellConfig,
    RhombohedralCellConfig,
    SquareCellConfig,
    SystemConfig,
    TetragonalCellConfig,
    TriclinicCellConfig,
    UnitCellConfig,
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
        return Monomer(id=config.id, volume=config.volume)

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
    def build_cell(config: UnitCellConfig) -> UnitCell:
        """Build a UnitCell from configuration.

        Args:
            config: Unit cell configuration

        Returns:
            UnitCell instance
        """
        match config:
            case LamellarCellConfig(a=a):
                return LamellarCell(a=a)
            case SquareCellConfig(a=a):
                return SquareCell(a=a)
            case RectangularCellConfig(a=a, b=b):
                return RectangularCell(a=a, b=b)
            case Hexagonal2DCellConfig(a=a):
                return Hexagonal2DCell(a=a)
            case ObliqueCellConfig(a=a, b=b, gamma=gamma):
                return ObliqueCell(a=a, b=b, gamma=gamma)
            case CubicCellConfig(a=a):
                return CubicCell(a=a)
            case TetragonalCellConfig(a=a, c=c):
                return TetragonalCell(a=a, c=c)
            case OrthorhombicCellConfig(a=a, b=b, c=c):
                return OrthorhombicCell(a=a, b=b, c=c)
            case RhombohedralCellConfig(a=a, alpha=alpha):
                return RhombohedralCell(a=a, alpha=alpha)
            case Hexagonal3DCellConfig(a=a, c=c):
                return Hexagonal3DCell(a=a, c=c)
            case MonoclinicCellConfig(a=a, b=b, c=c, beta=beta):
                return MonoclinicCell(a=a, b=b, c=c, beta=beta)
            case TriclinicCellConfig(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma):
                return TriclinicCell(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
            case _:
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
