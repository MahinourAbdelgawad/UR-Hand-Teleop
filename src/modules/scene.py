import mujoco
import numpy as np

class Scene:
    def __init__(self):
        self.TABLE_POS = (0.0,  0.0, 0.0) # table centre (x, y, z of top surface)
        self.TABLE_SIZE = (0.4,  0.5, 0.02) 
        self.TABLE_Z = 0.02 

        self.BLOCK_SIZE = 0.025 
        self.BLOCK_MASS = 0.05  #in kg

        # Block positions on table
        self.BLOCK_A_POS = (0.25, 0.10, self.TABLE_Z + self.BLOCK_SIZE)
        self.BLOCK_B_POS = (0.25, -0.10, self.TABLE_Z + self.BLOCK_SIZE)

        # Palette positions 
        self.PALETTE_SIZE = (0.07, 0.07, 0.002)
        self.PALETTE_A_POS = (-0.10,  0.25, self.TABLE_Z + 0.002)
        self.PALETTE_B_POS = (-0.10, -0.25, self.TABLE_Z + 0.002)

    def build_scene(self):
        """
        Returns an MjSpec with the scene
        """
        spec = mujoco.MjSpec()

        spec.worldbody.add_light(
            name="main_light",
            pos=(0, 0, 2.5),
            dir=(0, 0, -1),
            diffuse=(0.8, 0.8, 0.8),
            specular=(0.2, 0.2, 0.2),
            castshadow=True,
        )

        spec.worldbody.add_geom(
            name="ground",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=(2.0, 2.0, 0.1),
            pos=(0, 0, 0),
            rgba=(0.75, 0.75, 0.75, 1),
            friction=(0.8, 0.005, 0.0001),
        )

        table = spec.worldbody.add_body(name="table")
        table.pos = (self.TABLE_POS[0], self.TABLE_POS[1], self.TABLE_SIZE[2])
        table.add_geom(
            name="table_top",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=self.TABLE_SIZE,
            rgba=(0.55, 0.40, 0.25, 1),
            friction=(0.8, 0.005, 0.0001),
        )

        self._add_block(spec, "block_a", self.BLOCK_A_POS, (0.85, 0.2, 0.2, 1))# red
        self._add_block(spec, "block_b", self.BLOCK_B_POS, (0.2, 0.4, 0.85, 1)) # blue

        self._add_palette(spec, "palette_a", self.PALETTE_A_POS, (0.2, 0.75, 0.3, 0.5)) # green
        self._add_palette(spec, "palette_b", self.PALETTE_B_POS, (0.85, 0.75, 0.1, 0.5)) # yellow

        return spec


    def _add_block(self, spec, name, pos, rgba):
        body = spec.worldbody.add_body(name=name)
        body.pos = pos
        body.add_freejoint(name=f"{name}_joint")
        body.add_geom(
            name=f"{name}_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE),
            rgba=rgba,
            mass=self.BLOCK_MASS,
            friction=(0.9, 0.005, 0.0001),
        )


    def _add_palette(self, spec, name, pos, rgba):
        body = spec.worldbody.add_body(name=name)
        body.pos = pos
        body.add_geom(
            name=f"{name}_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=self.PALETTE_SIZE,
            rgba=rgba,
            contype=0, # no collision
            conaffinity=0,
        )