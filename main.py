import pygame as pg
import moderngl as mgl
import numpy as np
import math


class RaytracerApp:
    def __init__(self):
        pg.init()
        self.screen_size = (1024, 768)
        self.screen = pg.display.set_mode(self.screen_size, pg.OPENGL | pg.DOUBLEBUF)
        pg.display.set_caption("Raytracer")

        pg.mouse.set_visible(False)
        pg.event.set_grab(True)

        self.ctx = mgl.create_context()

        def load_shader(filepath):
            with open(filepath, 'r') as file:
                return file.read()

        self.texture = self.ctx.texture(self.screen_size, 4, dtype='f4')
        compute_src = load_shader('shaders/raytracer.comp')
        self.compute_prog = self.ctx.compute_shader(compute_src)

        vert_src = load_shader('shaders/quad.vert')
        frag_src = load_shader('shaders/quad.frag')
        self.quad_prog = self.ctx.program(
            vertex_shader=vert_src,
            fragment_shader=frag_src
        )

        # note that the values in your quad buffer are defined using two different coordinate systems packed into a single array:
        # Normalized Device Coordinates (NDC) for position and UV coordinates for texture mapping
        # NDC (-1,-1) boittom left; (1,1) top right; (0,0) center)
        # UV coordinates: (0,0) bottom left, (1,1) top right)
        # buffer encoding: [x,y,u,v] and four vertices to spann the quad:
        #   columns: NDC_x, NDC_y, uv_u, uv_v
        #   rows: the four vertices
        self.quad_buffer = self.ctx.buffer(np.array(
            [
                -1.0, 1.0, 0.0, 1.0,
                -1.0, -1.0, 0.0, 0.0,
                1.0, 1.0, 1.0, 1.0,
                1.0, -1.0, 1.0, 0.0,
            ], dtype='f4'))
        self.quad_vao = self.ctx.simple_vertex_array(self.quad_prog, self.quad_buffer, 'in_vert', 'in_uv')

        self.cam_pos = np.array([0.0, 0.0, 5.0])
        self.yaw = -90.0
        self.pitch = 0.0
        self.fps = 0.0

    def get_camera_vectors(self):
        rad_yaw, rad_pitch = math.radians(self.yaw), math.radians(self.pitch)
        front = np.array(
            [
                math.cos(rad_yaw) * math.cos(rad_pitch),
                math.sin(rad_pitch),
                math.sin(rad_yaw) * math.cos(rad_pitch)
            ]
        )
        front /= np.linalg.norm(front)
        right = np.cross(front, [0, 1, 0])
        right /= np.linalg.norm(right)
        up = np.cross(right, front)
        return front, right, up / np.linalg.norm(up)

    def handle_input(self, dt):
        keys = pg.key.get_pressed()
        speed = 2.0 * dt
        front, right, up = self.get_camera_vectors()
        if keys[pg.K_w]:
            self.cam_pos += front * speed
        if keys[pg.K_s]:
            self.cam_pos -= front * speed
        if keys[pg.K_a]:
            self.cam_pos -= right * speed
        if keys[pg.K_d]:
            self.cam_pos += right * speed

        rel_x, rel_y = pg.mouse.get_rel()
        self.yaw += rel_x * 0.1
        self.pitch = max(-89.0, min(89.0, self.pitch - rel_y * 0.1))

    def render(self):
        front, right, up = self.get_camera_vectors()

        # dispatch Compute Shader
        self.texture.bind_to_image(0, read=False, write=True)
        self.compute_prog['cam_pos'].value = tuple(self.cam_pos)
        self.compute_prog['cam_dir'].value = tuple(front)
        self.compute_prog['cam_right'].value = tuple(right)
        self.compute_prog['cam_up'].value = tuple(up)

        gx, gy = math.ceil(self.screen_size[0] / 16), math.ceil(self.screen_size[1] / 16)
        self.compute_prog.run(gx, gy)

        # Compute shaders do not belong into standard graphics pipeline => their results are wrtitten into a texture object.
        # To see the results a quad (i.e. two triangles) are rendered (-1, -1) and (1,1)
        # Render Texture to Quad
        self.ctx.clear()
        self.texture.use(0)
        self.quad_vao.render(mgl.TRIANGLE_STRIP)

        pg.display.flip()

    def run(self):
        clock = pg.time.Clock()
        while True:
            dt = clock.tick(0) / 1000.0
            if dt > 0: self.fps = 1.0 / dt

            for event in pg.event.get():
                if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                    return

            self.handle_input(dt)
            self.render()

            # Update Window Title with FPS (Most efficient way in Pygame+OpenGL)
            pg.display.set_caption(f"FPS: {int(self.fps)}")


if __name__ == "__main__":
    RaytracerApp().run()
