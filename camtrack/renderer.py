#! /usr/bin/env python3

__all__ = [
    'CameraTrackRenderer'
]

from typing import List, Tuple

import numpy as np
from OpenGL import GL
from OpenGL.GL import shaders
from OpenGL import GLUT
from OpenGL.arrays import vbo

import data3d


def _build_program_color():
    vertex_shader = shaders.compileShader(
        """
        #version 140
        uniform mat4 mvp;

        in  vec3 color;
        in  vec3 position;
        out vec3 cur_color;
        
        void main() {
            vec4 camera_space_position = mvp * vec4(position, 1.0);
            gl_Position = camera_space_position;
            cur_color = color;
        }""",
        GL.GL_VERTEX_SHADER
    )
    fragment_shader = shaders.compileShader(
        """
        #version 140
        
        in  vec3 cur_color;
        out vec3 out_color; 
        
        
        void main() {
            out_color = cur_color;
        }""",
        GL.GL_FRAGMENT_SHADER
    )

    return shaders.compileProgram(
        vertex_shader, fragment_shader
    )


def _build_program_uniform_color():
    vertex_shader = shaders.compileShader(
        """
        #version 140
        uniform mat4 mvp;

        in vec3 position;

        void main() {
            vec4 camera_space_position = mvp * vec4(position, 1.0);
            gl_Position = camera_space_position;
        }""",
        GL.GL_VERTEX_SHADER
    )
    fragment_shader = shaders.compileShader(
        """
        #version 140
        uniform vec3 color_all;
        out     vec3 out_color;

        void main() {
            out_color = color_all;
        }""",
        GL.GL_FRAGMENT_SHADER
    )

    return shaders.compileProgram(
        vertex_shader, fragment_shader
    )


def _to_mat4(rot_mat, pos_vec):
    return np.block([
            [rot_mat, pos_vec[:,np.newaxis]],
            [np.zeros((1, 3)), np.ones((1,1))]
    ])


def _to_buffer(items):
    return vbo.VBO(np.array(items, dtype=np.float32))


class CameraTrackRenderer:

    def __init__(self,
                 cam_model_files: Tuple[str, str],
                 tracked_cam_parameters: data3d.CameraParameters,
                 tracked_cam_track: List[data3d.Pose],
                 point_cloud: data3d.PointCloud):
        """
        Initialize CameraTrackRenderer. Load camera model, create buffer objects, load textures,
        compile shaders, e.t.c.

        :param cam_model_files: path to camera model obj file and texture. The model consists of
        triangles with per-point uv and normal attributes
        :param tracked_cam_parameters: tracked camera field of view and aspect ratio. To be used
        for building tracked camera frustrum
        :param point_cloud: colored point cloud
        """

        self._program_color = _build_program_color()
        self._program_uniform_color = _build_program_uniform_color()

        self.cam_model_files = cam_model_files
        self.tracked_cam_parameters = tracked_cam_parameters
        self.tracked_cam_track = tracked_cam_track
        self.point_cloud = point_cloud

        self._points_buffer = _to_buffer(point_cloud.points)
        self._point_color_buffer = _to_buffer(point_cloud.colors)
        self._track_points = _to_buffer([track.t_vec for track in tracked_cam_track])

        GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA | GLUT.GLUT_DOUBLE | GLUT.GLUT_DEPTH)
        GL.glEnable(GL.GL_DEPTH_TEST)

    def display(self, camera_tr_vec, camera_rot_mat, camera_fov_y, tracked_cam_track_pos_float):
        """
        Draw everything with specified render camera position, projection parameters and 
        tracked camera position

        :param camera_tr_vec: vec3 position of render camera in global space
        :param camera_rot_mat: mat3 rotation matrix of render camera in global space
        :param camera_fov_y: render camera field of view. To be used for building a projection
        matrix. Use glutGet to calculate current aspect ratio
        :param tracked_cam_track_pos_float: a frame in which tracked camera
        model and frustrum should be drawn (see tracked_cam_track_pos for basic task)
        :return: returns nothing
        """

        # a frame in which a tracked camera model and frustrum should be drawn
        # without interpolation
        tracked_cam_track_pos = int(tracked_cam_track_pos_float)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        V = np.linalg.inv(_to_mat4(camera_rot_mat, camera_tr_vec))
        M = np.diag([1, -1, -1, 1])
        P = self._get_projection_matrix(camera_fov_y, 0.01, 100)
        mvp = np.dot(np.dot(P, V), M)

        cur_track_pos = self.tracked_cam_track[tracked_cam_track_pos].t_vec

        pyramid_top, pyramid_bottom = self._get_pyramid(self.tracked_cam_track[tracked_cam_track_pos])
        pyramid_left = pyramid_top[2:] + pyramid_bottom[:2]
        pyramid_right = pyramid_top[2:] + pyramid_bottom[2:]

        self._render(mvp, self._points_buffer, gl_type=GL.GL_POINTS, color_buffer=self._point_color_buffer)
        self._render(mvp, self._track_points, gl_type=GL.GL_LINES, color_all=(1., 1., 1.))
        self._render(mvp, _to_buffer([cur_track_pos]), gl_type=GL.GL_POINTS, color_all=(0., 1., 0.))

        self._render(mvp, _to_buffer(pyramid_top), gl_type=GL.GL_LINE_LOOP, color_all=(0., 1., 0.))
        self._render(mvp, _to_buffer(pyramid_bottom), gl_type = GL.GL_LINE_LOOP, color_all=(0., 1., 0.))
        self._render(mvp, _to_buffer(pyramid_left), gl_type=GL.GL_LINE_LOOP, color_all=(0., 1., 0.))
        self._render(mvp, _to_buffer(pyramid_right), gl_type=GL.GL_LINE_LOOP, color_all=(0., 1., 0.))


        GLUT.glutSwapBuffers()

    @staticmethod
    def _get_projection_matrix(fov_y, near, far):
        w = float(GLUT.glutGet(GLUT.GLUT_WINDOW_WIDTH))
        h = float(GLUT.glutGet(GLUT.GLUT_WINDOW_HEIGHT))
        fov_x = fov_y / w * h
        a = - (far + near) / (far - near)
        b = - 2 * far * near / (far - near)

        return np.asarray([
            [fov_x, 0, 0, 0],
            [0, fov_y, 0, 0],
            [0,    0,  a, b],
            [0,    0, -1, 0]
        ])

    def _get_pyramid(self, cur_track, near=1, far=25):
        fov_y = self.tracked_cam_parameters.fov_y
        aspect = self.tracked_cam_parameters.aspect_ratio
        fov_x = fov_y * aspect
        V_inv = _to_mat4(cur_track.r_mat, cur_track.t_vec).astype(np.float32)

        def get_part(dist):
            points = []
            for i, j in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                points.append(dist * np.asarray([i * fov_x, j * fov_y, 1], dtype=np.float32))

            return [np.dot(V_inv, np.append(point, [1.]))[:3] for point in points]

        return get_part(near), get_part(far)

    def _render(self, mvp, position_buffer, gl_type, color_all=(0., 0., 1.), color_buffer=None):
        program = self._program_color if (color_buffer is not None) else self._program_uniform_color

        shaders.glUseProgram(program)

        GL.glUniformMatrix4fv(
            GL.glGetUniformLocation(program, 'mvp'),
            1, True, mvp
        )

        buf_size = position_buffer.size

        color_loc = None
        if color_buffer is not None:
            color_buffer.bind()
            color_loc = GL.glGetAttribLocation(program, 'color')
            GL.glEnableVertexAttribArray(color_loc)
            GL.glVertexAttribPointer(color_loc, 3, GL.GL_FLOAT, False, 0, color_buffer)
        else:
            color_all = np.array(color_all, dtype=np.float32)
            GL.glUniform3fv(
                GL.glGetUniformLocation(program, 'color_all'),
                1, True, color_all
            )

        position_buffer.bind()
        position_loc = GL.glGetAttribLocation(program, 'position')
        GL.glEnableVertexAttribArray(position_loc)
        GL.glVertexAttribPointer(position_loc, 3, GL.GL_FLOAT, False, 0,  position_buffer)

        GL.glDrawArrays(gl_type, 0, buf_size)

        GL.glDisableVertexAttribArray(position_loc)
        position_buffer.unbind()

        if color_buffer is not None:
            GL.glDisableVertexAttribArray(color_loc)
            color_buffer.unbind()

