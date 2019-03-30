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
        
        uniform vec3 color;
        out     vec3 out_color;

        void main() {
            out_color = color;
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

        face_template = [(-1, 1), (-1, -1), (1, -1), (1, 1)]

        pyramid_near  = [[x, y, -1] for x, y in face_template]
        pyramid_far   = [[x, y,  1] for x, y in face_template]
        pyramid_left  = pyramid_near[:2] + pyramid_far[:2][::-1]
        pyramid_right = pyramid_near[2:] + pyramid_far[2:][::-1]


        self._pyramid_faces = [
            _to_buffer(pyramid_near),
            _to_buffer(pyramid_far),
            _to_buffer(pyramid_left),
            _to_buffer(pyramid_right)
        ]
        self._track_points_buffers = [_to_buffer([track.t_vec]) for track in tracked_cam_track]

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

        w = float(GLUT.glutGet(GLUT.GLUT_WINDOW_WIDTH))
        h = float(GLUT.glutGet(GLUT.GLUT_WINDOW_HEIGHT))
        widow_aspect = w / h

        V = np.linalg.inv(_to_mat4(camera_rot_mat, camera_tr_vec))
        gl2cv = cv2gl = np.diag([1, -1, -1, 1])

        P = self._get_projection_matrix(camera_fov_y, widow_aspect, 0.01, 100)
        mvp = np.dot(np.dot(P, cv2gl), V)

        cur_track = self.tracked_cam_track[tracked_cam_track_pos]
        V_inv_pyramid = _to_mat4(cur_track.r_mat, cur_track.t_vec).astype(np.float32)
        P_pyramid = self._get_projection_matrix(self.tracked_cam_parameters.fov_y,
                                                self.tracked_cam_parameters.aspect_ratio,
                                                1, 22)
        P_inv_pyramid = np.linalg.inv(P_pyramid)
        mvp_pyramid = np.dot(mvp, np.dot(np.dot(np.dot(V_inv_pyramid, gl2cv), P_inv_pyramid), cv2gl))

        for pyramid_face in self._pyramid_faces:
            self._render(mvp_pyramid, _to_buffer(pyramid_face), 4, gl_fig_type=GL.GL_LINE_LOOP, color_all=(0., 0.9, 0.))

        self._render(mvp, self._points_buffer, len(self.point_cloud.points), gl_fig_type=GL.GL_POINTS,
                     color_buffer=self._point_color_buffer)
        self._render(mvp, self._track_points, len(self.tracked_cam_track), gl_fig_type=GL.GL_LINES,
                     color_all=(0.9, 0.9, 0.9))
        self._render(mvp, self._track_points_buffers[tracked_cam_track_pos], 1, gl_fig_type=GL.GL_POINTS, color_all=(0., 0.9, 0.))

        GLUT.glutSwapBuffers()

    @staticmethod
    def _get_projection_matrix(fov_y, aspect, near, far):
        f_y = 1. / np.tan(fov_y / 2)
        f_x = f_y / aspect  # aspect: w / h
        a = - (far + near) / (far - near)
        b = - 2 * far * near / (far - near)

        return np.asarray([
            [f_x, 0,   0, 0],
            [0, f_y,   0, 0],
            [0,    0,  a, b],
            [0,    0, -1, 0]
        ])

    def _render(self, mvp, position_buffer, buf_size, gl_fig_type, color_all=(0., 0., 0.9), color_buffer=None):
        program = self._program_color if (color_buffer is not None) else self._program_uniform_color

        shaders.glUseProgram(program)

        GL.glUniformMatrix4fv(
            GL.glGetUniformLocation(program, 'mvp'),
            1, True, mvp
        )

        color_loc = None
        if color_buffer is not None:
            color_buffer.bind()
            color_loc = GL.glGetAttribLocation(program, 'color')
            GL.glEnableVertexAttribArray(color_loc)
            GL.glVertexAttribPointer(color_loc, 3, GL.GL_FLOAT, False, 0, color_buffer)
        else:
            color_all = np.array(color_all, dtype=np.float32)
            color_loc = GL.glGetUniformLocation(program, 'color')
            GL.glUniform3fv(color_loc, 1, color_all)

        position_buffer.bind()
        position_loc = GL.glGetAttribLocation(program, 'position')
        GL.glEnableVertexAttribArray(position_loc)
        GL.glVertexAttribPointer(position_loc, 3, GL.GL_FLOAT, False, 0,  position_buffer)

        GL.glDrawArrays(gl_fig_type, 0, buf_size)

        GL.glDisableVertexAttribArray(position_loc)
        position_buffer.unbind()

        if color_buffer is not None:
            GL.glDisableVertexAttribArray(color_loc)
            color_buffer.unbind()

