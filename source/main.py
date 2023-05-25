import argparse
import logging

import ctypes
from ctypes import *

import cv2

from read_obj import read_obj


def bounding_sphere(verts):
    a = verts.ctypes.data_as(POINTER(c_float32))
    n = len(verts).ctypes.data_as(c_int)


    __boundingSphere(a, n, center, radius)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='rbxb_renderer',
                    description='Render a scene using CUDA')
    parser.add_argument('objectfile')
    parser.add_argument('-l', '-library', default='./rbxb_render_lib.so')
    args = parser.parse_args()

    #dll = ctypes.CDLL(args.l, mode=ctypes.RTLD_GLOBAL)

    objects, mtllib = read_obj(args.objectfile)

    cube = objects[0]
    texture = cv2.imread('texture.png')

    bounding_radius, bounding_center = bounding_sphere(cube['verts'])

    camera = np.array([6, -2, -10, -5.5, 2.5, 10.5], dtype=np.float32)
    
    out = np.zeros((256, 256, 3), dtype=np.uint8)
    device_rays, n_rays = create_camera_rays(camera, out)

    m = intersect_sphere(device_rays, np.arange(n_rays, dtype=np.int), bounding_radius, bounding_center)

    row = m[0]
    indices = m.nonzero()

    device_geometry = upload_geometry(cube)
    color, depth = intersect(device_rays, indices, device_geometry)

    apply_color(out, color, depth, indices)
    cv2.imwrite('out.png', out)
        
