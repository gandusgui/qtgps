import numpy as np

def ov2vertices(o, v, vertices):
    for i, z in enumerate([o[2], o[2]+v[2]]):
        vertices[0+i*4,:] = o[0], o[1], z
        vertices[1+i*4,:] = o[0]+v[0], o[1], z
        vertices[2+i*4,:] = o[0]+v[0], o[1]+v[1], z
        vertices[3+i*4,:] = o[0], o[1]+v[1], z
