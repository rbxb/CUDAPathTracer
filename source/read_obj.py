import numpy as np

def __expand(v, indices, step):
    if len(v) == 0:
        return None
    a = np.zeros(len(indices) * step, dtype=np.float32)
    for i in range(len(indices)):
        for k in range(step):
            a[i*step+k] = v[indices[i]*step+k]
            pass
    return a


def read_obj(path):
    objects = []
    mtllib = None

    with open(path, 'r') as f:
        name = ''

        verts = []
        tex_coords = []
        normals = []
        
        v_indices = []
        t_indices = []
        n_indices = []

        material = None

        for line in f.readlines():
            first = None
            fan_count = 0
            fan_start = len(v_indices)

            for token in line.split():
                if token[0] == '#':
                    break
                if first == None:
                    first = token
                elif first == 'v':
                    verts.append(float(token))
                elif first == 'vt':
                    tex_coords.append(float(token))
                elif first == 'vn':
                    tex_coords.append(float(token))
                elif first == 'f':
                    face_element_count = 0

                    if (fan_count > 2):
                        v_indices += v_indices[fan_start:fan_start+2]
                        if len(t_indices) > 0:
                            t_indices += t_indices[fan_start:fan_start+2]
                        if len(n_indices) > 0:
                            n_indices += n_indices[fan_start:fan_start+2]

                    fan_count += 1

                    indices = token.split('/')
                    v_indices.append(int(indices[0]) - 1)
                    if len(indices) > 1:
                        t_indices.append(int(indices[1]) - 1)
                    if len(indices) > 2:
                        n_indices.append(int(indices[2]) - 1)
                elif first == 'usemtl':
                    material = token
                elif first == 'mtllib':
                    mtllib = token
                elif first == 'g':
                    pass
                elif first == 'o':
                    name = token
                    if len(verts) > 0:
                        objects.append({
                            'path': path,
                            'name': name,
                            'verts': __expand(verts, v_indices, 3),
                            'tex_coords': __expand(tex_coords, t_indces, 2),
                            'normals': __expand(normals, n_indices, 3),
                            'material': material,
                        })

                        verts = []
                        tex_coords = []
                        normals = []
                        
                        v_indices = []
                        t_indices = []
                        n_indices = []
                else:
                    raise ValueError('Error parsing OBJ file, unknown first token: %s  ' % first)

    if len(verts) > 0:
        objects.append({
            'path': path,
            'name': name,
            'verts': __expand(verts, v_indices, 3),
            'tex_coords': __expand(tex_coords, t_indices, 2),
            'normals': __expand(normals, n_indices, 3),
            'material': material,
        })

    return objects, mtllib
        
            

    
    
