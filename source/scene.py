from read_obj import read_obj

def parse_scene(path):
    scene = {'objects':[], 'camera':None}
    with open(path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[0]
            tokens = list(map(lambda t: t.strip(), line.split()))
            if tokens[0] == 'object':
                scene['objects'] += read_obj(' '.join(tokens[1:]))
            elif tokens[0] == 'translate':
                scene['objects'][-1].translate = parse_translate(tokens[1:])
            elif tokens[0] == 'rotate':
                scene['objects'][-1].rotate = parse_rotate(tokens[1:])
            elif tokens[0] == 'scale':
                scene['objects'][-1].scale = parse_scale(tokens[1:])
            elif tokens[0] == 'camera':
                scene['camera'] = parse_camera(tokens[1:])
            else:
                raise ValueError('Error parsing scene file on line %d:\n\t%s', i, line)

    return scene


def parse_translate(tokens):
    return None


def parse_rotate(tokens):
    return None


def parse_scale(tokens):
    return None


def parse_camera(tokens):
    return None