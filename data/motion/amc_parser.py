import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import euler2mat
from mpl_toolkits.mplot3d import Axes3D


class Joint:
    def __init__(self, name, direction, length, axis, dof, limits):
        """
    Definition of basic joint. The joint also contains the information of the
    bone between it's parent joint and itself. Refer
    [here](https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/ASF-AMC.html)
    for detailed description for asf files.
    Parameter
    ---------
    name: Name of the joint defined in the asf file. There should always be one
    root joint. String.
    direction: Default direction of the joint(bone). The motions are all defined
    based on this default pose.
    length: Length of the bone.
    axis: Axis of rotation for the bone.
    dof: Degree of freedom. Specifies the number of motion channels and in what
    order they appear in the AMC file.
    limits: Limits on each of the channels in the dof specification

    基本关节的定义。 关节还包含其父关节与其自身之间的骨骼信息。 有关 asf 文件的详细说明，请参阅“https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/ASF-AMC.html”。

    范围
     ---------
     name：asf 文件中定义的关节名称。 总应该有一个
     根关节。 细绳。
     方向：关节（骨骼）的默认方向。 动作都是基于这个默认姿势定义的。
     长度：骨头的长度。
     axis：骨骼的旋转轴。
     dof：自由度。 指定运动通道的数量以及它们在 AMC 文件中出现的顺序。
     限制：自由度规范中每个通道的限制
    """
        self.name = name
        self.direction = np.reshape(direction, [3, 1])
        self.length = length
        axis = np.deg2rad(axis)
        self.C = euler2mat(*axis)
        self.Cinv = np.linalg.inv(self.C)
        self.limits = np.zeros([3, 2])
        for lm, nm in zip(limits, dof):
            if nm == 'rx':
                self.limits[0] = lm
            elif nm == 'ry':
                self.limits[1] = lm
            else:
                self.limits[2] = lm
        self.parent = None
        self.children = []
        self.coordinate = None
        self.matrix = None

    def set_motion(self, motion):
        if self.name == 'root':
            self.coordinate = np.reshape(np.array(motion['root'][:3]), [3, 1])
            rotation = np.deg2rad(motion['root'][3:])
            self.matrix = self.C.dot(euler2mat(*rotation)).dot(self.Cinv)
        else:
            idx = 0
            rotation = np.zeros(3)
            for axis, lm in enumerate(self.limits):
                if not np.array_equal(lm, np.zeros(2)):
                    rotation[axis] = motion[self.name][idx]
                    idx += 1
            rotation = np.deg2rad(rotation)
            self.matrix = self.parent.matrix.dot(self.C).dot(euler2mat(*rotation)).dot(self.Cinv)
            self.coordinate = self.parent.coordinate + self.length * self.matrix.dot(self.direction)
        for child in self.children:
            child.set_motion(motion)

    def get_name_to_idx(self):
        joints = self.to_dict()
        name_to_idx = {}
        for idx, joint in enumerate(joints.values()):
            assert joint.name not in name_to_idx
            name_to_idx[joint.name] = idx
        self.name_to_idx = name_to_idx

    def output_edges(self):
        joints = self.to_dict()
        name_to_idx = self.name_to_idx
        edges = []
        for idx, joint in enumerate(joints.values()):
            child = joint
            if child.parent is not None:
                parent = child.parent
                edges.append([name_to_idx[child.name], name_to_idx[parent.name]])
        return edges

    def output_coord(self):
        N = len(self.name_to_idx)
        X = np.zeros((N, 3))
        joints = self.to_dict()
        name_to_idx = self.name_to_idx
        for idx, joint in enumerate(joints.values()):
            X[name_to_idx[joint.name]] = joint.coordinate.reshape(-1)
        return X

    def draw(self):
        joints = self.to_dict()
        fig = plt.figure()
        ax = Axes3D(fig)

        ax.set_xlim3d(-50, 10)
        ax.set_ylim3d(-20, 40)
        ax.set_zlim3d(-20, 40)

        xs, ys, zs = [], [], []
        for joint in joints.values():
            xs.append(joint.coordinate[0, 0])
            ys.append(joint.coordinate[1, 0])
            zs.append(joint.coordinate[2, 0])
        plt.plot(zs, xs, ys, 'b.')

        for joint in joints.values():
            child = joint
            if child.parent is not None:
                parent = child.parent
                xs = [child.coordinate[0, 0], parent.coordinate[0, 0]]
                ys = [child.coordinate[1, 0], parent.coordinate[1, 0]]
                zs = [child.coordinate[2, 0], parent.coordinate[2, 0]]
                plt.plot(zs, xs, ys, 'r')
        plt.show()

    def to_dict(self):
        ret = {self.name: self}
        for child in self.children:
            ret.update(child.to_dict())
        return ret

    def pretty_print(self):
        print('===================================')
        print('joint: %s' % self.name)
        print('direction:')
        print(self.direction)
        print('limits:', self.limits)
        print('parent:', self.parent)
        print('children:', self.children)


def read_line(stream, idx):
    if idx >= len(stream):
        return None, idx
    line = stream[idx].strip().split()
    idx += 1
    return line, idx


def parse_asf(file_path):
    '''read joint data only'''
    with open(file_path) as f:
        content = f.read().splitlines()

    for idx, line in enumerate(content):
        # meta infomation is ignored
        if line == ':bonedata':
            content = content[idx + 1:]
            break

    # read joints
    joints = {'root': Joint('root', np.zeros(3), 0, np.zeros(3), [], [])}
    idx = 0
    while True:
        # the order of each section is hard-coded

        line, idx = read_line(content, idx)

        if line[0] == ':hierarchy':
            break

        assert line[0] == 'begin'

        line, idx = read_line(content, idx)
        assert line[0] == 'id'

        line, idx = read_line(content, idx)
        assert line[0] == 'name'
        name = line[1]

        line, idx = read_line(content, idx)
        assert line[0] == 'direction'
        direction = np.array([float(axis) for axis in line[1:]])

        # skip length
        line, idx = read_line(content, idx)
        assert line[0] == 'length'
        length = float(line[1])

        line, idx = read_line(content, idx)
        assert line[0] == 'axis'
        assert line[4] == 'XYZ'

        axis = np.array([float(axis) for axis in line[1:-1]])

        dof = []
        limits = []

        line, idx = read_line(content, idx)
        if line[0] == 'dof':
            dof = line[1:]
            for i in range(len(dof)):
                line, idx = read_line(content, idx)
                if i == 0:
                    assert line[0] == 'limits'
                    line = line[1:]
                assert len(line) == 2
                mini = float(line[0][1:])
                maxi = float(line[1][:-1])
                limits.append((mini, maxi))

            line, idx = read_line(content, idx)

        assert line[0] == 'end'
        joints[name] = Joint(
            name,
            direction,
            length,
            axis,
            dof,
            limits
        )

    # read hierarchy
    assert line[0] == ':hierarchy'

    line, idx = read_line(content, idx)

    assert line[0] == 'begin'

    while True:
        line, idx = read_line(content, idx)
        if line[0] == 'end':
            break
        assert len(line) >= 2
        for joint_name in line[1:]:
            joints[line[0]].children.append(joints[joint_name])
        for nm in line[1:]:
            joints[nm].parent = joints[line[0]]

    return joints


def parse_amc(file_path):
    with open(file_path) as f:
        content = f.read().splitlines()

    for idx, line in enumerate(content):
        if line == ':DEGREES':
            content = content[idx + 1:]
            break

    frames = []
    idx = 0
    line, idx = read_line(content, idx)
    assert line[0].isnumeric(), line
    EOF = False
    while not EOF:
        joint_degree = {}
        while True:
            line, idx = read_line(content, idx)
            if line is None:
                EOF = True
                break
            if line[0].isnumeric():
                break
            joint_degree[line[0]] = [float(deg) for deg in line[1:]]
        frames.append(joint_degree)
    return frames
