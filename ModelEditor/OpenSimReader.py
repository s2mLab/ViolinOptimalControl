# coding: utf-8

from lxml import etree
import inspect
import numpy as np
from numpy.linalg import inv
from ConvertModel import *


def index_go_to(_root, _tag, _attrib='False', _attribvalue='', index=''):
    # return index to go to _tag which can have condition on its attribute
    i = 0
    for _child in _root:
        if type(_child) == str:
            return ''
        if _attrib != 'False':
            if _child.tag == _tag and _child.get(_attrib) == _attribvalue:
                return index + '[{}]'.format(i)
            else:
                i += 1
        else:
            if _child.tag == _tag:
                return index + '[{}]'.format(i)
            else:
                i += 1
                # not found in children, go to grand children
    else:
        j = 0
        if _root is not None:
            for _child in _root:
                a = index_go_to(_child, _tag, _attrib, _attribvalue, index + '[{}]'.format(j))
                if a:
                    return index_go_to(_child, _tag, _attrib, _attribvalue, index + '[{}]'.format(j))
                else:
                    j += 1
            else:
                return None


def retrieve_name(var):
    """
    Gets the name of var. Does it from the out most frame inner-wards.
    :param var: variable to get name from.
    :return: string
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]


def go_to(_root, _tag, _attrib='False', _attribvalue=''):
    # return element corresponding to _tag
    # which can have condition on its attribute
    _index = index_go_to(_root, _tag, _attrib, _attribvalue)
    if _index is None:
        return 'None'
    else:
        _index = index_go_to(_root, _tag, _attrib, _attribvalue)
        return eval(retrieve_name(_root) + _index)


def coord_sys(axis):
    # define orthonormal coordinate system with given z-axis
    [a, b, c] = axis
    if a == 0:
        if b == 0:
            if c == 0:
                return [[1, 0, 0], [0, 1, 0], [0, 0, 1]], ''
            else:
                return [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 'z'
        else:
            if c == 0:
                return [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 'y'
            else:
                y_temp = [0, -c / b, 1]
    else:
        if b == 0:
            if c == 0:
                return [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 'x'
            else:
                y_temp = [-c / a, 0, 1]
        else:
            y_temp = [-b / a, 1, 0]
    z_temp = [a, b, c]
    x_temp = np.cross(y_temp, z_temp)
    norm_x_temp = np.linalg.norm(x_temp)
    norm_z_temp = np.linalg.norm(z_temp)
    x = [1 / norm_x_temp * x_el for x_el in x_temp]
    z = [1 / norm_z_temp * z_el for z_el in z_temp]
    y = [y_el for y_el in np.cross(z, x)]
    return [x, y, z], ''


class OrthoMatrix:
    def __init__(self, translation=[0, 0, 0], rotation_1=[0, 0, 0], rotation_2=[0, 0, 0], rotation_3=[0, 0, 0]):
        self.trans = np.transpose(np.array([translation]))
        self.axe_1 = rotation_1  # axis of rotation for theta_1
        self.axe_2 = rotation_2  # axis of rotation for theta_2
        self.axe_3 = rotation_3  # axis of rotation for theta_3
        self.rot_1 = np.transpose(np.array(coord_sys(self.axe_1)[0]))  # rotation matrix for theta_1
        self.rot_2 = np.transpose(np.array(coord_sys(self.axe_2)[0]))  # rotation matrix for theta_2
        self.rot_3 = np.transpose(np.array(coord_sys(self.axe_3)[0]))  # rotation matrix for theta_3
        self.rotation_matrix = self.rot_3.dot(self.rot_2.dot(self.rot_1))  # rotation matrix for
        self.matrix = np.append(np.append(self.rotation_matrix, self.trans, axis=1), np.array([[0, 0, 0, 1]]), axis=0)

    def get_rotation_matrix(self):
        return self.rotation_matrix

    def set_rotation_matrix(self, rotation_matrix):
        self.rotation_matrix = rotation_matrix

    def get_translation(self):
        return self.trans

    def set_translation(self, trans):
        self.trans = trans

    def get_matrix(self):
        self.matrix = np.append(np.append(self.rotation_matrix, self.trans, axis=1), np.array([[0, 0, 0, 1]]), axis=0)
        return self.matrix

    def transpose(self):
        self.rotation_matrix = np.transpose(self.rotation_matrix)
        self.trans = -self.rotation_matrix.dot(self.trans)
        self.matrix = np.append(np.append(self.rotation_matrix, self.trans, axis=1), np.array([[0, 0, 0, 1]]), axis=0)
        return self.matrix

    def product(self, other):
        self.rotation_matrix = self.rotation_matrix.dot(other.get_rotation_matrix())
        self.trans = self.trans + other.get_translation()
        self.matrix = np.append(np.append(self.rotation_matrix, self.trans, axis=1), np.array([[0, 0, 0, 1]]), axis=0)

    def get_axis(self):
        return coord_sys(self.axe_1)[1] + coord_sys(self.axe_2)[1] + coord_sys(self.axe_3)[1]


def out_product(rotomatrix_1, rotomatrix_2):
    rotomatrix_prod = OrthoMatrix()
    rotomatrix_prod.set_translation(rotomatrix_1.get_translation() + rotomatrix_2.get_translation())
    rotomatrix_prod.set_rotation_matrix(rotomatrix_1.get_rotation_matrix().dot(rotomatrix_2.get_rotation_matrix()))
    rotomatrix_prod.get_matrix()
    return rotomatrix_prod


class OpenSimReader:
    def __init__(self, originfile, version=3):
        self.originfile = originfile
        self.version = str(version)

        self.data_origin = etree.parse(self.originfile)
        self.root = self.data_origin.getroot()
        self.biorbd_model = BiorbdModel()

        written_version = self.data_origin.xpath('/OpenSimDocument')[0].get('Version')
        if int(written_version) < 4000:
            self.version_opensim = 3
        else:
            self.version_opensim = 4

        def new_text(element):
            if type(element) == str:
                return element
            else:
                return element.text

        def body_list(_self):
            L = []
            for _body in _self.data_origin.xpath(
                    '/OpenSimDocument/Model/BodySet/objects/Body'):
                L.append(_body.get("name"))
            return L

        def parent_body(_body, _late_body):
            ref = new_text(go_to(go_to(self.root, 'Body', 'name', _body), 'parent_body'))
            if ref == 'None':
                return _late_body
            else:
                return ref

        def matrix_inertia(_body):
            if self.version_opensim == 3:
                ref = new_text(go_to(go_to(self.root, 'Body', 'name', _body), 'inertia_xx'))
                if ref == 'None':
                    _inertia_str = new_text(go_to(go_to(self.root, 'Body', 'name', _body), 'inertia'))
                    _inertia = [float(s) for s in _inertia_str.split(' ')]
                    return _inertia
                else:
                    return [ref,
                            new_text(go_to(go_to(self.root, 'Body', 'name', _body), 'inertia_yy')),
                            new_text(go_to(go_to(self.root, 'Body', 'name', _body), 'inertia_zz')),
                            new_text(go_to(go_to(self.root, 'Body', 'name', _body), 'inertia_xy')),
                            new_text(go_to(go_to(self.root, 'Body', 'name', _body), 'inertia_xz')),
                            new_text(go_to(go_to(self.root, 'Body', 'name', _body), 'inertia_yz'))]
            elif self.version_opensim == 4:
                _ref = new_text(go_to(go_to(self.root, 'Body', 'name', _body), 'inertia'))
                if _ref != 'None':
                    _inertia_str = _ref
                    _inertia = [float(s) for s in _inertia_str.split(' ')]
                    return _inertia
                else:
                    return 'None'

        def muscle_list(_self):
            _list = []
            for _muscle in _self.data_origin.xpath(
                    '/OpenSimDocument/Model/ForceSet/objects/Thelen2003Muscle'):
                _list.append(_muscle.get("name"))
            return _list

        def list_pathpoint_muscle(_muscle):
            # return list of viapoint for each muscle
            _viapoint = []
            # TODO warning for other type of pathpoint
            index_pathpoint = index_go_to(go_to(self.root, 'Thelen2003Muscle', 'name', _muscle), 'PathPoint')
            _list_index = list(index_pathpoint)
            _tronc_list_index = _list_index[:len(_list_index) - 2]
            _tronc_index = ''.join(_tronc_list_index)
            index_root = index_go_to(self.root, 'Thelen2003Muscle', 'name', _muscle)
            index_tronc_total = index_root + _tronc_index
            _i = 0
            while True:
                try:
                    child = eval('self.root' + index_tronc_total + str(_i) + ']')
                    _viapoint.append(child.get("name"))
                    _i += 1
                except:  # Exception as e:   print('Error', e)
                    break
            return _viapoint

        def list_markers_body(_body):
            # return list of transformation for each body
            markers = []
            index_markers = index_go_to(self.root, 'Marker')
            if index_markers is None:
                return []
            else:
                _list_index = list(index_markers)
                _tronc_list_index = _list_index[:len(_list_index) - 2]
                _tronc_index = ''.join(_tronc_list_index)
                _i = 0
                while True:
                    try:
                        child = eval('self.root' + _tronc_index + str(_i) + ']').get('name')
                        which_body = new_text(go_to(go_to(self.root, 'Marker', 'name', child), 'socket_parent_frame'))[
                                     9:]
                        if which_body == _body:
                            markers.append(child) if child is not None else True
                        _i += 1
                    except:  # Exception as e:  print('Error', e)
                        break
                return markers

        # list of joints with parent and child
        list_joint = []
        index_joints = index_go_to(self.root, 'WeldJoint')
        if index_joints is not None:
            list_index = list(index_joints)
            tronc_list_index = list_index[:len(list_index) - 2]
            tronc_index = ''.join(tronc_list_index)
            i = 0
            while True:
                try:
                    new_joint = eval('self.root' + tronc_index + str(i) + ']').get('name')
                    if new_text(go_to(self.root, 'WeldJoint', 'name', new_joint)) != 'None':
                        _parent_joint = new_text(
                            go_to(go_to(self.root, 'WeldJoint', 'name', new_joint), 'socket_parent_frame'))[:-7]
                        _child_joint = new_text(
                            go_to(go_to(self.root, 'WeldJoint', 'name', new_joint), 'socket_child_frame'))[:-7]
                        list_joint.append([new_joint, _parent_joint, _child_joint, 'WeldJoint'])
                    i += 1
                except Exception as e:
                    # print('Error', e)
                    break
        index_joints = index_go_to(self.root, 'CustomJoint')
        if index_joints is not None:
            list_index = list(index_joints)
            tronc_list_index = list_index[:len(list_index) - 2]
            tronc_index = ''.join(tronc_list_index)
            i = int(list_index[len(list_index) - 2])
            while True:
                try:
                    new_joint = eval('self.root' + tronc_index + str(i) + ']').get('name')
                    if new_text(go_to(self.root, 'CustomJoint', 'name', new_joint)) != 'None':
                        _parent_joint = new_text(
                            go_to(go_to(self.root, 'CustomJoint', 'name', new_joint), 'socket_parent_frame'))[:-7]
                        _child_joint = new_text(
                            go_to(go_to(self.root, 'CustomJoint', 'name', new_joint), 'socket_child_frame'))[:-7]
                        list_joint.append([new_joint, _parent_joint, _child_joint, 'CustomJoint'])
                    i += 1
                except:  # Exception as e:print('Error', e)
                    break

        def dof_of_joint(_joint, _joint_type):
            dof = []
            _index_dof = index_go_to(go_to(self.root, _joint_type, 'name', _joint), 'Coordinate')
            if _index_dof is None:
                return []
            else:
                _list_index = list(_index_dof)
                _tronc_list_index = _list_index[:len(_list_index) - 2]
                _tronc_index = ''.join(_tronc_list_index)
                _index_root = index_go_to(self.root, _joint_type, 'name', _joint)
                _index_tronc_total = _index_root + _tronc_index
                _i = 0
                while True:
                    try:
                        child = eval('self.root' + _index_tronc_total + str(_i) + ']')
                        if child.get('name') is not None:
                            dof.append(child.get("name"))
                        _i += 1
                    except:
                        break
            return dof

        def parent_child(_child):
            # return parent of a child
            # suppose that a parent can only have one child
            for _joint in list_joint:
                if _joint[2] == _child:
                    return _joint[1]
            else:
                return 'None'

        def joint_body(_body):
            # return the joint to which the body is child
            for _joint in list_joint:
                if _joint[2] == _body:
                    return _joint[0], _joint[3]
            else:
                return 'None', 'None'

        def transform_of_joint(_joint, _joint_type):
            _translation = []
            _rotation = []
            if _joint is 'None':
                return [[], []]
            _index_transform = index_go_to(go_to(self.root, _joint_type, 'name', _joint), 'TransformAxis')
            if _index_transform is None:
                return [[], []]
            else:
                _list_index = list(_index_transform)
                _tronc_list_index = _list_index[:len(_list_index) - 2]
                _tronc_index = ''.join(_tronc_list_index)
                _index_root = index_go_to(self.root, _joint_type, 'name', _joint)
                if not _index_root:
                    pass
                _index_tronc_total = _index_root + _tronc_index
                i = 0
                while True:
                    try:
                        child = eval('self.root' + _index_tronc_total + str(i) + ']')
                        if child.get('name') is not None:
                            _translation.append(child.get("name")) \
                                if child.get('name').find('translation') == 0 else True
                            _rotation.append(child.get("name")) \
                                if child.get('name').find('rotation') == 0 else True
                        i += 1
                    except:  # Exception as e:  print('Error', e)
                        break
            return [_translation, _rotation]

        def get_body_pathpoint(_pathpoint):
            while True:
                try:
                    if index_go_to(go_to(self.root, 'PathPoint', 'name', _pathpoint),
                                   'socket_parent_frame') is not None or '':
                        _ref = new_text(go_to(
                            go_to(self.root, 'PathPoint', 'name', _pathpoint), 'socket_parent_frame'))
                        return _ref[9:]
                    if index_go_to(go_to(self.root, 'ConditionalPathPoint', 'name', _pathpoint),
                                   'socket_parent_frame') is not None or '':
                        _ref = new_text(go_to(
                            go_to(self.root, 'ConditionalPathPoint', 'name', _pathpoint), 'socket_parent_frame'))
                        return _ref[9:]
                    if index_go_to(go_to(self.root, 'MovingPathPoint', 'name', _pathpoint),
                                   'socket_parent_frame') is not None or '':
                        _ref = new_text(
                            go_to(go_to(self.root, 'MovingPathPoint', 'name', _pathpoint), 'socket_parent_frame'))
                        return _ref[9:]
                    else:
                        return 'None'
                except Exception as e:
                    break

        def get_pos(_pathpoint):
            while True:
                try:
                    if index_go_to(go_to(self.root, 'PathPoint', 'name', _pathpoint), 'location') != '':
                        return new_text(go_to(go_to(self.root, 'PathPoint', 'name', _pathpoint), 'location'))
                    elif index_go_to(go_to(self.root, 'ConditionalPathPoint', 'name', _pathpoint), 'location') != '':
                        return new_text(go_to(go_to(self.root, 'ConditionalPathPoint', 'name', _pathpoint), 'location'))
                    elif index_go_to(go_to(self.root, 'MovingPathPoint', 'name', _pathpoint), 'location') != '':
                        return new_text(go_to(go_to(self.root, 'MovingPathPoint', 'name', _pathpoint), 'location'))
                    else:
                        return 'None'
                except Exception as e:
                    break

        def muscle_group_reference(_muscle, ref_group):
            for el in ref_group:
                if _muscle == el[0]:
                    return el[1]
            else:
                return 'None'

        # Segment definition
        def add_segment(_body, _name, ):
            segment = Segment(_name, parent_name, _rotomatrix, dof_rotation, dof_translation)
        def printing_segment(_body, _name, parent_name, _rotomatrix, transformation_type='', _is_dof='None',
                             true_segment=False, _dof_total_trans=''):
            if _name == 'None':
                name = ''
            if parent_name == 'None':
                parent = ''
            rt_in_matrix = 1
            if _rotomatrix == 'None':
                rot_trans_matrix = [[], [], [], []]
            else:
                [[r11, r12, r13, r14],
                [r21, r22, r23, r24],
                [r31, r32, r33, r34],
                [r41, r42, r43, r44]] = _rotomatrix.get_matrix().tolist()
            for _i in range(4):
                for _j in range(4):
                    round(eval('r' + str(_i + 1) + str(_j + 1)), 9)
            [i11, i22, i33, i12, i13, i23] = matrix_inertia(_body)
            inertia = [[i11, i12, i13],
                        [i12, i22, i23],
                        [i13, i23, i33]]
            mass = new_text(go_to(go_to(self.root, 'Body', 'name', _body), 'mass'))
            com = new_text(go_to(go_to(self.root, 'Body', 'name', _body), 'mass_center'))
            path_mesh_file = new_text(go_to(go_to(self.root, 'Body', 'name', _body), 'mesh_file'))
            dof_rotation =
            if transformation_type == 'translation' and _dof_total_trans != '':
                dof_translation = dof_total_trans
            else:
                dof_translation = ''
            if _is_dof == 'True':
                dof_rotation = 'z'
            else:
                dof_rotation = ''
            # TODO add mesh files
            segment = Segment(name, parent_name, rot_trans_matrix, dof_rotation, dof_translation, mass, inertia, com, rt_in_matrix)
            # writing data
            self.write('    // Segment\n')
            self.write('    segment {}\n'.format(_name)) if _name != 'None' else self.write('')
            self.write('        parent {} \n'.format(parent_name)) if parent_name != 'None' else self.write('')
            self.write('        RTinMatrix    {}\n'.format(rt_in_matrix)) if rt_in_matrix != 'None' else self.write('')
            self.write('        RT\n')
            self.write(
                '            {}    {}    {}    {}\n'
                '            {}    {}    {}    {}\n'
                '            {}    {}    {}    {}\n'
                '            {}    {}    {}    {}\n'
                    .format(r11, r12, r13, r14,
                            r21, r22, r23, r24,
                            r31, r32, r33, r34,
                            r41, r42, r43, r44))
            self.write('        translations {}\n'.format(
                dof_total_trans)) if transformation_type == 'translation' and _dof_total_trans != '' else True
            self.write('        rotations {}\n'.format('z')) if _is_dof == 'True' else True
            self.write('        mass {}\n'.format(mass)) if true_segment is True else True
            self.write('        inertia\n'
                       '            {}    {}    {}\n'
                       '            {}    {}    {}\n'
                       '            {}    {}    {}\n'
                       .format(i11, i12, i13,
                               i12, i22, i23,
                               i13, i23, i33)) if true_segment is True else True
            self.write('        com    {}\n'.format(com)) if true_segment is True else True
            self.write('        //meshfile {}\n'.format(path_mesh_file)) if path_mesh_file != 'None' else True
            self.write('    endsegment\n')

        # Division of body in segment depending of transformation
        for body in body_list(self):
            rotomatrix = OrthoMatrix([0, 0, 0])
            self.write('\n// Information about {} segment\n'.format(body))
            parent = parent_child(body)
            if parent == 'ground':
                parent = 'None'
            joint, joint_type = joint_body(body)
            list_transform = transform_of_joint(joint, joint_type)
            rotation_for_markers = rotomatrix.get_rotation_matrix()
            # segment data
            if list_transform[0] == []:
                if list_transform[1] == []:
                    printing_segment(body, body, parent, rotomatrix, true_segment=True)
                    parent = body
            else:
                body_trans = body + '_translation'
                dof_total_trans = ''
                j = 0
                list_trans_dof = ['x', 'y', 'z']
                for translation in list_transform[0]:
                    if translation.find('translation') == 0:
                        axis_str = new_text(go_to(
                            go_to(go_to(self.root, joint_type, 'name', joint), 'TransformAxis', 'name', translation),
                            'axis'))
                        axis = [float(s) for s in axis_str.split(' ')]
                        rotomatrix.product(OrthoMatrix([0, 0, 0], axis))
                        is_dof = new_text(go_to(
                            go_to(go_to(self.root, joint_type, 'name', joint), 'TransformAxis', 'name', translation),
                            'coordinates'))
                        if is_dof in dof_of_joint(joint, joint_type):
                            dof_total_trans += list_trans_dof[j]
                    j += 1
                trans_str = new_text(go_to(
                    go_to(go_to(self.root, joint_type, 'name', joint), 'PhysicalOffsetFrame', 'name',
                          parent + '_offset'), 'translation'))
                trans_value = []
                for s in trans_str.split(' '):
                    if s != '' and s is not 'None':
                        trans_value.append(float(s))
                rotomatrix.product(OrthoMatrix(trans_value))
                rotation_for_markers = rotomatrix.get_rotation_matrix()
                if list_transform[1] == []:
                    is_true_segment = True
                else:
                    is_true_segment = False
                printing_segment(body, body_trans, parent, rotomatrix, 'translation', dof_total_trans,
                                 true_segment=is_true_segment)
                parent = body_trans
            if list_transform[1] != []:
                rotomatrix = OrthoMatrix([0, 0, 0])
                for rotation in list_transform[1]:
                    if rotation.find('rotation') == 0:
                        axis_str = new_text(
                            go_to(go_to(go_to(self.root, joint_type, 'name', joint), 'TransformAxis', 'name', rotation),
                                  'axis'))
                        axis = [float(s) for s in axis_str.split(' ')]
                        rotomatrix = OrthoMatrix([0, 0, 0], axis)
                        is_dof = new_text(
                            go_to(go_to(go_to(self.root, joint_type, 'name', joint), 'TransformAxis', 'name', rotation),
                                  'coordinates'))
                        if is_dof in dof_of_joint(joint, joint_type):
                            is_dof = 'True'
                        else:
                            is_dof = 'None'
                        printing_segment(body, body + '_' + rotation, parent, rotomatrix, 'rotation', is_dof)
                        rotation_for_markers = rotation_for_markers.dot(rotomatrix.get_rotation_matrix())
                        parent = body + '_' + rotation

                # segment to cancel axis effects
                rotomatrix.set_rotation_matrix(inv(rotation_for_markers))
                printing_segment(body, body, parent, rotomatrix, true_segment=True)
                parent = body

            # Markers
            _list_markers = list_markers_body(body)
            if _list_markers is not []:
                self.write('\n    // Markers')
                for marker in _list_markers:
                    position = new_text(go_to(go_to(self.root, 'Marker', 'name', marker), 'location'))
                    self.write('\n    marker    {}'.format(marker))
                    self.write('\n        parent    {}'.format(parent))
                    self.write('\n        position    {}'.format(position))
                    self.write('\n    endmarker\n')
            late_body = body

        # Muscle definition
        self.write('\n// MUSCLE DEFINIION\n')
        sort_muscle = []
        muscle_ref_group = []
        for muscle in muscle_list(self):
            viapoint = list_pathpoint_muscle(muscle)
            bodies_viapoint = []
            for pathpoint in viapoint:
                bodies_viapoint.append(get_body_pathpoint(pathpoint))
            # it is supposed that viapoints are organized in order
            # from the parent body to the child body
            body_start = bodies_viapoint[0]
            body_end = bodies_viapoint[len(bodies_viapoint) - 1]
            sort_muscle.append([body_start, body_end])
            muscle_ref_group.append([muscle, body_start + '_to_' + body_end])
        # selecting muscle group
        group_muscle = []
        for ext_muscle in sort_muscle:
            if ext_muscle not in group_muscle:
                group_muscle.append(ext_muscle)
                # print muscle group
        for muscle_group in group_muscle:
            self.write('\n// {} > {}\n'.format(muscle_group[0], muscle_group[1]))
            self.write('musclegroup {}\n'.format(muscle_group[0] + '_to_' + muscle_group[1]))
            self.write('    OriginParent        {}\n'.format(muscle_group[0]))
            self.write('    InsertionParent        {}\n'.format(muscle_group[1]))
            self.write('endmusclegroup\n')
            # muscle
            for muscle in muscle_list(self):
                # muscle data
                m_ref = muscle_group_reference(muscle, muscle_ref_group)
                if m_ref == muscle_group[0] + '_to_' + muscle_group[1]:
                    muscle_type = 'hillthelen'
                    state_type = 'buchanan'
                    list_pathpoint = list_pathpoint_muscle(muscle)
                    start_point = list_pathpoint.pop(0)
                    end_point = list_pathpoint.pop()
                    start_pos = get_pos(start_point)
                    insert_pos = get_pos(end_point)
                    opt_length = new_text(
                        go_to(go_to(self.root, 'Thelen2003Muscle', 'name', muscle), 'optimal_fiber_length'))
                    max_force = new_text(
                        go_to(go_to(self.root, 'Thelen2003Muscle', 'name', muscle), 'max_isometric_force'))
                    tendon_slack_length = new_text(
                        go_to(go_to(self.root, 'Thelen2003Muscle', 'name', muscle), 'tendon_slack_length'))
                    pennation_angle = new_text(
                        go_to(go_to(self.root, 'Thelen2003Muscle', 'name', muscle), 'pennation_angle_at_optimal'))
                    pcsa = new_text(go_to(go_to(self.root, 'Thelen2003Muscle', 'name', muscle), 'pcsa'))
                    max_velocity = new_text(
                        go_to(go_to(self.root, 'Thelen2003Muscle', 'name', muscle), 'max_contraction_velocity'))

                    # print muscle data
                    self.write('\n    muscle    {}'.format(muscle))
                    self.write('\n        Type    {}'.format(muscle_type)) if muscle_type != 'None' else self.write('')
                    self.write('\n        statetype    {}'.format(state_type)) if state_type != 'None' else self.write(
                        '')
                    self.write('\n        musclegroup    {}'.format(m_ref)) if m_ref != 'None' else self.write('')
                    self.write(
                        '\n        OriginPosition    {}'.format(start_pos)) if start_pos != 'None' else self.write('')
                    self.write(
                        '\n        InsertionPosition    {}'.format(insert_pos)) if insert_pos != 'None' else self.write(
                        '')
                    self.write(
                        '\n        optimalLength    {}'.format(opt_length)) if opt_length != 'None' else self.write('')
                    self.write('\n        maximalForce    {}'.format(max_force)) if max_force != 'None' else self.write(
                        '')
                    self.write('\n        tendonSlackLength    {}'.format(
                        tendon_slack_length)) if tendon_slack_length != 'None' else self.write('')
                    self.write('\n        pennationAngle    {}'.format(
                        pennation_angle)) if pennation_angle != 'None' else self.write('')
                    self.write('\n        PCSA    {}'.format(pcsa)) if pcsa != 'None' else self.write('')
                    self.write(
                        '\n        maxVelocity    {}'.format(max_velocity)) if max_velocity != 'None' else self.write(
                        '')
                    self.write('\n    endmuscle\n')
                    # viapoint
                    for viapoint in list_pathpoint:
                        # viapoint data
                        parent_viapoint = get_body_pathpoint(viapoint)
                        viapoint_pos = get_pos(viapoint)
                        # print viapoint data
                        self.write('\n        viapoint    {}'.format(viapoint))
                        self.write('\n            parent    {}'.format(
                            parent_viapoint)) if parent_viapoint != 'None' else self.write('')
                        self.write('\n            muscle    {}'.format(muscle))
                        self.write('\n            musclegroup    {}'.format(m_ref)) if m_ref != 'None' else self.write(
                            '')
                        self.write('\n            position    {}'.format(
                            viapoint_pos)) if viapoint_pos != 'None' else self.write('')
                        self.write('\n        endviapoint')
                    self.write('\n')

        self.file.close()

    def __getattr__(self, attr):
        print('Error : {} is not an attribute of this class'.format(attr))

    def get_path(self):
        return self.path

    def write(self, string):
        self.file = open(self.path, 'a')
        self.file.write(string)
        self.file.close()

    def get_origin_file(self):
        return self.originfile

    def credits(self):
        return self.data_origin.xpath(
            '/OpenSimDocument/Model/credits')[0].text

    def publications(self):
        return self.data_origin.xpath(
            '/OpenSimDocument/Model/publications')[0].text

    def body_list(self):
        _list = []
        for body in self.data_origin.xpath(
                '/OpenSimDocument/Model/BodySet/objects/Body'):
            _list.append(body.get("name"))
        return _list
