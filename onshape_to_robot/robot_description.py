import numpy as np
import os
import re
import math
import uuid
import mujoco
from bs4 import BeautifulSoup
from pathlib import Path
from xml.sax.saxutils import escape
from colorama import Fore, Back, Style
from . import stl_combine
from .simplify_mesh import reduce_faces


def xml_escape(unescaped: str) -> str:
    """Escapes XML characters in a string so that it can be safely added to an XML file
    """
    return escape(unescaped, entities={"'": "&apos;", "\"": "&quot;"})


def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def mat2quat(rmat):
    """
    Converts given rotation matrix to quaternion.

    Args:
        rmat (np.array): 3x3 rotation matrix

    Returns:
        np.array: (x,y,z,w) float quaternion angles
    """
    M = np.asarray(rmat).astype(np.float32)[:3, :3]

    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]
    # symmetric matrix K
    K = np.array([
        [m00 - m11 - m22, np.float32(0.0),
         np.float32(0.0), np.float32(0.0)],
        [m01 + m10, m11 - m00 - m22, np.float32(0.0),
         np.float32(0.0)],
        [m02 + m20, m12 + m21, m22 - m00 - m11,
         np.float32(0.0)],
        [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
    ])
    K /= 3.0
    # quaternion is Eigen vector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(K)
    inds = np.array([3, 0, 1, 2])
    q1 = V[inds, np.argmax(w)]
    if q1[0] < 0.0:
        np.negative(q1, q1)
    inds = np.array([1, 2, 3, 0])
    return q1[inds]


def mujoco_pose(matrix):
    p = '%.20g %.20g %.20g'
    q = '%.20g %.20g %.20g %.20g'
    x = matrix[0, 3]
    y = matrix[1, 3]
    z = matrix[2, 3]
    quat = mat2quat(matrix[:3, :3])
    return p % (x, y, z), q % (quat[0], quat[1], quat[2], quat[3])


def origin(matrix):
    urdf = '<origin xyz="%.20g %.20g %.20g" rpy="%.20g %.20g %.20g" />'
    x = matrix[0, 3]
    y = matrix[1, 3]
    z = matrix[2, 3]
    rpy = rotationMatrixToEulerAngles(matrix)

    return urdf % (x, y, z, rpy[0], rpy[1], rpy[2])


def pose(matrix, frame=''):
    sdf = '<pose>%.20g %.20g %.20g %.20g %.20g %.20g</pose>'
    x = matrix[0, 3]
    y = matrix[1, 3]
    z = matrix[2, 3]
    rpy = rotationMatrixToEulerAngles(matrix)

    if frame != '':
        sdf = '<frame name="' + frame + '_frame">' + sdf + '</frame>'

    return sdf % (x, y, z, rpy[0], rpy[1], rpy[2])


class RobotDescription(object):

    def __init__(self, name):
        self.drawCollisions = False
        self.relative = True
        self.mergeSTLs = 'no'
        self.mergeSTLsCollisions = False
        self.useFixedLinks = False
        self.simplifySTLs = 'no'
        self.maxSTLSize = 3
        self.xml = ''
        self.jointMaxEffort = 1
        self.jointMaxVelocity = 10
        self.noDynamics = False
        self.packageName = ""
        self.addDummyBaseLink = False
        self.robotName = name
        self.meshDir = None

    def shouldMergeSTLs(self, node):
        return self.mergeSTLs == 'all' or self.mergeSTLs == node

    def shouldSimplifySTLs(self, node):
        return self.simplifySTLs == 'all' or self.simplifySTLs == node

    def append(self, str):
        self.xml += str + "\n"

    def jointMaxEffortFor(self, jointName):
        if isinstance(self.jointMaxEffort, dict):
            if jointName in self.jointMaxEffort:
                return self.jointMaxEffort[jointName]
            else:
                return self.jointMaxEffort['default']
        else:
            return self.jointMaxEffort

    def jointMaxVelocityFor(self, jointName):
        if isinstance(self.jointMaxVelocity, dict):
            if jointName in self.jointMaxVelocity:
                return self.jointMaxVelocity[jointName]
            else:
                return self.jointMaxVelocity['default']
        else:
            return self.jointMaxVelocity

    def resetLink(self):
        self._mesh = {'visual': None, 'collision': None}
        self._color = np.array([0., 0., 0.])
        self._color_mass = 0
        self._link_childs = 0
        self._visuals = []
        self._dynamics = []

    def addLinkDynamics(self, matrix, mass, com, inertia):
        # Inertia
        I = np.matrix(np.reshape(inertia[:9], (3, 3)))
        R = matrix[:3, :3]
        # Expressing COM in the link frame
        com = np.array((matrix * np.matrix([com[0], com[1], com[2], 1]).T).T)[0][:3]
        # Expressing inertia in the link frame
        inertia = R * I * R.T

        self._dynamics.append({'mass': mass, 'com': com, 'inertia': inertia})

    def mergeSTL(self, stl, matrix, color, mass, node='visual'):
        if node == 'visual':
            self._color += np.array(color) * mass
            self._color_mass += mass

        m = stl_combine.load_mesh(stl)
        stl_combine.apply_matrix(m, matrix)

        if self._mesh[node] is None:
            self._mesh[node] = m
        else:
            self._mesh[node] = stl_combine.combine_meshes(self._mesh[node], m)

    def linkDynamics(self):
        mass = 0
        com = np.array([0.0] * 3)
        inertia = np.matrix(np.zeros((3, 3)))
        identity = np.matrix(np.eye(3))

        for dynamic in self._dynamics:
            mass += dynamic['mass']
            com += dynamic['com'] * dynamic['mass']

        if mass > 0:
            com /= mass

        # https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=246
        for dynamic in self._dynamics:
            r = dynamic['com'] - com
            p = np.matrix(r)
            inertia += dynamic['inertia'] + \
                (np.dot(r, r)*identity - p.T*p)*dynamic['mass']

        return mass, com, inertia

    def export(self, output_dir):
        print("\n" + Style.BRIGHT + "* Writing " + self.ext.upper() + " file" + Style.RESET_ALL)
        with open(output_dir + '/robot.' + self.ext, 'w', encoding="utf-8") as stream:
            stream.write(self.xml)


class RobotURDF(RobotDescription):

    def __init__(self, name):
        super().__init__(name)
        self.ext = 'urdf'
        self.append('<robot name="' + self.robotName + '">')
        pass

    def addDummyLink(self, name, visualMatrix=None, visualSTL=None, visualColor=None):
        self.append('<link name="' + name + '">')
        self.append('<inertial>')
        self.append('<origin xyz="0 0 0" rpy="0 0 0" />')
        # XXX: We use a low mass because PyBullet consider mass 0 as world fixed
        if self.noDynamics:
            self.append('<mass value="0" />')
        else:
            self.append('<mass value="1e-9" />')
        self.append('<inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0" />')
        self.append('</inertial>')
        if visualSTL is not None:
            self.addSTL(visualMatrix, visualSTL, visualColor, name + "_visual", 'visual')
        self.append('</link>')

    def addDummyBaseLinkMethod(self, name):
        # adds a dummy base_link for ROS users
        self.append('<link name="base_link"></link>')
        self.append('<joint name="base_link_to_base" type="fixed">')
        self.append('<parent link="base_link"/>')
        self.append('<child link="' + name + '" />')
        self.append('<origin rpy="0.0 0 0" xyz="0 0 0"/>')
        self.append('</joint>')

    def addFixedJoint(self, parent, child, matrix, name=None):
        if name is None:
            name = parent + '_' + child + '_fixing'

        self.append('<joint name="' + name + '" type="fixed">')
        self.append(origin(matrix))
        self.append('<parent link="' + parent + '" />')
        self.append('<child link="' + child + '" />')
        self.append('<axis xyz="0 0 0"/>')
        self.append('</joint>')
        self.append('')

    def startLink(self, name, matrix):
        self._link_name = name
        self.resetLink()

        if self.addDummyBaseLink:
            self.addDummyBaseLinkMethod(name)
            self.addDummyBaseLink = False
        self.append('<link name="' + name + '">')

    def endLink(self):
        mass, com, inertia = self.linkDynamics()

        for node in ['visual', 'collision']:
            if self._mesh[node] is not None:
                if node == 'visual' and self._color_mass > 0:
                    color = self._color / self._color_mass
                else:
                    color = [0.5, 0.5, 0.5]

                filename = self._link_name + '_' + node + '.stl'
                stl_combine.save_mesh(self._mesh[node], self.meshDir + '/' + filename)
                reduce_faces(self.meshDir + '/' + filename)
                if self.shouldSimplifySTLs(node):
                    stl_combine.simplify_stl(self.meshDir + '/' + filename, self.maxSTLSize)
                self.addSTL(np.identity(4), filename, color, self._link_name, node)

        self.append('<inertial>')
        self.append('<origin xyz="%.20g %.20g %.20g" rpy="0 0 0"/>' % (com[0], com[1], com[2]))
        self.append('<mass value="%.20g" />' % mass)
        self.append(
            '<inertia ixx="%.20g" ixy="%.20g"  ixz="%.20g" iyy="%.20g" iyz="%.20g" izz="%.20g" />' %
            (inertia[0, 0], inertia[0, 1], inertia[0, 2], inertia[1, 1], inertia[1, 2], inertia[2,
                                                                                                2]))
        self.append('</inertial>')

        if self.useFixedLinks:
            self.append('<visual><geometry><box size="0 0 0" /></geometry></visual>')

        self.append('</link>')
        self.append('')

        if self.useFixedLinks:
            n = 0
            for visual in self._visuals:
                n += 1
                visual_name = '%s_%d' % (self._link_name, n)
                self.addDummyLink(visual_name, visual[0], visual[1], visual[2])
                self.addJoint('fixed', self._link_name, visual_name, np.eye(4),
                              visual_name + '_fixing', None)

    def addFrame(self, name, matrix):
        # Adding a dummy link
        self.addDummyLink(name)

        # Linking it with last link with a fixed link
        self.addFixedJoint(self._link_name, name, matrix, name + '_frame')

    def addSTL(self, matrix, stl, color, name, node='visual'):
        stl_file = self.packageName.strip("/") + "/" + stl
        stl_file = xml_escape(stl_file)

        material_name = name + "_material"
        material_name = xml_escape(material_name)

        self.append('<' + node + '>')
        self.append(origin(matrix))
        self.append('<geometry>')
        self.append(f'<mesh filename="package://{stl_file}"/>')
        self.append('</geometry>')
        if node == 'visual':
            self.append(f'<material name="{material_name}">')
            self.append('<color rgba="%.20g %.20g %.20g 1.0"/>' % (color[0], color[1], color[2]))
            self.append('</material>')
        self.append('</' + node + '>')

    def addPart(self, matrix, stl, mass, com, inertia, color, shapes=None, name=''):
        if stl is not None:
            if not self.drawCollisions:
                if self.useFixedLinks:
                    self._visuals.append([matrix, self.packageName + os.path.basename(stl), color])
                elif self.shouldMergeSTLs('visual'):
                    self.mergeSTL(stl, matrix, color, mass)
                else:
                    self.addSTL(matrix, os.path.basename(stl), color, name, 'visual')

            entries = ['collision']
            if self.drawCollisions:
                entries.append('visual')
            for entry in entries:

                if shapes is None:
                    # We don't have pure shape, we use the mesh
                    if self.shouldMergeSTLs(entry):
                        self.mergeSTL(stl, matrix, color, mass, entry)
                    else:
                        self.addSTL(matrix, os.path.basename(stl), color, name, entry)
                else:
                    # Inserting pure shapes in the URDF model
                    self.append('<!-- Shapes for ' + name + ' -->')
                    for shape in shapes:
                        self.append('<' + entry + '>')
                        self.append(origin(matrix * shape['transform']))
                        self.append('<geometry>')
                        if shape['type'] == 'cube':
                            self.append('<box size="%.20g %.20g %.20g" />' %
                                        tuple(shape['parameters']))
                        if shape['type'] == 'cylinder':
                            self.append('<cylinder length="%.20g" radius="%.20g" />' %
                                        tuple(shape['parameters']))
                        if shape['type'] == 'sphere':
                            self.append('<sphere radius="%.20g" />' % shape['parameters'])
                        self.append('</geometry>')

                        if entry == 'visual':
                            self.append('<material name="' + name + '_material">')
                            self.append('<color rgba="%.20g %.20g %.20g 1.0"/>' %
                                        (color[0], color[1], color[2]))
                            self.append('</material>')
                        self.append('</' + entry + '>')

        self.addLinkDynamics(matrix, mass, com, inertia)

    def addJoint(self, jointType, linkFrom, linkTo, transform, name, jointLimits, zAxis=[0, 0, 1]):
        self.append('<joint name="' + name + '" type="' + jointType + '">')
        self.append(origin(transform))
        self.append('<parent link="' + linkFrom + '" />')
        self.append('<child link="' + linkTo + '" />')
        self.append('<axis xyz="%.20g %.20g %.20g"/>' % tuple(zAxis))
        lowerUpperLimits = ''
        if jointLimits is not None:
            lowerUpperLimits = 'lower="%.20g" upper="%.20g"' % jointLimits
        self.append(
            '<limit effort="%.20g" velocity="%.20g" %s/>' %
            (self.jointMaxEffortFor(name), self.jointMaxVelocityFor(name), lowerUpperLimits))
        self.append('<joint_properties friction="0.0"/>')
        self.append('</joint>')
        self.append('')

    def finalize(self):
        self.append(self.additionalXML)
        self.append('</robot>')


class RobotSDF(RobotDescription):

    def __init__(self, name):
        super().__init__(name)
        self.ext = 'sdf'
        self.relative = False
        self.append('<sdf version="1.6">')
        self.append('<model name="' + self.robotName + '">')
        pass

    def addFixedJoint(self, parent, child, matrix, name=None):
        if name is None:
            name = parent + '_' + child + '_fixing'

        self.append('<joint name="' + name + '" type="fixed">')
        self.append(pose(matrix))
        self.append('<parent>' + parent + '</parent>')
        self.append('<child>' + child + '</child>')
        self.append('</joint>')
        self.append('')

    def addDummyLink(self, name, visualMatrix=None, visualSTL=None, visualColor=None):
        self.append('<link name="' + name + '">')
        self.append('<pose>0 0 0 0 0 0</pose>')
        self.append('<inertial>')
        self.append('<pose>0 0 0 0 0 0</pose>')
        self.append('<mass>1e-9</mass>')
        self.append('<inertia>')
        self.append('<ixx>0</ixx><ixy>0</ixy><ixz>0</ixz><iyy>0</iyy><iyz>0</iyz><izz>0</izz>')
        self.append('</inertia>')
        self.append('</inertial>')
        if visualSTL is not None:
            self.addSTL(visualMatrix, visualSTL, visualColor, name + "_visual", "visual")
        self.append('</link>')

    def startLink(self, name, matrix):
        self._link_name = name
        self.resetLink()
        self.append('<link name="' + name + '">')
        self.append(pose(matrix, name))

    def endLink(self):
        mass, com, inertia = self.linkDynamics()

        for node in ['visual', 'collision']:
            if self._mesh[node] is not None:
                color = self._color / self._color_mass
                filename = self._link_name + '_' + node + '.stl'
                stl_combine.save_mesh(self._mesh[node], self.meshDir + '/' + filename)
                if self.shouldSimplifySTLs(node):
                    stl_combine.simplify_stl(self.meshDir + '/' + filename, self.maxSTLSize)
                self.addSTL(np.identity(4), filename, color, self._link_name, 'visual')

        self.append('<inertial>')
        self.append('<pose frame="' + self._link_name + '_frame">%.20g %.20g %.20g 0 0 0</pose>' %
                    (com[0], com[1], com[2]))
        self.append('<mass>%.20g</mass>' % mass)
        self.append(
            '<inertia><ixx>%.20g</ixx><ixy>%.20g</ixy><ixz>%.20g</ixz><iyy>%.20g</iyy><iyz>%.20g</iyz><izz>%.20g</izz></inertia>'
            % (inertia[0, 0], inertia[0, 1], inertia[0, 2], inertia[1, 1], inertia[1, 2],
               inertia[2, 2]))
        self.append('</inertial>')

        if self.useFixedLinks:
            self.append('<visual><geometry><box><size>0 0 0</size></box></geometry></visual>')

        self.append('</link>')
        self.append('')

        if self.useFixedLinks:
            n = 0
            for visual in self._visuals:
                n += 1
                visual_name = '%s_%d' % (self._link_name, n)
                self.addDummyLink(visual_name, visual[0], visual[1], visual[2])
                self.addJoint('fixed', self._link_name, visual_name, np.eye(4),
                              visual_name + '_fixing', None)

    def addFrame(self, name, matrix):
        # Adding a dummy link
        self.addDummyLink(name)

        # Linking it with last link with a fixed link
        self.addFixedJoint(self._link_name, name, matrix, name + '_frame')

    def material(self, color):
        m = '<material>'
        m += '<ambient>%.20g %.20g %.20g 1</ambient>' % (color[0], color[1], color[2])
        m += '<diffuse>%.20g %.20g %.20g 1</diffuse>' % (color[0], color[1], color[2])
        m += '<specular>0.1 0.1 0.1 1</specular>'
        m += '<emissive>0 0 0 0</emissive>'
        m += '</material>'

        return m

    def addSTL(self, matrix, stl, color, name, node='visual'):
        self.append('<' + node + ' name="' + name + '_visual">')
        self.append(pose(matrix))
        self.append('<geometry>')
        self.append('<mesh><uri>file://' + stl + '</uri></mesh>')
        self.append('</geometry>')
        if node == 'visual':
            self.append(self.material(color))
        self.append('</' + node + '>')

    def addPart(self, matrix, stl, mass, com, inertia, color, shapes=None, name=''):
        name = self._link_name + '_' + str(self._link_childs) + '_' + name
        self._link_childs += 1

        # self.append('<link name="'+name+'">')
        # self.append(pose(matrix))

        if stl is not None:
            if not self.drawCollisions:
                if self.useFixedLinks:
                    self._visuals.append([matrix, self.packageName + os.path.basename(stl), color])
                elif self.shouldMergeSTLs('visual'):
                    self.mergeSTL(stl, matrix, color, mass)
                else:
                    self.addSTL(matrix, os.path.basename(stl), color, name, 'visual')

            entries = ['collision']
            if self.drawCollisions:
                entries.append('visual')
            for entry in entries:
                if shapes is None:
                    # We don't have pure shape, we use the mesh
                    if self.shouldMergeSTLs(entry):
                        self.mergeSTL(stl, matrix, color, mass, entry)
                    else:
                        self.addSTL(matrix, stl, color, name, entry)
                else:
                    # Inserting pure shapes in the URDF model
                    k = 0
                    self.append('<!-- Shapes for ' + name + ' -->')
                    for shape in shapes:
                        k += 1
                        self.append('<' + entry + ' name="' + name + '_' + entry + '_' + str(k) +
                                    '">')
                        self.append(pose(matrix * shape['transform']))
                        self.append('<geometry>')
                        if shape['type'] == 'cube':
                            self.append('<box><size>%.20g %.20g %.20g</size></box>' %
                                        tuple(shape['parameters']))
                        if shape['type'] == 'cylinder':
                            self.append(
                                '<cylinder><length>%.20g</length><radius>%.20g</radius></cylinder>'
                                % tuple(shape['parameters']))
                        if shape['type'] == 'sphere':
                            self.append('<sphere><radius>%.20g</radius></sphere>' %
                                        shape['parameters'])
                        self.append('</geometry>')

                        if entry == 'visual':
                            self.append(self.material(color))
                        self.append('</' + entry + '>')

        self.addLinkDynamics(matrix, mass, com, inertia)

    def addJoint(self, jointType, linkFrom, linkTo, transform, name, jointLimits, zAxis=[0, 0, 1]):
        self.append('<joint name="' + name + '" type="' + jointType + '">')
        self.append(pose(transform))
        self.append('<parent>' + linkFrom + '</parent>')
        self.append('<child>' + linkTo + '</child>')
        self.append('<axis>')
        self.append('<xyz>%.20g %.20g %.20g</xyz>' % tuple(zAxis))
        lowerUpperLimits = ''
        if jointLimits is not None:
            lowerUpperLimits = '<lower>%.20g</lower><upper>%.20g</upper>' % jointLimits
        self.append(
            '<limit><effort>%.20g</effort><velocity>%.20g</velocity>%s</limit>' %
            (self.jointMaxEffortFor(name), self.jointMaxVelocityFor(name), lowerUpperLimits))
        self.append('</axis>')
        self.append('</joint>')
        self.append('')
        # print('Joint from: '+linkFrom+' to: '+linkTo+', transform: '+str(transform))

    def finalize(self):
        self.append(self.additionalXML)
        self.append('</model>')
        self.append('</sdf>')


class RobotMujocoXML(RobotURDF):

    def __init__(self, name):
        super().__init__(name)

    def export(self, output_dir):
        super().export(output_dir)
        output_dir = Path(output_dir).absolute()
        print((output_dir / f'robot.{self.ext}').as_posix())
        mj_model = mujoco.MjModel.from_xml_path((output_dir / f'robot.{self.ext}').as_posix())
        DOF = mj_model.nv
        for i in range(mj_model.nv):
            j = mj_model.joint(mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i))
            j.frictionloss = 0.1
            j.damping = 1.5
        mujoco.mj_saveLastXML((output_dir / "robot.xml").as_posix(), mj_model)

        with open((output_dir / "robot.urdf").as_posix(), "r") as f:
            urdfsoup = BeautifulSoup(f.read(), "xml")
        with open((output_dir / "robot.xml").as_posix(), "r") as f:
            xmlsoup = BeautifulSoup(f.read(), "xml")
        model = xmlsoup.find('mujoco')
        default_tag = xmlsoup.find('default')
        if not default_tag:

            default_tag = xmlsoup.new_tag('default')
        geom_default = xmlsoup.new_tag('geom',
                                       attrs=dict(contype="0",
                                                  conaffinity="0",
                                                  group="2",
                                                  type="mesh"))
        act_default = xmlsoup.new_tag('general',
                                      attrs=dict(dyntype="none",
                                                 biastype="affine",
                                                 ctrlrange="-2.8 2.8",
                                                 forcerange="-87 87",
                                                 gainprm="4500",
                                                 biasprm="0 -4500 -450"))

        default_tag.append(geom_default)
        default_tag.append(act_default)
        # Create the asset tag
        asset_tag = xmlsoup.new_tag('asset')

        # Create the texture tag and its attributes
        texture_tag = xmlsoup.new_tag('texture',
                                      attrs={
                                          'name': 'grid',
                                          'type': '2d',
                                          'builtin': 'checker',
                                          'rgb1': '.2 .3 .4',
                                          'rgb2': '.1 0.15 0.2',
                                          'width': '512',
                                          'height': '512',
                                      })

        # Create the material tag and its attributes
        material_tag = xmlsoup.new_tag('material',
                                       attrs={
                                           'name': 'grid',
                                           'texture': 'grid',
                                           'texrepeat': '1 1',
                                           'texuniform': 'true'
                                       })

        # Append the texture and material tags to the asset tag
        asset_tag.append(texture_tag)
        asset_tag.append(material_tag)

        def find_joints_containing_name(soup, name_substring):
            return soup.find_all(
                'joint', attrs={'name': lambda value: name_substring in value if value else False})

        for joint in find_joints_containing_name(urdfsoup, '_frame'):
            parent_tag = joint.find('parent')
            child_tag = joint.find('child')
            origin_tag = joint.find('origin')
            parent_body_tag = xmlsoup.find('body', attrs={'name': parent_tag['link']})
            new_frame_tag = xmlsoup.new_tag('site',
                                            attrs=dict(name=child_tag['link'],
                                                       size="0.01",
                                                       pos=origin_tag['xyz'],
                                                       euler=origin_tag['rpy'],
                                                       rgba="1 0 0 1",
                                                       group="1"))
            parent_body_tag.append(new_frame_tag)

        act_tag = xmlsoup.new_tag('actuator')
        for i in range(mj_model.nv):
            joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            general_tag = xmlsoup.new_tag('general',
                                          attrs=dict(name=joint_name + "_act", joint=joint_name))
            act_tag.append(general_tag)
        compiler_tag = xmlsoup.find('compiler')
        if not compiler_tag:
            compiler_tag = xmlsoup.new_tag('compiler')
        compiler_tag['angle'] = "radian"
        compiler_tag['autolimits'] = "true"
        compiler_tag['balanceinertia'] = "true"
        compiler_tag['meshdir'] = "."

        world_tag = xmlsoup.find('worldbody')
        target_body_tag = xmlsoup.new_tag('body',
                                          attrs=dict(name="target",
                                                     pos="0 0 0.5",
                                                     quat="1 0 0 0",
                                                     mocap="true"))
        target_body_geom_tag = xmlsoup.new_tag('geom',
                                               attrs=dict(type="box",
                                                          size=".05 .05 .05",
                                                          rgba=".6 .3 .3 .5"))
        target_body_geom_site = xmlsoup.new_tag('site',
                                                attrs=dict(name="target",
                                                           type="sphere",
                                                           size="0.01",
                                                           rgba="0 0 1 1"))
        target_body_tag.append(target_body_geom_tag)
        target_body_tag.append(target_body_geom_site)
        world_tag.append(target_body_tag)

        light_tag = xmlsoup.new_tag('light', attrs=dict(name="top", pos="0 0 1", mode="trackcom"))
        floor_tag = xmlsoup.new_tag('geom',
                                    attrs=dict(name="floor",
                                               size="1 1 0.01",
                                               type="plane",
                                               material="grid"))
        world_tag.append(floor_tag)
        world_tag.append(light_tag)

        keyframe_tag = xmlsoup.new_tag('keyframe')
        homekey_tag = xmlsoup.new_tag('key',
                                      attrs=dict(name="home",
                                                 qpos=" ".join(['0'] * DOF),
                                                 ctrl=" ".join(['0'] * DOF)))
        keyframe_tag.append(homekey_tag)

        option_tag = xmlsoup.find('option')
        if not option_tag:
            option_tag = xmlsoup.new_tag('option')
        option_tag['integrator'] = "implicitfast"
        option_tag['impratio'] = 10

        model.insert(0, keyframe_tag)
        model.insert(0, act_tag)
        model.insert(0, asset_tag)
        model.insert(0, default_tag)
        model.insert(0, option_tag)
        model.insert(0, compiler_tag)
        with open((output_dir / "robot.xml").as_posix(), "w") as f:
            f.write(xmlsoup.prettify())
