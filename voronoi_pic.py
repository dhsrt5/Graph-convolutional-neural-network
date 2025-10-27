from abaqus import *
from abaqusConstants import *
from scipy.spatial import *
import numpy as np
import random
from scipy.io import loadmat
import os
from abaqus import*
from part import*
from material import*
from section import*
from assembly import*
from step import*
from interaction import*
from load import*
from mesh import*
from job import*
from sketch import*
from visualization import*
from string import*
from math import*
import section
import regionToolset
import displayGroupMdbToolset as dgm
import part
import material
import assembly
import step
import interaction
import load
import mesh
import math
import job
import sketch
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior
import __main__
import math
global pi
import re
import regionToolset
pi=acos(-1.0)

os.chdir(r"H:\voronoi_0522")
Work_place = "H:\voronoi_0522"
f = np.loadtxt('idx.txt')
IDX = int(f)
openMdb(pathName='vor1.cae')

vornb = loadmat(r'H:\voronoi_0522\data\stru_'+str(int(IDX))+r'\vornb.mat')
K = loadmat(r'H:\voronoi_0522\data\stru_'+str(int(IDX))+r'\K.mat')
points = loadmat(r'H:\voronoi_0522\data\stru_'+str(int(IDX))+r'\points.mat')
vorvx = loadmat(r'H:\voronoi_0522\data\stru_'+str(int(IDX))+r'\vorvx.mat')
V = loadmat(r'H:\voronoi_0522\data\stru_'+str(int(IDX))+r'\V.mat')

V=V['V']
vorvx=vorvx['vorvx']
points=points['pos']
K=K['K1']
vornb=vornb['vornb']
Random_matrix = np.loadtxt(r'H:\voronoi_0522\data\stru_'+str(int(IDX))+r'\material.txt')
myModel = mdb.models['Model-1']

#####material-1  NI
mdb.models['Model-1'].Material(name='NI')
mdb.models['Model-1'].materials['NI'].Density(table=((8.902e-09, ), ))
mdb.models['Model-1'].materials['NI'].Elastic(temperatureDependency=ON, table=(
    (207000.0, 0.31, 20.0), (160000.0, 0.31, 1000.0)))
mdb.models['Model-1'].materials['NI'].Conductivity(table=((82.9, ), ))
mdb.models['Model-1'].materials['NI'].SpecificHeat(table=((471000000.0, ), ))
mdb.models['Model-1'].materials['NI'].Expansion(table=((1.33e-05, ), ))
mdb.models['Model-1'].HomogeneousSolidSection(name='Section-1', material='NI', 
    thickness=None)

#####material-1  CU
mdb.models['Model-1'].Material(name='CU')
mdb.models['Model-1'].materials['CU'].Density(table=((8.93E-9, ), ))
mdb.models['Model-1'].materials['CU'].Elastic(temperatureDependency=ON, table=(
    (128000.0, 0.35, 20.0), (90000.0, 0.35, 1000.0)))
mdb.models['Model-1'].materials['CU'].Conductivity(table=((398, ), ))
mdb.models['Model-1'].materials['CU'].SpecificHeat(table=((386000000.0, ), ))
mdb.models['Model-1'].materials['CU'].Expansion(table=((1.67e-05, ), ))
mdb.models['Model-1'].HomogeneousSolidSection(name='Section-2', material='CU', 
    thickness=None)


#####material-1  AL
mdb.models['Model-1'].Material(name='AL')
mdb.models['Model-1'].materials['AL'].Density(table=((2.97E-9, ), ))
mdb.models['Model-1'].materials['AL'].Elastic(temperatureDependency=ON, table=(
    (70000.0, 0.3, 20.0), (55000.0, 0.3, 1000.0)))
mdb.models['Model-1'].materials['AL'].Conductivity(table=((205, ), ))
mdb.models['Model-1'].materials['AL'].SpecificHeat(table=((896000000.0, ), ))
mdb.models['Model-1'].materials['AL'].Expansion(table=((2.31e-05, ), ))
mdb.models['Model-1'].HomogeneousSolidSection(name='Section-3', material='AL', 
    thickness=None)
    
#============================================================================================
for i in range(len(vorvx[0])):
    face_points=[]
    for j in range(len(K[0][i])):
        face_points.append(vorvx[0][i][K[0][i][j]-1])
    
    ###voronoi JIANMO
    for ii in range(len(face_points)):
        myPart = myModel.Part(name='Part-vor'+str(ii), dimensionality=THREE_D, type=DEFORMABLE_BODY)
        j=0
        points_list = [(face_points[ii][j], face_points[ii][(j + 1) % len(face_points[ii])]) for j in range(len(face_points[ii]))]
        wire = myPart.WirePolyLine(points=points_list, mergeType=SEPARATE, meshable=ON)
        face_edge = myPart.getFeatureEdges(name=wire.name)
        myPart.CoverEdges(edgeList = face_edge, tryAnalytical=True)
    
    #Assembly
    a = mdb.models['Model-1'].rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    for num1 in range(len(face_points)):
        p = mdb.models['Model-1'].parts['Part-vor'+str(num1)]
        a.Instance(name='Part-vor'+str(num1)+'-1', part=p, dependent=ON)
    
    #merge
    prefix = 'Part-vor'
    suffix = '-1'
    instance_name=[]
    for num2 in range(len(face_points)):
        instance_name.append(prefix+str(num2)+suffix)
    
    instances=tuple(a.instances[name] for name in instance_name)
    a.InstanceFromBooleanMerge(name='voronoi_3D'+str(int(i)), instances=instances, originalInstances=DELETE, domain=GEOMETRY)
    
    #CELLS
    p = mdb.models['Model-1'].parts['voronoi_3D'+str(int(i))]
    faces = p.faces[:]
    p.AddCells(faceList = faces)
    #========================================================================================
    pattern = re.compile(r'^voronoi_3D\d+$')
    for key in list(myModel.parts.keys()):
        if not pattern.match(key):
            del myModel.parts[key]

for jj in range(len(Random_matrix)):
    p = mdb.models['Model-1'].parts['voronoi_3D'+str(int(jj))]
    cells = p.cells.findAt(((points[jj][0], points[jj][1], points[jj][2]), ))
    region = p.Set(cells=cells, name='Set-1')
    p.SectionAssignment(region=region, sectionName='Section-'+str(int(Random_matrix[jj])), offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)

a.regenerate()
a = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].enableMultipleColors()
session.viewports['Viewport: 1'].setColor(initialColor='#BDBDBD')
cmap=session.viewports['Viewport: 1'].colorMappings['Section']
session.viewports['Viewport: 1'].setColor(colorMapping=cmap)
session.viewports['Viewport: 1'].disableMultipleColors()
session.viewports['Viewport: 1'].enableMultipleColors()
session.viewports['Viewport: 1'].setColor(initialColor='#BDBDBD')
cmap = session.viewports['Viewport: 1'].colorMappings['Section']
cmap.updateOverrides(overrides={'Section-1':(True, '#FF0000', 'Default', 
    '#FF0000'), 'Section-2':(True, '#0000FF', 'Default', '#0000FF'), 
    'Section-3': (True, '#00FF8C', 'Default', '#00FF8C')})
session.viewports['Viewport: 1'].setColor(colorMapping=cmap)
session.viewports['Viewport: 1'].disableMultipleColors()
session.viewports['Viewport: 1'].enableMultipleColors()
session.viewports['Viewport: 1'].setColor(initialColor='#BDBDBD')
cmap = session.viewports['Viewport: 1'].colorMappings['Section']
session.viewports['Viewport: 1'].setColor(colorMapping=cmap)
session.viewports['Viewport: 1'].disableMultipleColors()
session.viewports['Viewport: 1'].view.setValues(session.views['Top'])
session.viewports['Viewport: 1'].view.setProjection(projection=PARALLEL)
session.viewports['Viewport: 1'].assemblyDisplay.geometryOptions.setValues(
    datumPoints=OFF, datumAxes=OFF, datumPlanes=OFF, datumCoordSystems=OFF, 
    referencePointLabels=OFF, referencePointSymbols=OFF)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    activeCutName='Y-Plane', viewCut=ON)
session.viewports['Viewport: 1'].assemblyDisplay.viewCuts['Y-Plane'].setValues(
    position=0.99)
session.printToFile(fileName='H:/voronoi_0522/data/stru_'+str(int(IDX))+r'/1', format=PNG, 
    canvasObjects=(session.viewports['Viewport: 1'], ))
session.viewports['Viewport: 1'].assemblyDisplay.viewCuts['Y-Plane'].setValues(
    position=0.75)
session.printToFile(fileName='H:/voronoi_0522/data/stru_'+str(int(IDX))+r'/2', format=PNG, 
    canvasObjects=(session.viewports['Viewport: 1'], ))
session.viewports['Viewport: 1'].assemblyDisplay.viewCuts['Y-Plane'].setValues(
    position=0.5)
session.printToFile(fileName='H:/voronoi_0522/data/stru_'+str(int(IDX))+r'/3', format=PNG, 
    canvasObjects=(session.viewports['Viewport: 1'], ))
session.viewports['Viewport: 1'].assemblyDisplay.viewCuts['Y-Plane'].setValues(
    position=0.25)
session.printToFile(fileName='H:/voronoi_0522/data/stru_'+str(int(IDX))+r'/4', format=PNG, 
    canvasObjects=(session.viewports['Viewport: 1'], ))
session.viewports['Viewport: 1'].assemblyDisplay.viewCuts['Y-Plane'].setValues(
    position=0.01)
session.printToFile(fileName='H:/voronoi_0522/data/stru_'+str(int(IDX))+r'/5', format=PNG, 
    canvasObjects=(session.viewports['Viewport: 1'], ))