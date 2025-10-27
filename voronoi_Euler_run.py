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

def euler_to_rot(phi1, Phi, phi2):
    c1 = np.cos(phi1)
    s1 = np.sin(phi1)
    c2 = np.cos(Phi)
    s2 = np.sin(Phi)
    c3 = np.cos(phi2)
    s3 = np.sin(phi2)
    
    R = np.array([
        [c1*c3 - s1*s3*c2,  s1*c3 + c1*s3*c2,  s3*s2],
        [-c1*s3 - s1*c3*c2, -s1*s3 + c1*c3*c2, c3*s2],
        [s1*s2,             -c1*s2,            c2]
    ])
    return R

def get_M(R):
    M = np.zeros((6,6))
    
    M[0,0] = R[0,0]**2
    M[0,1] = R[0,1]**2
    M[0,2] = R[0,2]**2
    M[0,3] = 2 * R[0,0]*R[0,1]
    M[0,4] = 2 * R[0,0]*R[0,2]
    M[0,5] = 2 * R[0,1]*R[0,2]
    
    M[1,0] = R[1,0]**2
    M[1,1] = R[1,1]**2
    M[1,2] = R[1,2]**2
    M[1,3] = 2 * R[1,0]*R[1,1]
    M[1,4] = 2 * R[1,0]*R[1,2]
    M[1,5] = 2 * R[1,1]*R[1,2]
    
    M[2,0] = R[2,0]**2
    M[2,1] = R[2,1]**2
    M[2,2] = R[2,2]**2
    M[2,3] = 2 * R[2,0]*R[2,1]
    M[2,4] = 2 * R[2,0]*R[2,2]
    M[2,5] = 2 * R[2,1]*R[2,2]
    
    M[3,0] = R[0,0]*R[1,0]
    M[3,1] = R[0,1]*R[1,1]
    M[3,2] = R[0,2]*R[1,2]
    M[3,3] = R[0,1]*R[1,0] + R[0,0]*R[1,1]
    M[3,4] = R[0,2]*R[1,0] + R[0,0]*R[1,2]
    M[3,5] = R[0,2]*R[1,1] + R[0,1]*R[1,2]
    
    M[4,0] = R[0,0]*R[2,0]
    M[4,1] = R[0,1]*R[2,1]
    M[4,2] = R[0,2]*R[2,2]
    M[4,3] = R[0,1]*R[2,0] + R[0,0]*R[2,1]
    M[4,4] = R[0,2]*R[2,0] + R[0,0]*R[2,2]
    M[4,5] = R[0,2]*R[2,1] + R[0,1]*R[2,2]
    
    M[5,0] = R[1,0]*R[2,0]
    M[5,1] = R[1,1]*R[2,1]
    M[5,2] = R[1,2]*R[2,2]
    M[5,3] = R[1,1]*R[2,0] + R[1,0]*R[2,1]
    M[5,4] = R[1,2]*R[2,0] + R[1,0]*R[2,2]
    M[5,5] = R[1,2]*R[2,1] + R[1,1]*R[2,2]
    
    return M

def cubic_to_anisotropic(C11, C12, C44, phi1_deg, Phi_deg, phi2_deg):
    phi1 = np.radians(phi1_deg)
    Phi = np.radians(Phi_deg)
    phi2 = np.radians(phi2_deg)
    
    R = euler_to_rot(phi1, Phi, phi2)
    
    M = get_M(R)
    
    C_local = np.zeros((6,6))
    indices = [(0,0), (1,1), (2,2), (3,3), (4,4), (5,5)]
    for i,j in indices[:3]:
        C_local[i,j] = C11
    for i,j in [(0,1), (0,2), (1,2)]:
        C_local[i,j] = C12
        C_local[j,i] = C12
    for i,j in indices[3:]:
        C_local[i,j] = C44
    
    # C_global = M @ C_local @ M.T
    C_global = np.dot(np.dot(M, C_local), M.T)
    
    params = []
    for i in range(6):
        for j in range(i, 6):
            params.append(C_global[i,j])
    
    return params[:21]

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
random_matrix = np.random.rand(125, 1)
Random_matrix = np.ceil(3*random_matrix)
myModel = mdb.models['Model-1']

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

#zhuangpei
a = mdb.models['Model-1'].rootAssembly
prefix_delete = 'voronoi_3D'
suffix_delete = '-1'
instance_name=[]
for num5 in range(len(random_matrix)):
    instance_name.append(prefix_delete+str(num5)+suffix_delete)
    a.deleteFeatures(instance_name)

for num6 in range(len(random_matrix)):
    p = mdb.models['Model-1'].parts['voronoi_3D'+str(int(num6))]
    a.Instance(name='voronoi_3D'+str(int(num6))+'-1', part=p, dependent=ON)

# make indenpendent
# a = mdb.models['Model-1'].rootAssembly
# for num7 in range(125):
    # a.makeIndependent(instances=(a.instances['voronoi_3D'+str(int(num7))+'-1'], ))

#merge vornonoi_3d
a1 = mdb.models['Model-1'].rootAssembly
instances_merge = [a1.instances[name] for name in instance_name]
a1.InstanceFromBooleanMerge(name='Voronoi_3D', instances=instances_merge, keepIntersections=ON, 
    originalInstances=DELETE, domain=GEOMETRY)

#findAt()
session.journalOptions.setValues(replayGeometry=COORDINATE,recoverGeometry= COORDINATE)
p = mdb.models['Model-1'].parts['Voronoi_3D']

for num9 in range(len(p.cells)):
    cellsAtPoint = p.cells.findAt(((points[num9][0], points[num9][1], points[num9][2]),))
    p.Set(cells=cellsAtPoint, name = "Long-{}".format(num9))

# for num4 in range(len(p.cells)):
    ## E11 E22 E44 alphe Euler1 Euler2 Euler3
    # mdb.models['Model-1-stress'].Material(name='Material-'+str(num4))
    # mdb.models['Model-1-stress'].materials['Material-4'].Depvar(n=7)
    # mdb.models['Model-1-stress'].materials['Material-4'].UserMaterial(
        # mechanicalConstants=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0))
    # mdb.models['Model-1'].HomogeneousSolidSection(name='Section-'+str(num4), material='Material-'+str(num4), 
        # thickness=None)
    # region = p.sets['Long-'+str(num4)]
    # p.SectionAssignment(region=region, sectionName='Section-'+str(num4), offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)

#####material-1  NI
mdb.models['Model-1'].Material(name='NI')
mdb.models['Model-1'].materials['NI'].Density(table=((8.902e-09, ), ))
mdb.models['Model-1'].materials['NI'].Elastic(temperatureDependency=ON, table=(
    (207000.0, 0.31, 20.0), (160000.0, 0.31, 500.0)))
mdb.models['Model-1'].materials['NI'].Conductivity(table=((82.9, ), ))
mdb.models['Model-1'].materials['NI'].SpecificHeat(table=((471000000.0, ), ))
mdb.models['Model-1'].materials['NI'].Expansion(table=((1.33e-05, ), ))
mdb.models['Model-1'].HomogeneousSolidSection(name='Section-1', material='NI', 
    thickness=None)

#####material-1  CU
mdb.models['Model-1'].Material(name='CU')
mdb.models['Model-1'].materials['CU'].Density(table=((8.93E-9, ), ))
mdb.models['Model-1'].materials['CU'].Elastic(temperatureDependency=ON, table=(
    (128000.0, 0.35, 20.0), (90000.0, 0.35, 500.0)))
mdb.models['Model-1'].materials['CU'].Conductivity(table=((398, ), ))
mdb.models['Model-1'].materials['CU'].SpecificHeat(table=((386000000.0, ), ))
mdb.models['Model-1'].materials['CU'].Expansion(table=((1.67e-05, ), ))
mdb.models['Model-1'].HomogeneousSolidSection(name='Section-2', material='CU', 
    thickness=None)


#####material-1  AL
mdb.models['Model-1'].Material(name='AL')
mdb.models['Model-1'].materials['AL'].Density(table=((2.97E-9, ), ))
mdb.models['Model-1'].materials['AL'].Elastic(temperatureDependency=ON, table=(
    (70000.0, 0.3, 20.0), (55000.0, 0.3, 500.0)))
mdb.models['Model-1'].materials['AL'].Conductivity(table=((205, ), ))
mdb.models['Model-1'].materials['AL'].SpecificHeat(table=((896000000.0, ), ))
mdb.models['Model-1'].materials['AL'].Expansion(table=((2.31e-05, ), ))
mdb.models['Model-1'].HomogeneousSolidSection(name='Section-3', material='AL', 
    thickness=None)

#section assgin
for num3 in range(len(random_matrix)):
    p = mdb.models['Model-1'].parts['Voronoi_3D']
    region = p.sets['Long-'+str(num3)]
    p.SectionAssignment(region=region, sectionName='Section-'+str(int(Random_matrix[num3][0])), offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)

###STEP
# mdb.models['Model-1'].HeatTransferStep(name='Step-1', previous='Initial', 
    # timePeriod=10.0, maxNumInc=10000, initialInc=0.1, minInc=1e-05, maxInc=1.0, 
    # deltmx=20.0)
    
#steady heat transfer
mdb.models['Model-1'].HeatTransferStep(name='Step-1', previous='Initial', 
    response=STEADY_STATE, maxNumInc=500, initialInc=0.1, mxdem=0.3, 
    amplitude=RAMP)

#assembly regenerate
a = mdb.models['Model-1'].rootAssembly
a.regenerate()

##top
partname='Voronoi_3D'
setname='Voronoi_3DTOP'
p = mdb.models['Model-1'].parts[partname]
tol =1e-6
topfaces_list=[]
topfaces_list.append(p.faces.getByBoundingBox(yMin = 1-tol, yMax = 1))
p.Set(faces= topfaces_list, name=setname)


##bottom
partname='Voronoi_3D'
setname='Voronoi_3Dbottom'
p = mdb.models['Model-1'].parts[partname]
tol =1e-6
topfaces_list=[]
topfaces_list.append(p.faces.getByBoundingBox(yMin = -tol, yMax = tol))
p.Set(faces= topfaces_list, name=setname)


##ASSEM top
partname='Voronoi_3D'
setname='Voronoi_3DTOP'
a = mdb.models['Model-1'].rootAssembly
s1 = a.instances['Voronoi_3D-1'].faces
tol =1e-6
face_top = s1.findAt((0.5, 1, 0.5)).getFacesByFaceAngle(tol)
a.Surface(side1Faces= face_top, name=setname)
a.Set(faces=face_top, name='temp-top')

##ASSEM bottom
partname='Voronoi_3D'
setname='Voronoi_3Dbottom'
a = mdb.models['Model-1'].rootAssembly
s1 = a.instances['Voronoi_3D-1'].faces
tol =1e-6
face_bottom = s1.findAt((0.5, 0, 0.5), ).getFacesByFaceAngle(tol)
a.Surface(side1Faces= face_bottom, name=setname)
a.Set(faces=face_bottom, name='temp-down')

###Predefined = 20
cells1 = a.instances['Voronoi_3D-1'].cells[:]
region = a.Set(vertices=[], edges=[], faces=[], cells=cells1, 
    name='Set-temp')
mdb.models['Model-1'].Temperature(name='Predefined Field-1', 
    createStepName='Initial', region=region, distributionType=UNIFORM, 
    crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(20.0, ))


mdb.models['Model-1'].setValues(absoluteZero=0)###-273.13####

####LOADS   need set[faces]
region = a.sets['temp-top']
mdb.models['Model-1'].TemperatureBC(name='top', createStepName='Step-1', 
    region=region, fixed=OFF, distributionType=UNIFORM, fieldName='', 
    magnitude=500.0, amplitude=UNSET)
    
region = a.sets['temp-down']
mdb.models['Model-1'].TemperatureBC(name='down', createStepName='Step-1', 
    region=region, fixed=OFF, distributionType=UNIFORM, fieldName='', 
    magnitude=20.0, amplitude=UNSET)
# mdb.models['Model-1'].SurfaceHeatFlux(name='Load-1', createStepName='Step-1', 
    # region=region, magnitude=500.0)

# region = a.surfaces['Voronoi_3Dbottom']
# mdb.models['Model-1'].SurfaceHeatFlux(name='Load-2', createStepName='Step-1', 
    # region=region, magnitude=500.0)

###mesh
a = mdb.models['Model-1'].rootAssembly
a.makeIndependent(instances=(a.instances['Voronoi_3D-1'], ))
cells = a.instances['Voronoi_3D-1'].cells[:]
a.setMeshControls(regions=cells, elemShape=TET, technique=FREE)
elemType1 = mesh.ElemType(elemCode=DC3D8, elemLibrary=STANDARD)
elemType2 = mesh.ElemType(elemCode=DC3D6, elemLibrary=STANDARD)
elemType3 = mesh.ElemType(elemCode=DC3D4, elemLibrary=STANDARD)
pickedRegions =(cells, )
a.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2, 
    elemType3))
partInstances =(a.instances['Voronoi_3D-1'], )
a.seedPartInstance(regions=partInstances, size=0.03, deviationFactor=0.1, 
    minSizeFactor=0.5)
a.generateMesh(regions=partInstances)


####job ht
mdb.Job(name='HT', model='Model-1', description='', type=ANALYSIS, atTime=None, 
    waitMinutes=0, waitHours=0, queue=None, memory=90, memoryUnits=PERCENTAGE, 
    getMemoryFromAnalysis=True, explicitPrecision=SINGLE, 
    nodalOutputPrecision=SINGLE, echoPrint=OFF, modelPrint=OFF, 
    contactPrint=OFF, historyPrint=OFF, userSubroutine='', scratch='', 
    resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=12, numDomains=12, 
    numGPUs=0)
mdb.jobs['HT'].submit(consistencyChecking=OFF)
mdb.jobs['HT'].waitForCompletion()






####start stress ananylsis
mdb.Model(name='Model-1-stress', objectToCopy=mdb.models['Model-1'])

#step
del mdb.models['Model-1-stress'].steps['Step-1']
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Initial')
mdb.models['Model-1-stress'].StaticStep(name='Step-1', previous='Initial', 
    maxNumInc=10000, initialInc=0.1)

for jj in range(len(mdb.models['Model-1-stress'].parts['Voronoi_3D'].sectionAssignments)):
    del mdb.models['Model-1-stress'].parts['Voronoi_3D'].sectionAssignments[-1]

######Euler from matlab metx-6.0
random_Euler = np.loadtxt(r'H:\voronoi_0522\data\stru_'+str(IDX)+r'\Euler.txt')

p = mdb.models['Model-1-stress'].parts['Voronoi_3D']
for num4 in range(len(p.cells)):
    ### E11 E22 E44 alphe Euler1 Euler2 Euler3 #########NI CU AL######
    if Random_matrix[num4][0] == 1:
        C11 = 246500
        C12 = 147300
        C44 = 124700
        phi1 = random_Euler[num4][0]
        Phi = random_Euler[num4][1]
        phi2 = random_Euler[num4][2]
        aba_params = cubic_to_anisotropic(C11, C12, C44, phi1, Phi, phi2)
        mdb.models['Model-1-stress'].Material(name='Material-'+str(num4))
        mdb.models['Model-1-stress'].materials['Material-'+str(num4)].Elastic(type=ANISOTROPIC, table=(
            (aba_params[0], aba_params[1], aba_params[6], aba_params[2], aba_params[7], aba_params[11], aba_params[3], aba_params[8], aba_params[12], 
            aba_params[15], aba_params[4], aba_params[9], aba_params[13], aba_params[16], aba_params[18], aba_params[5], aba_params[10], aba_params[14], 
            aba_params[17], aba_params[19], aba_params[20]), ))
        mdb.models['Model-1-stress'].materials['Material-'+str(num4)].Expansion(table=((1.33e-5, ), ))
    elif Random_matrix[num4][0] == 2:
        C11 = 168400
        C12 = 121400
        C44 = 75400
        phi1 = random_Euler[num4][0]
        Phi = random_Euler[num4][1]
        phi2 = random_Euler[num4][2]
        aba_params = cubic_to_anisotropic(C11, C12, C44, phi1, Phi, phi2)
        mdb.models['Model-1-stress'].Material(name='Material-'+str(num4))
        mdb.models['Model-1-stress'].materials['Material-'+str(num4)].Elastic(type=ANISOTROPIC, table=(
            (aba_params[0], aba_params[1], aba_params[6], aba_params[2], aba_params[7], aba_params[11], aba_params[3], aba_params[8], aba_params[12], 
            aba_params[15], aba_params[4], aba_params[9], aba_params[13], aba_params[16], aba_params[18], aba_params[5], aba_params[10], aba_params[14], 
            aba_params[17], aba_params[19], aba_params[20]), ))
        mdb.models['Model-1-stress'].materials['Material-'+str(num4)].Expansion(table=((1.67e-5, ), ))
    elif Random_matrix[num4][0] == 3:
        C11 = 108200
        C12 = 61300
        C44 = 28500
        phi1 = random_Euler[num4][0]
        Phi = random_Euler[num4][1]
        phi2 = random_Euler[num4][2]
        aba_params = cubic_to_anisotropic(C11, C12, C44, phi1, Phi, phi2)
        mdb.models['Model-1-stress'].Material(name='Material-'+str(num4))
        mdb.models['Model-1-stress'].materials['Material-'+str(num4)].Elastic(type=c, table=(
            (aba_params[0], aba_params[1], aba_params[6], aba_params[2], aba_params[7], aba_params[11], aba_params[3], aba_params[8], aba_params[12], 
            aba_params[15], aba_params[4], aba_params[9], aba_params[13], aba_params[16], aba_params[18], aba_params[5], aba_params[10], aba_params[14], 
            aba_params[17], aba_params[19], aba_params[20]), ))
        mdb.models['Model-1-stress'].materials['Material-'+str(num4)].Expansion(table=((1.67e-5, ), ))
    
    ###section assign
    mdb.models['Model-1-stress'].HomogeneousSolidSection(name='Section-'+str(num4), material='Material-'+str(num4), 
        thickness=None)
    region = p.sets['Long-'+str(num4)]
    p.SectionAssignment(region=region, sectionName='Section-'+str(num4), offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)


###FANGXIANG
cells = p.cells[:]
region = regionToolset.Region(cells=cells)
orientation=None
mdb.models['Model-1-stress'].parts['Voronoi_3D'].MaterialOrientation(
    region=region, orientationType=GLOBAL, axis=AXIS_1, 
    additionalRotationType=ROTATION_NONE, localCsys=None, fieldName='', 
    stackDirection=STACK_1)

###coupling and load
a = mdb.models['Model-1-stress'].rootAssembly
region1=a.sets['m_Set-1']
region2=a.instances['Voronoi_3D-1'].sets['Voronoi_3DTOP']
mdb.models['Model-1-stress'].Coupling(name='Constraint-1', controlPoint=region1, 
    surface=region2, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, 
    localCsys=None, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)
#bc1
region = a.instances['Voronoi_3D-1'].sets['Voronoi_3Dbottom']
mdb.models['Model-1-stress'].EncastreBC(name='BC-1', createStepName='Initial', 
    region=region, localCsys=None)

region = a.sets['m_Set-1']
mdb.models['Model-1-stress'].DisplacementBC(name='BC-2', createStepName='Step-1', 
    region=region, u1=0.0, u2=-0.1, u3=0.0, ur1=0.0, ur2=0.0, ur3=0.0, 
    amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='', 
    localCsys=None)

## INCRMENTS NUMBER
odb = openOdb(path='HT.odb')
num_increments = odb.steps['Step-1'].frames[-1].frameId
odb.close()

###field
a = mdb.models['Model-1-stress'].rootAssembly
cells = a.instances['Voronoi_3D-1'].cells[:]
region = a.Set(cells=cells, name='vor-all')
mdb.models['Model-1-stress'].Temperature(name='Predefined Field-1', 
    createStepName='Step-1', distributionType=FROM_FILE, 
    fileName='H:\\voronoi_0522\\HT.odb', beginStep=1, beginIncrement=1, 
    endStep=1, endIncrement=num_increments, interpolate=OFF, absoluteExteriorTolerance=0.0, 
    exteriorTolerance=0.05)


##MESH
a = mdb.models['Model-1-stress'].rootAssembly
cells = a.instances['Voronoi_3D-1'].cells[:]
a.setMeshControls(regions=cells, elemShape=TET, technique=FREE)
elemType1 = mesh.ElemType(elemCode=C3D8R, elemLibrary=STANDARD)
elemType2 = mesh.ElemType(elemCode=C3D6, elemLibrary=STANDARD)
elemType3 = mesh.ElemType(elemCode=C3D4, elemLibrary=STANDARD, 
    secondOrderAccuracy=OFF, distortionControl=DEFAULT)
pickedRegions =(cells, )
a.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2, 
    elemType3))
a.regenerate()


# mdb.Job(name='Job-1', model='Model-1-stress', description='', type=ANALYSIS, 
    # atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
    # memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
    # explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
    # modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, 
    # userSubroutine='H:\\voronoi_0522\\Euler_v4.f', scratch='', 
    # resultsFormat=ODB, numThreadsPerMpiProcess=1, multiprocessingMode=DEFAULT, 
    # numCpus=12, numDomains=12, numGPUs=0)

mdb.Job(name='Job-1', model='Model-1-stress', description='', type=ANALYSIS, 
    atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
    memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
    explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
    modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
    scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=12, 
    numDomains=12, numGPUs=0)
mdb.jobs['Job-1'].submit(consistencyChecking=OFF)
mdb.jobs['Job-1'].waitForCompletion()


###shuju
import visualization
from odbAccess import*
odb= visualization.openOdb('Job-1.odb')
node=odb.rootAssembly.nodeSets['M_SET-1']
# p=odb.rootAssembly.instances['Voronoi_3D-1'.upper()]
step=odb.steps['Step-1']
RFY=[]
temp=step.frames[-1].fieldOutputs['RF'].getSubset(region=node).values[0].data[1]
RFY.append(abs(temp)*10)

#rfy output
with open(r'H:\voronoi_0522\data\stru_'+str(int(IDX))+'\label.txt', 'w') as outfile:
    outfile.write("{:.2f}".format(RFY[0]))

np.savetxt(r'H:\voronoi_0522\data\stru_'+str(int(IDX))+'\Material.txt', Random_matrix, fmt='%f', newline='\n')
np.savetxt(r'H:\voronoi_0522\data\stru_'+str(int(IDX))+'\Euler.txt', random_Euler, fmt='%f', newline='\n')