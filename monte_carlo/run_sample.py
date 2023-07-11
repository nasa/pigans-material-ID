import sys

'''
Script to read Elastic Modulus field from csv, generate plane stress
Abaqus model, run model, and extract nodal displacements.

Required command line args:
---------------------------
1) E field csv path
'''

'''
-----------------------------------------------------------------------------
 Create model of a two dimensional plate using plane stress elements (CPS8).
-----------------------------------------------------------------------------
'''

from abaqus import *
import testUtils
testUtils.setBackwardCompatibility()
from abaqusConstants import *
from odbAccess import *

import part, material, section, assembly, step, interaction
import regionToolset, displayGroupMdbToolset as dgm, mesh, load, job 
 

def run(LX, LY, POISSON, PRESSURE, E_FIELD, THICKNESS=0.2):
    #---------------------------------------------------------------------------
    
    # Create a model
    
    Mdb()
    modelName = 'weld_2d'
    myModel = mdb.Model(name=modelName)
        
    # Create a new viewport in which to display the model
    # and the results of the analysis.
    
    myViewport = session.Viewport(name=modelName)
    myViewport.makeCurrent()
    myViewport.maximize()
        
    #---------------------------------------------------------------------------
    
    # Create a part
    
    # Create a sketch for the base feature
    
    mySketch = myModel.Sketch(name='plateProfile',sheetSize=max([LX, LY]) * 2.5)
    mySketch.sketchOptions.setValues(viewStyle=AXISYM)
    mySketch.setPrimaryObject(option=STANDALONE)
    
    mySketch.rectangle(point1=(0.0, 0.0), point2=(LX, LY))
    
    myPlate = myModel.Part(name='Plate', 
        dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    myPlate.BaseShell(sketch=mySketch)
    mySketch.unsetPrimaryObject()
    del myModel.sketches['plateProfile']
    
    myViewport.setValues(displayedObject=myPlate)
    
    # Create a set referring to the whole part
    
    faces1 = myPlate.faces.findAt(((LX / 2, LY / 4, 0),),)
    myPlate.Set(faces=faces1, name='All')
    
   
    #---------------------------------------------------------------------------
    
    # Assign material properties
    
    # Create linear elastic material
    E_values = [n[3] for n in E_FIELD]
    max_E = max(E_values)
    min_E = min(E_values)
    temp_table = ((min_E, POISSON, min_E), (max_E, POISSON, max_E))
    myModel.Material(name='LinearElastic')
    myModel.materials['LinearElastic'].Elastic(temperatureDependency=ON,
                                               table=temp_table)
    myModel.HomogeneousSolidSection(name='SolidHomogeneous',
        material='LinearElastic', thickness=THICKNESS)
    
    region = myPlate.sets['All']
    
    # Assign the above section to the part
    
    myPlate.SectionAssignment(region=region, sectionName='SolidHomogeneous')
    
    #---------------------------------------------------------------------------
    
    # Create an assembly
    
    myAssembly = myModel.rootAssembly
    myViewport.setValues(displayedObject=myAssembly)
    myAssembly.DatumCsysByDefault(CARTESIAN)
    myAssembly.Instance(name='myPlate-1', part=myPlate)
    myPlateInstance = myAssembly.instances['myPlate-1']
    
    # Create a set for the edge to be fixed in X
    
    edges = myPlateInstance.edges
    e1 = myPlateInstance.edges.findAt((0, LY / 2, 0))
    edges1 = edges[e1.index:(e1.index+1)]
    myAssembly.Set(edges=edges1, name='fixedEdge')
    
    # Create a set for the point to be fixed in Y
    
    verts1 = myPlateInstance.vertices
    vmp = myPlateInstance.vertices.findAt((0, 0, 0))
    myPoint = verts1[vmp.index:(vmp.index+1)]
    myAssembly.Set(vertices=myPoint, name='fixedPt')
    
    # Create a set for the load edge
    
    e1 = myPlateInstance.edges.findAt((LX, LY / 2, 0))
    side1Edges1 = edges[e1.index:(e1.index+1)]
    myAssembly.Surface(side1Edges=side1Edges1, name='loadSurf')
    
    #---------------------------------------------------------------------------
    
    # Create a step for applying a load
    
    myModel.StaticStep(name='LoadPlate', previous='Initial',
        description='Apply the load')
    
    #---------------------------------------------------------------------------
    
    # Create loads and boundary conditions 
    
    # Assign boundary conditions
    
    region = myAssembly.sets['fixedEdge']
    myModel.DisplacementBC(name='FixedInX', 
        createStepName='Initial', region=region, u1=0.0,
        fixed=OFF, distributionType=UNIFORM, localCsys=None)
    
    region = myAssembly.sets['fixedPt']
    myModel.DisplacementBC(name='FixedInY', 
        createStepName='Initial', region=region, u2=0.0,
        fixed=OFF, distributionType=UNIFORM, localCsys=None)
    
    # Assign load conditions
    
    tSurf = myAssembly.surfaces['loadSurf']
    myModel.Pressure(name='Load', createStepName='LoadPlate',
        region=tSurf, distributionType=UNIFORM, magnitude=PRESSURE)
    
    #---------------------------------------------------------------------------
    
    # Create a mesh 
    
    # Seed all the edges

    myAssembly.seedPartInstance(regions=(myPlateInstance,), size=0.025,
                                deviationFactor=0.1, minSizeFactor=0.1)

    # Assign meshing controls to the respective regions
    
    faces1 = myPlateInstance.faces
    elemType1 = mesh.ElemType(elemCode=CPS8, elemLibrary=STANDARD)
    #elemType2 = mesh.ElemType(elemCode=CPS6M, elemLibrary=STANDARD)
    
    pickedRegions =(faces1, )
    myAssembly.setElementType(regions=pickedRegions,
        elemTypes=(elemType1,))
    
    # Generate mesh

    partInstances =(myPlateInstance, )
    myAssembly.generateMesh(regions=partInstances)
    

    #---------------------------------------------------------------------------
    
    # Create temperature field to map material properties

    mdb.models[modelName].MappedField(name='AnalyticalField-1',
        description='', regionType=POINT, partLevelData=False, localCsys=None,
        pointDataFormat=XYZ, fieldDataType=SCALAR, xyzPointData=E_FIELD)
    a = mdb.models[modelName].rootAssembly
    f1 = a.instances['myPlate-1'].faces
    faces1 = f1.getSequenceFromMask(mask=('[#3 ]', ), )
    region = a.Set(faces=faces1, name='All')
    mdb.models[modelName].Temperature(name='Predefined Field-1',
        createStepName='LoadPlate', region=region, distributionType=FIELD,
        crossSectionDistribution=CONSTANT_THROUGH_THICKNESS,
        field='AnalyticalField-1', magnitudes=(1.0, ))
    
    #---------------------------------------------------------------------------

    # Create history request

    regionDef=myAssembly.allInstances['myPlate-1'].sets['All']
    myModel.historyOutputRequests['H-Output-1'].setValues(
        variables=( 'UT', ),
        frequency=LAST_INCREMENT,
        region=regionDef,
        sectionPoints=DEFAULT, 
        rebar=EXCLUDE)
    
    #---------------------------------------------------------------------------

    # Create the job 
    
    myJob = mdb.Job(name=modelName, model=modelName,
        description='PIGAN 2D weld analysis')
    mdb.saveAs(pathName=modelName)
    
    #---------------------------------------------------------------------------
    
    '''
    ----------------------------------------------------------------------------
     Run model using plane stress elements (CPS8).
    ----------------------------------------------------------------------------
    '''
    
   
    # Open the CAE model and create a viewport for viewing it.
    
    Mdb()
    openMdb(modelName)
    myModel = mdb.models[modelName]
    myAssembly = myModel.rootAssembly
    myViewport = session.Viewport(name=modelName)
    myViewport.makeCurrent()
    myViewport.maximize()
    myViewport.setValues(displayedObject=myAssembly)
    
    #---------------------------------------------------------------------------
    
    # Submit the job for analysis
    
    mdb.jobs[modelName].submit()
    mdb.jobs[modelName].waitForCompletion()
    
    #---------------------------------------------------------------------------
    
    '''
    ----------------------------------------------------------------------------
     Postprocess.
    ----------------------------------------------------------------------------
    '''
    
    odb = openOdb(path=modelName + '.odb')
    openMdb(modelName)
    myModel = mdb.models[modelName]
    myAssembly = myModel.rootAssembly
    step = odb.steps['LoadPlate']
    regionDef=myAssembly.allInstances['myPlate-1'].sets['All']

    u1 = []
    u2 = []
    x = []
    y = []
    for n in regionDef.nodes:
        id_ = 'Node MYPLATE-1.{}'.format(n.label)
        h_region = step.historyRegions[id_]
        x.append(n.coordinates[0])
        y.append(n.coordinates[1])
        u1.append(h_region.historyOutputs['U1'].data[1][1])
        u2.append(h_region.historyOutputs['U2'].data[1][1])
    
    out = '\n'.join([','.join(map(str, z)) for z in zip(x, y, u1, u2)])
    print >> sys.__stdout__, out


def read_sample(SAMPLE_FILENAME):

    with open(SAMPLE_FILENAME, 'r') as fid:
        content = fid.readlines()

    e_field = tuple()
    for line in content:
        data = line.split(',')
        e_field = e_field + ((float(data[0]), float(data[1]),
                              float(data[2]), float(data[3])),)

    return e_field



if __name__ == '__main__':

    SAMPLE_FILENAME = sys.argv[-1]
    e_field = read_sample(SAMPLE_FILENAME)

    inputs = {
        'LX': 1.5,
        'LY': 0.5,
        'POISSON': 0.3,
        'PRESSURE': -0.004,
        'E_FIELD': e_field}

    run(**inputs)

