#!/usr/bin/env python

###
### This file is generated automatically by SALOME v9.9.0 with dump python functionality
###

import sys
import salome

salome.salome_init()
import salome_notebook
notebook = salome_notebook.NoteBook()
sys.path.insert(0, r'/home/empathy/workstation/projects/project_academic/zhang_zhijie1/dissemination/laser_soldering/geometry_mesh')

###
### GEOM component
###

import GEOM
from salome.geom import geomBuilder
import math
import SALOMEDS


geompy = geomBuilder.New()

O = geompy.MakeVertex(0, 0, 0)
OX = geompy.MakeVectorDXDYDZ(1, 0, 0)
OY = geompy.MakeVectorDXDYDZ(0, 1, 0)
OZ = geompy.MakeVectorDXDYDZ(0, 0, 1)
Vector_1 = geompy.MakeVectorDXDYDZ(150, 150, 150)
Vertex_1 = geompy.MakeVertex(50, 50, 30)
Vertex_2 = geompy.MakeVertex(0, 0, -20)
Vertex_3 = geompy.MakeVertex(100, 100, 0)
Sphere_1 = geompy.MakeSpherePntR(Vertex_1, 50)
Box_1 = geompy.MakeBoxTwoPnt(Vertex_2, Vertex_3)
Solid_1 = geompy.MakeCutList(Sphere_1, [Box_1], True)
[Face_1base,Face_2curved_surface] = geompy.ExtractShapes(Solid_1, geompy.ShapeType["FACE"], True)
[Face_1base, Face_2curved_surface] = geompy.GetExistingSubObjects(Solid_1, False)
[Face_1base, Face_2curved_surface] = geompy.GetExistingSubObjects(Solid_1, False)
geompy.addToStudy( O, 'O' )
geompy.addToStudy( OX, 'OX' )
geompy.addToStudy( OY, 'OY' )
geompy.addToStudy( OZ, 'OZ' )
geompy.addToStudy( Vector_1, 'Vector_1' )
geompy.addToStudy( Vertex_1, 'Vertex_1' )
geompy.addToStudy( Sphere_1, 'Sphere_1' )
geompy.addToStudy( Vertex_2, 'Vertex_2' )
geompy.addToStudy( Vertex_3, 'Vertex_3' )
geompy.addToStudy( Box_1, 'Box_1' )
geompy.addToStudy( Solid_1, 'Solid_1' )
geompy.addToStudyInFather( Solid_1, Face_1base, 'Face_1base' )
geompy.addToStudyInFather( Solid_1, Face_2curved_surface, 'Face_2curved_surface' )

###
### SMESH component
###

import  SMESH, SALOMEDS
from salome.smesh import smeshBuilder

smesh = smeshBuilder.New()
#smesh.SetEnablePublish( False ) # Set to False to avoid publish in study if not needed or in some particular situations:
                                 # multiples meshes built in parallel, complex and numerous mesh edition (performance)

Mesh_1 = smesh.Mesh(Solid_1,'Mesh_1')
Regular_1D = Mesh_1.Segment()
Local_Length_1 = Regular_1D.LocalLength(5,None,1e-07)
MEFISTO_2D = Mesh_1.Triangle(algo=smeshBuilder.MEFISTO)
Max_Element_Area_1 = MEFISTO_2D.MaxElementArea(100)
NETGEN_3D = Mesh_1.Tetrahedron()
Max_Element_Volume_1 = NETGEN_3D.MaxElementVolume(1000)
Face_1base_1 = Mesh_1.GroupOnGeom(Face_1base,'Face_1base',SMESH.FACE)
Face_2curved_surface_1 = Mesh_1.GroupOnGeom(Face_2curved_surface,'Face_2curved_surface',SMESH.FACE)
isDone = Mesh_1.Compute()
[ Face_1base_1, Face_2curved_surface_1 ] = Mesh_1.GetGroups()
Mesh_2 = smesh.Mesh(Solid_1,'Mesh_2')
Regular_1D_1 = Mesh_2.Segment()
Local_Length_2 = Regular_1D_1.LocalLength(5,None,1e-07)
MEFISTO_2D_1 = Mesh_2.Triangle(algo=smeshBuilder.MEFISTO)
Max_Element_Area_2 = MEFISTO_2D_1.MaxElementArea(50)
NETGEN_3D_1 = Mesh_2.Tetrahedron()
Max_Element_Volume_2 = NETGEN_3D_1.MaxElementVolume(500)
Face_1base_2 = Mesh_2.GroupOnGeom(Face_1base,'Face_1base',SMESH.FACE)
Face_2curved_surface_2 = Mesh_2.GroupOnGeom(Face_2curved_surface,'Face_2curved_surface',SMESH.FACE)
isDone = Mesh_2.Compute()
[ Face_1base_2, Face_2curved_surface_2 ] = Mesh_2.GetGroups()


## Set names of Mesh objects
smesh.SetName(Regular_1D.GetAlgorithm(), 'Regular_1D')
smesh.SetName(NETGEN_3D.GetAlgorithm(), 'NETGEN 3D')
smesh.SetName(MEFISTO_2D.GetAlgorithm(), 'MEFISTO_2D')
smesh.SetName(Max_Element_Area_1, 'Max. Element Area_1')
smesh.SetName(Local_Length_1, 'Local Length_1')
smesh.SetName(Max_Element_Volume_1, 'Max. Element Volume_1')
smesh.SetName(Local_Length_2, 'Local Length_2')
smesh.SetName(Max_Element_Volume_2, 'Max. Element Volume_2')
smesh.SetName(Face_1base_1, 'Face_1base')
smesh.SetName(Max_Element_Area_2, 'Max. Element Area_2')
smesh.SetName(Face_2curved_surface_1, 'Face_2curved_surface')
smesh.SetName(Mesh_1.GetMesh(), 'Mesh_1')
smesh.SetName(Mesh_2.GetMesh(), 'Mesh_2')
smesh.SetName(Face_2curved_surface_2, 'Face_2curved_surface')
smesh.SetName(Face_1base_2, 'Face_1base')


if salome.sg.hasDesktop():
  salome.sg.updateObjBrowser()
