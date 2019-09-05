# (c) Lohit Petikam 2019

# bl_info = {
# 	"name": "SpectralMeshProcessing",
# 	"category": "Object",
# }

import bpy
import bmesh

import time

import numpy as np
from numpy import linalg as LA


def SMP_DefineProps():

	def cb_bw_update(self, context):
		if not self.SMP_Initialised:
			SMP_InitObject(self)

		num_vertices = len(self.data.vertices)
		w = 1*(np.arange(0, num_vertices) <= num_vertices*self.SMP_Bandwidth)
		SMP_Reconstruct(self, w)



	bpy.types.Object.SMP_Use = bpy.props.BoolProperty(name="SMP_Use",
		default=False,
		update=cb_bw_update
		)

	bpy.types.Object.SMP_Initialised = bpy.props.BoolProperty(name="SMP_Initialised",
		default=False
		)

	bpy.types.Object.SMP_Bandwidth = bpy.props.FloatProperty(name="SMP_Bandwidth",
		default=1.0,
		min=0,
		max=1.0,
		description="Fraction of coefficients to reconstruct mesh with.",
		update=cb_bw_update
		)


def SMP_InitObject(obj):

	mesh = obj.data

	print("SMP Intialising mesh:", mesh.name)

	start_time = time.time()

	edges = mesh.edges

	num_vertices = len(mesh.vertices)
	num_edges = len(mesh.edges)

	print("SMP loading vertices...")
	X = np.zeros(num_vertices)
	Y = np.zeros(num_vertices)
	Z = np.zeros(num_vertices)
	for i in range(0, num_vertices):
		X[i] = mesh.vertices[i].co.x
		Y[i] = mesh.vertices[i].co.y
		Z[i] = mesh.vertices[i].co.z


	W = np.zeros(num_edges)
	print("SMP generating cotangent weights...")
	# bpy.context.active_object = obj
	bpy.ops.object.mode_set(mode='EDIT')
	bpy.ops.mesh.select_all(action='DESELECT')
	bm = bmesh.from_edit_mesh(mesh)
	bm.edges.ensure_lookup_table()

	def calc_cotangent_weight(e):

		if len(e.link_faces) == 2:
			v0 = e.verts[0]
			v1 = e.verts[1]

			f0 = e.link_faces[0]
			f1 = e.link_faces[1]

			cv0 = None
			cv1 = None

			for v in f0.verts:
				if not v is v0:
					if not v is v1:
						cv0 = v

			for v in f1.verts:
				if not v is v0:
					if not v is v1:
						cv1 = v

			a0 = np.arccos((v0.co - cv0.co).normalized().dot((v1.co - cv0.co).normalized()))
			a1 = np.arccos((v0.co - cv1.co).normalized().dot((v1.co - cv1.co).normalized()))

			return ((0.5/np.tan(a0)+0.5/np.tan(a1)))
		else:
			print("link faces not == 2: ", len(e.link_faces))
			return 1

	for i in range(0, len(bm.edges)):
		ei = bm.edges[i]
		# W[i] = calc_cotangent_weight(ei)
		W[i] = 1


	print("SMP generating L...")
	L = np.zeros((num_vertices, num_vertices))

	for  i in range(0, len(edges)):
		e = edges[i]
		wi = W[i]
		L[e.vertices[0], e.vertices[1]] = wi
		L[e.vertices[1], e.vertices[0]] = wi
	
	v_sum = np.sum(L, axis=0)

	for i in range(0, num_vertices):
		L[i, i] = -v_sum[i]

	bpy.ops.object.mode_set(mode='OBJECT')

	print("SMP solving E...")
	D, E = LA.eig(L)

	e_X = np.matmul( E.transpose() , X )
	e_Y = np.matmul( E.transpose() , Y )
	e_Z = np.matmul( E.transpose() , Z )


	print("SMP saving data...")
	mesh.id_data['num_vertices'] = num_vertices

	mesh.id_data['X'] = X.astype(float).tostring()
	mesh.id_data['Y'] = Y.astype(float).tostring()
	mesh.id_data['Z'] = Z.astype(float).tostring()

	mesh.id_data['D'] = D.astype(float).tostring()
	mesh.id_data['E'] = E.astype(float).tostring()

	mesh.id_data['e_X'] = e_X.astype(float).tostring()
	mesh.id_data['e_Y'] = e_Y.astype(float).tostring()
	mesh.id_data['e_Z'] = e_Z.astype(float).tostring()

	mesh.id_data['spectrum'] = np.array([1,1,1]).astype(float).tostring()

	obj.SMP_Initialised = True

	print("SMP object initialised! (%.2f s)", time.time() - start_time)

def SMP_Reconstruct(obj, SMP_Spectrum):

	mesh = obj.data

	num_vertices = mesh.id_data['num_vertices']

	E = np.fromstring(mesh.id_data['E'], dtype=float).reshape((num_vertices, num_vertices))

	e_X = np.fromstring(mesh.id_data['e_X'], dtype=float)
	e_Y = np.fromstring(mesh.id_data['e_Y'], dtype=float)
	e_Z = np.fromstring(mesh.id_data['e_Z'], dtype=float)
	
	# w = 1*(np.arange(0, num_vertices) < num_vertices*bandwidth)

	w = SMP_Spectrum

	if not len(SMP_Spectrum) == num_vertices:
		hist = SMP_Spectrum
		hist_len = len(hist)
		hist_x = np.arange(hist_len)/hist_len
		vert_x = np.arange(num_vertices)/num_vertices
		w = np.interp(vert_x, np_x, hist)

	out_x = np.matmul(E, e_X * w)
	out_y = np.matmul(E, e_Y * w)
	out_z = np.matmul(E, e_Z * w)

	for i in range(0, num_vertices):
		mesh.vertices[i].co.x = out_x[i]
		mesh.vertices[i].co.y = out_y[i]
		mesh.vertices[i].co.z = out_z[i]

def SMP_Revert(obj):

	mesh = obj.data

	num_vertices = mesh.id_data['num_vertices']
	
	out_x = np.fromstring(mesh.id_data['X'], dtype=float)
	out_y = np.fromstring(mesh.id_data['Y'], dtype=float)
	out_z = np.fromstring(mesh.id_data['Z'], dtype=float)

	for i in range(0, num_vertices):
		mesh.vertices[i].co.x = out_x[i]
		mesh.vertices[i].co.y = out_y[i]
		mesh.vertices[i].co.z = out_z[i]


class SMPInitOperator(bpy.types.Operator):
	"""Tooltip"""
	bl_idname = "object.smp_init"
	bl_label = "SMP Initialise"

	@classmethod
	def poll(cls, context):
		return context.active_object is not None

	def execute(self, context):
		for o in context.selected_objects:
			SMP_InitObject(o)
		return {'FINISHED'}


class SMPRevertOperator(bpy.types.Operator):
	"""Tooltip"""
	bl_idname = "object.smp_revert"
	bl_label = "SMP Revert"

	@classmethod
	def poll(cls, context):
		return context.active_object is not None

	def execute(self, context):
		for o in context.selected_objects:
			SMP_Revert(o)
		return {'FINISHED'}


class SMPPanel(bpy.types.Panel):
	"""Creates a Panel in the Object properties window"""
	bl_label = "SMP Panel"
	bl_idname = "OBJECT_PT_smp"
	bl_space_type = 'PROPERTIES'
	bl_region_type = 'WINDOW'
	bl_context = "object"

	def draw(self, context):
		layout = self.layout

		obj = context.object

		row = layout.row()
		row.label(text="SMP Panel v1", icon='MOD_SMOOTH')

		row = layout.row()
		row.label(text="Selected Objects: " + str(len(context.selected_objects)))

		row = layout.row()
		row.label(text="Active Object: " + context.active_object.name)
		
		row = layout.row()
		row.label(text="Mesh :" + obj.data.name)

		row = layout.row()
		row.label(text="SMP Initialised: " + str(obj.SMP_Initialised==True))

		row = layout.row()
		row.operator("object.smp_init")

		row = layout.row()
		row.operator("object.smp_revert")
		
		if obj.SMP_Initialised:
			row = layout.row()
			row.prop(obj, "SMP_Bandwidth")




def register():
	SMP_DefineProps()
	bpy.utils.register_class(SMPInitOperator)
	bpy.utils.register_class(SMPRevertOperator)
	bpy.utils.register_class(SMPPanel)


def unregister():
	bpy.utils.unregister_class(SMPInitOperator)
	bpy.utils.unregister_class(SMPRevertOperator)
	bpy.utils.unregister_class(SMPPanel)


if __name__ == "__main__":

	register()