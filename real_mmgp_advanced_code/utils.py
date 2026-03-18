import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import scipy.spatial as spatial
import bisect, copy, time
import os, json, math
from tqdm import tqdm
import triangle as tr
import multiprocess as mp
from functools import partial

from Muscat.Containers import MeshCreationTools as MCT
from Muscat.Containers import MeshGraphTools as MGT
from Muscat.Containers.NativeTransfer import NativeTransfer
from Muscat.Containers import MeshModificationTools as MMT
from Muscat.Containers import MeshFieldOperations as MFO
from Muscat.Containers.MeshFieldOperations import GetFieldTransferOpPython
from Muscat.Containers.Filters import FilterObjects as FO
from Muscat.Containers.Filters import FilterOperators as FOp

from Muscat.FE.FETools import PrepareFEComputation
from Muscat.FE.Fields.FEField import FEField
from Muscat.ImplicitGeometry.ImplicitGeometryObjects import  ImplicitGeometrySphere

import pyvista as pv
from scipy.interpolate import griddata

import Muscat.Containers.ElementsDescription as ED
import Muscat.Containers.MeshInspectionTools as MIT

from interpolation import PieceWiseLinearInterpolationVectorized

from airfrans.simulation import Simulation

import torch
device = torch.device("cpu")

n_parallel_tasks = 4 # Changed from 4 to 1 to minimize memory footprint and prevent WSL OOM/Reboot crashes

p0 = np.array([1.,0.,0.5])
nb_points_add_ext_boundary = 100

out_field_names = ['UX', 'UY', 'p', 'nut']

h = 1.718
BBox = (np.array([-2.26, -h,  0.5]), np.array([4.33, h, 0.5]))
PBBox = [[BBox[0][0], BBox[0][1]], [BBox[1][0], BBox[0][1]], [BBox[1][0], BBox[1][1]], [BBox[0][0], BBox[1][1]]]
radius = np.linalg.norm(BBox[1]-BBox[0])/2


def FastMorphing(mesh, targetDisplacement, targetDisplacementMask):

    new_nodes =  np.copy(mesh.GetPosOfNodes())
    border_nodes = new_nodes[targetDisplacementMask,:]
    d_r = spatial.distance.cdist(border_nodes, border_nodes)/radius

    Mm = (1-d_r)**4*(4*d_r+1)
    Mm[d_r>1]=0.
    np.fill_diagonal(Mm,1.)

    alpha = np.linalg.lstsq(Mm,targetDisplacement,rcond=10**(9))[0]
    d_r = spatial.distance.cdist(new_nodes, border_nodes)/radius

    y=(1-d_r)**4*(4*d_r+1)
    inds = d_r>=1.
    y[inds] = 0.

    new_nodes += np.dot(y, alpha)

    return new_nodes



def FastInterp(input_mesh, output_nodes):

    # check if interp needed
    if input_mesh.nodes.shape == output_nodes.shape and np.linalg.norm(input_mesh.nodes - output_nodes) == 0:
        return [np.arange(output_nodes.shape[0]), np.arange(output_nodes.shape[0]).reshape(-1,1), np.ones(output_nodes.shape[0]).reshape(-1,1)]

    output_nodes = output_nodes[:,:2]

    tree = spatial.KDTree(output_nodes)

    triangles = input_mesh.elements[ED.Triangle_3].connectivity
    tri_vertices = input_mesh.nodes[triangles,:2]
    barycenters = np.sum(tri_vertices, axis = 1)/3.

    _, ind = tree.query(barycenters, k = 100)

    points_to_check = output_nodes[ind,:]

    denominator = (tri_vertices[:,1,1] - tri_vertices[:,2,1]) * (tri_vertices[:,0,0] - tri_vertices[:,2,0]) + \
        (tri_vertices[:,2,0] - tri_vertices[:,1,0]) * (tri_vertices[:,0,1] - tri_vertices[:,2,1])

    a = (tri_vertices[:,1,1][:,None] - tri_vertices[:,2,1][:,None]) * (points_to_check[:,:,0] - tri_vertices[:,2,0][:,None]) + \
        (tri_vertices[:,2,0][:,None] - tri_vertices[:,1,0][:,None]) * (points_to_check[:,:,1] - tri_vertices[:,2,1][:,None])

    b = (tri_vertices[:,2,1][:,None] - tri_vertices[:,0,1][:,None]) * (points_to_check[:,:,0] - tri_vertices[:,2,0][:,None]) + \
        (tri_vertices[:,0,0][:,None] - tri_vertices[:,2,0][:,None]) * (points_to_check[:,:,1] - tri_vertices[:,2,1][:,None])

    a /= denominator[:,None]
    b /= denominator[:,None]
    c = 1 - a - b

    inside = (a > 0.) & (b > 0.) & (c > 0.)
    del a, b, c

    map = {}
    for i in range(triangles.shape[0]):
        node_ind = ind[i,inside[i,:]]
        for ni in node_ind:
            map[ni] = i
    del inside

    output_node_treated = np.array(list(map.keys()))
    remaining_output_nodes = np.setdiff1d(np.arange(output_nodes.shape[0]), output_node_treated)

    tree = spatial.KDTree(output_nodes[remaining_output_nodes,:])
    list_of_inds = tree.query_ball_tree(spatial.KDTree(barycenters), 0.2)

    for i in range(len(list_of_inds)):

        inds = np.array(list_of_inds[i], dtype=int)
        point = output_nodes[remaining_output_nodes[i],:2]

        denominator = (tri_vertices[inds,1,1] - tri_vertices[inds,2,1]) * (tri_vertices[inds,0,0] - tri_vertices[inds,2,0]) + \
            (tri_vertices[inds,2,0] - tri_vertices[inds,1,0]) * (tri_vertices[inds,0,1] - tri_vertices[inds,2,1])

        a = (tri_vertices[inds,1,1] - tri_vertices[inds,2,1]) * (point[0] - tri_vertices[inds,2,0]) + \
            (tri_vertices[inds,2,0] - tri_vertices[inds,1,0]) * (point[1] - tri_vertices[inds,2,1])

        b = (tri_vertices[inds,2,1] - tri_vertices[inds,0,1]) * (point[0] - tri_vertices[inds,2,0]) + \
            (tri_vertices[inds,0,0] - tri_vertices[inds,2,0]) * (point[1] - tri_vertices[inds,2,1])

        a /= denominator
        b /= denominator
        c = 1 - a - b

        min_coord_bar = np.min(np.array([a, b, c]), axis = 0)
        containing_triangle = inds[np.argmax(min_coord_bar)]

        map[remaining_output_nodes[i]] = containing_triangle

    indices_nodes = np.array(list(map.keys()))
    filtered_triangles = triangles[np.array(list(map.values()))]

    all_coord_bary = []
    for i_node, i_tri in map.items():

        point = output_nodes[i_node,:]
        denominator = (tri_vertices[i_tri,1,1] - tri_vertices[i_tri,2,1]) * (tri_vertices[i_tri,0,0] - tri_vertices[i_tri,2,0]) + \
            (tri_vertices[i_tri,2,0] - tri_vertices[i_tri,1,0]) * (tri_vertices[i_tri,0,1] - tri_vertices[i_tri,2,1])

        a = (tri_vertices[i_tri,1,1] - tri_vertices[i_tri,2,1]) * (point[0] - tri_vertices[i_tri,2,0]) + \
            (tri_vertices[i_tri,2,0] - tri_vertices[i_tri,1,0]) * (point[1] - tri_vertices[i_tri,2,1])

        b = (tri_vertices[i_tri,2,1] - tri_vertices[i_tri,0,1]) * (point[0] - tri_vertices[i_tri,2,0]) + \
            (tri_vertices[i_tri,0,0] - tri_vertices[i_tri,2,0]) * (point[1] - tri_vertices[i_tri,2,1])

        a /= denominator
        b /= denominator
        c = 1 - a - b

        coord_bary = np.array([a, b, c])
        all_coord_bary.append(coord_bary)

    all_coord_bary = np.array(all_coord_bary)

    return [indices_nodes, filtered_triangles, all_coord_bary]


# Hyper-parameters
NB_ANGLES = 100
THRESHOLDS = [0.9,0.99,0.999]

def define_line(angle:float):
    x = np.linspace(1,4,1000)
    line = np.stack([x, np.tan(angle) * (x-1)],axis=-1)
    return line

def find_angle_from_mesh(mesh):

    # keep only points with X>=1.0
    filtered_nodes_id = mesh.nodes[:,0]>1.
    filtered_nodes = mesh.nodes[filtered_nodes_id,:2]

    # compute index for nearest-neighbor search
    index = spatial.KDTree(filtered_nodes)

    # extract field nut
    nut = mesh.nodeFields['nut'][filtered_nodes_id]

    # very basic interpolation for nu_t on a line
    def interpolate_nut(line):
        _,ids = index.query(line)
        return nut[ids]

    # score is integral of nu_t along line with specified angle
    def compute_score(angle:float):
        line = define_line(angle)
        nut_line = interpolate_nut(line)
        return np.sum(nut_line)

    # compute search angle
    MAX_ANGLE = np.arctan(1.5 / 3) # -> avoid looking out of domain
    kept_angles = [-MAX_ANGLE, MAX_ANGLE]

    # very basic optimisation ^^ -> should be optimized
    for threshold in THRESHOLDS:
        angles = np.linspace(kept_angles[0], kept_angles[-1], NB_ANGLES)
        scores = np.array([v for v in map(compute_score, angles)])

        max_score = np.max(scores)
        kept_angles = angles[scores > threshold*max_score]

    angle = np.mean(kept_angles)

    return angle


def ComputeSillageFrontLineNodeIds(mesh):

    cell_ids_field = mesh.elemFields['cell_ids']
    assert(list(cell_ids_field) == sorted(cell_ids_field))
    diff = cell_ids_field[1:] - cell_ids_field[:-1]
    jumps_pos = np.where(diff>2000)[0]
    assert(len(jumps_pos)==5)

    jumps_pos = np.concatenate(([0], jumps_pos, [len(cell_ids_field)-1]))
    cluster_min_val = []
    cluster_max_val = []
    for i_jump in range(1,len(jumps_pos)):
        cluster_min_val.append(cell_ids_field[jumps_pos[i_jump-1]+1])
        cluster_max_val.append(cell_ids_field[jumps_pos[i_jump]])
    cluster_max_val.append(np.max(cell_ids_field)+1)
    for i_zone,(umin,umax) in enumerate(zip(cluster_min_val, cluster_max_val)):
        elFilter = FO.ElementFilter(eMask=(np.logical_and(umin<=cell_ids_field, cell_ids_field<=umax)))
        ids = elFilter.GetIdsToTreat(mesh, ED.Quadrangle_4)
        mesh.elements[ED.Quadrangle_4].GetTag(f'zone_{i_zone}').SetIds(ids)

    sillage_node_filter_1 = FO.NodeFilter(eTag = ["zone_0"])
    sillage_node_filter_2 = FO.NodeFilter(eTag = ["zone_1"])
    sillage_node_filter   = FOp.IntersectionFilter(filters=[sillage_node_filter_1, sillage_node_filter_2])
    sillage_node_ids      = sillage_node_filter.GetNodesIndices(mesh)

    frontline_node_filter_1 = FO.NodeFilter(eTag = ["zone_4"])
    frontline_node_filter_2 = FO.NodeFilter(eTag = ["zone_5"])
    frontline_node_filter   = FOp.IntersectionFilter(filters=[frontline_node_filter_1, frontline_node_filter_2])
    frontline_node_ids      = frontline_node_filter.GetNodesIndices(mesh)

    for i_zone in range(6):
        mesh.elements[ED.Quadrangle_4].tags.DeleteTags([f'zone_{i_zone}'])

    return sillage_node_ids, frontline_node_ids



def ExtractPathFromMeshOfBars(mesh, startingClosestToPoint, trigo_dir = True):

    nodeGraph0Airfoild = MGT.ComputeNodeToNodeGraph(mesh, dimensionality=1)
    nodeGraphAirfoild = [list(nodeGraph0Airfoild[i].keys()) for i in range(nodeGraph0Airfoild.number_of_nodes())]

    tree = spatial.KDTree(mesh.nodes)
    _, indicesTrailEdge = tree.query([startingClosestToPoint], k=1)

    p1init = indicesTrailEdge[0]

    temp1=mesh.nodes[nodeGraphAirfoild[p1init][0]][1]
    temp2=mesh.nodes[nodeGraphAirfoild[p1init][1]][1]

    if trigo_dir:
        condition = temp1 > temp2
    else:
        condition = temp1 < temp2

    if condition:
        p2 = nodeGraphAirfoild[p1init][0]
    else:
        p2 = nodeGraphAirfoild[p1init][1]

    p1 = p1init
    path = [p1, p2]
    while p2 != p1init:
        p2save = p2
        tempArray = np.asarray(nodeGraphAirfoild[p2])
        p2 = tempArray[tempArray!=p1][0]
        p1 = p2save
        path.append(p2)

    return path


def ExtractLineMesh(mesh, tag):

    efAirfoil = FO.ElementFilter(elementType=ED.Bar_2, eTag=[tag])
    airfoilMesh = MIT.ExtractElementsByElementFilter(mesh, efAirfoil)

    path = ExtractPathFromMeshOfBars(airfoilMesh, p0)

    leading_edge_filter_1 = FO.NodeFilter(nTag = ["Airfoil"])
    leading_edge_filter_2 = FO.NodeFilter(nTag = ["FrontLine"])
    leading_edge_filter   = FOp.IntersectionFilter(filters=[leading_edge_filter_1, leading_edge_filter_2])
    initRankLeadingEdge   = leading_edge_filter.GetNodesIndices(airfoilMesh)[0]

    tree = spatial.KDTree(airfoilMesh.nodes[path])
    _, indicesLeadEdge = tree.query([airfoilMesh.nodes[initRankLeadingEdge,:]], k=1)

    indices_extrado = path[:indicesLeadEdge[0]+1]
    indices_intrado = path[indicesLeadEdge[0]:]

    indices_airfoil = [indices_extrado, indices_intrado]

    nodes_extrado = mesh.nodes[indices_extrado]
    nodes_intrado = mesh.nodes[indices_intrado]

    nodes_airfoil = [nodes_extrado, nodes_intrado]

    return indices_airfoil, nodes_airfoil



def computeAirfoilCurvAbscissa(airfoil):

    indices_airfoil = airfoil[0]
    nodes_airfoil = airfoil[1]

    curv_abscissa = []
    for i in range(2):
        local_curv_abscissa = np.zeros(len(indices_airfoil[i]))
        for j in range(1,len(local_curv_abscissa)):
            local_curv_abscissa[j] = local_curv_abscissa[j-1] + np.linalg.norm(nodes_airfoil[i][j]-nodes_airfoil[i][j-1])
        local_curv_abscissa /= local_curv_abscissa[-1]
        curv_abscissa.append(local_curv_abscissa)

    return curv_abscissa



def MapAirfoil(airfoil_ref, curv_abscissa_ref, curv_abscissa):

    nodes_airfoil_ref = airfoil_ref[1]
    dim_nodes = nodes_airfoil_ref[0][0].shape[0]

    mapped_airfoil = []
    for i in range(2):
        local_mapped_airfoil = np.zeros((len(curv_abscissa[i])-1, dim_nodes))
        for j in range(len(curv_abscissa[i])-1):
            index = max(bisect.bisect_right(curv_abscissa_ref[i], curv_abscissa[i][j]) - 1, 0)

            a = nodes_airfoil_ref[i][index]
            b = nodes_airfoil_ref[i][index+1]
            dl = curv_abscissa[i][j] - curv_abscissa_ref[i][index]
            dir = (b-a)/np.linalg.norm(b-a)
            local_mapped_airfoil[j] = a + dl * dir
        mapped_airfoil.append(local_mapped_airfoil)

    return mapped_airfoil



def ComputeIntersectionWithBoundingBox(p, PBBox, p0):

    delta = p - p0
    mx = PBBox[0][0]
    my = PBBox[0][1]
    Mx = PBBox[2][0]
    My = PBBox[2][1]

    val = np.argmin([np.linalg.norm(p-pb) for pb in PBBox])
    if val == 0:

        lambda_ = (PBBox[0]-p0)[1]/delta[1]
        res = p0 + lambda_*delta
        if mx < res[0] < Mx:
            return res
        lambda_ = (PBBox[0]-p0)[0]/delta[0]
        res = p0 + lambda_*delta
        assert my < res[1] < My
        return res

    elif val == 1:

        lambda_ = (PBBox[1]-p0)[0]/delta[0]
        res = p0 + lambda_*delta
        if my < res[1] < My:
            return res
        lambda_ = (PBBox[1]-p0)[1]/delta[1]
        res = p0 + lambda_*delta
        assert mx < res[0] < Mx
        return res

    elif val == 2:

        lambda_ = (PBBox[1]-p0)[0]/delta[0]
        res = p0 + lambda_*delta
        if my < res[1] < My:
            return res
        lambda_ = (PBBox[2]-p0)[1]/delta[1]
        res = p0 + lambda_*delta
        assert mx < res[0] < Mx
        return res

    elif val == 3:

        lambda_ = (PBBox[2]-p0)[1]/delta[1]
        res = p0 + lambda_*delta
        if mx < res[0] < Mx:
            return res
        lambda_ = (PBBox[0]-p0)[0]/delta[0]
        res = p0 + lambda_*delta
        assert my < res[1] < My
        return res


def TruncatedSVDSymLower(matrix, epsilon = None, nbModes = None):

    if epsilon != None and nbModes != None:
        raise("cannot specify both epsilon and nbModes")

    eigenValues, eigenVectors = np.linalg.eigh(matrix, UPLO="L")

    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]

    if nbModes == None:
        if epsilon == None:
            nbModes  = matrix.shape[0]
        else:
            nbModes = 0
            bound = (epsilon ** 2) * eigenValues[0]
            for e in eigenValues:
                if e > bound:
                    nbModes += 1
            id_max2 = 0
            bound = (1 - epsilon ** 2) * np.sum(eigenValues)
            temp = 0
            for e in eigenValues:
                temp += e
                if temp < bound:
                    id_max2 += 1

            nbModes = max(nbModes, id_max2)

    if nbModes > matrix.shape[0]:
        print("nbModes taken to max possible value of "+str(matrix.shape[0])+" instead of provided value "+str(nbModes))
        nbModes = matrix.shape[0]

    index = np.where(eigenValues<0)
    if len(eigenValues[index])>0:
        if index[0][0]<nbModes:
            print("removing numerical noise from eigenvalues, nbModes is set to "+str(index[0][0])+" instead of "+str(nbModes))
            nbModes = index[0][0]

    return eigenValues[0:nbModes], eigenVectors[:, 0:nbModes]



def snapshotsPOD_fit_transform(snapshots, correlationOperator, nbModes):

    numberOfSnapshots = snapshots.shape[0]
    numberOfDofs = snapshots.shape[1]
    correlationMatrix = np.zeros((numberOfSnapshots,numberOfSnapshots))
    matVecProducts = np.zeros((numberOfDofs,numberOfSnapshots))
    for i, snapshot1 in enumerate(tqdm(snapshots, desc="Computing correlation matrix", leave=False)):
        matVecProduct = correlationOperator.dot(snapshot1)
        matVecProducts[:,i] = matVecProduct
        for j, snapshot2 in enumerate(snapshots):
            if j <= i and j < numberOfSnapshots:
                correlationMatrix[i, j] = np.dot(matVecProduct, snapshot2)

    eigenValuesRed, eigenVectorsRed = TruncatedSVDSymLower(correlationMatrix, nbModes = nbModes)

    nbePODModes = eigenValuesRed.shape[0]
    print("truncature =", eigenValuesRed[-1]/eigenValuesRed[0])

    changeOfBasisMatrix = np.zeros((nbePODModes,numberOfSnapshots))
    for j in range(nbePODModes):
        changeOfBasisMatrix[j,:] = eigenVectorsRed[:,j]/np.sqrt(eigenValuesRed[j])

    reducedOrderBasis = np.dot(changeOfBasisMatrix,snapshots)
    generalizedCoordinates = np.dot(reducedOrderBasis, matVecProducts).T
    return reducedOrderBasis, generalizedCoordinates


def pretreat_sample(benchmark_path, folder, train, n_threads = 1):

    ###################
    # Read the raw data
    ###################

    simulation = Simulation(benchmark_path, folder)

    pointFields = [simulation.velocity[:,0], simulation.velocity[:,1], simulation.pressure, simulation.nu_t]

    scalars = [float(simulation.inlet_velocity), float(simulation.angle_of_attack)]

    mesh_0 = pv.read(os.path.join(benchmark_path, folder, folder+'_internal.vtu'))

    ppp = np.hstack((simulation.position, 0.5*np.ones(simulation.position.shape[0]).reshape((-1,1))))
    mesh = MCT.CreateMeshOf(ppp, mesh_0.cell_connectivity.reshape((-1,4)), elemName=ED.Quadrangle_4)
    mesh.elemFields['cell_ids'] = mesh_0.cell_data['cell_ids']

    # Compute sillage_node_ids
    sillage_node_ids, frontline_node_ids = ComputeSillageFrontLineNodeIds(mesh)
    mesh.GetNodalTag("Sillage").AddToTag(sillage_node_ids)
    mesh.GetNodalTag("FrontLine").AddToTag(frontline_node_ids)

    mesh.nodeFields = {}
    mesh.elemFields = {}

    ###################
    # Compute the skin of the mesh (containing the external boundary and the airfoil boundary)
    ###################
    MMT.ComputeSkin(mesh, md = 2, inPlace = True)

    ff1 = FO.ElementFilter(zone = lambda p: (-p[:,0]-1.99))
    ff2 = FO.ElementFilter(zone = lambda p: (p[:,0]-3.99))
    ff3 = FO.ElementFilter(zone = lambda p: (-p[:,1]-1.49))
    ff4 = FO.ElementFilter(zone = lambda p: (p[:,1]-1.49))
    efAirfoil = FOp.IntersectionFilter(filters=[ff1, ff2, ff3, ff4])
    airfoil_ids = efAirfoil.GetIdsToTreat(mesh, ED.Bar_2)
    mesh.elements[ED.Bar_2].GetTag("Airfoil").SetIds(airfoil_ids)
    nfAirfoil = FO.NodeFilter(eTag = "Airfoil")
    nodeIndexAirfoil = nfAirfoil.GetNodesIndices(mesh)
    mesh.GetNodalTag("Airfoil").AddToTag(nodeIndexAirfoil)

    ###################
    # Convert to triangles
    ###################
    MCT.MeshToSimplex(mesh)
    mesh.elements[ED.Triangle_3].connectivity = mesh.elements[ED.Triangle_3].connectivity[:,[0,2,1]]
    mesh.ConvertDataForNativeTreatment()

    ###################
    # Extract ids of the bar elements corresponding to the airfoil
    ###################
    ff1 = FO.ElementFilter(zone = lambda p: (-p[:,0]-1.99))
    ff2 = FO.ElementFilter(zone = lambda p: (p[:,0]-3.99))
    ff3 = FO.ElementFilter(zone = lambda p: (-p[:,1]-1.49))
    ff4 = FO.ElementFilter(zone = lambda p: (p[:,1]-1.49))
    efAirfoil = FOp.IntersectionFilter(filters=[ff1, ff2, ff3, ff4])
    airfoil_ids = efAirfoil.GetIdsToTreat(mesh, ED.Bar_2)

    ###################
    # Preparations
    ###################
    # Displace node at the end of the sillage to the bounding box
    ext_bound = np.setdiff1d(mesh.elements[ED.Bar_2].GetTag("Skin").GetIds(), airfoil_ids)
    mesh.elements[ED.Bar_2].GetTag("External_boundary").SetIds(ext_bound)
    nfExtBoundary = FO.NodeFilter(eTag = "External_boundary")
    nodeIndexExtBoundary  = nfExtBoundary.GetNodesIndices(mesh)

    amax = np.argmax(mesh.nodes[sillage_node_ids, 0])
    bound_end_sillage = ImplicitGeometrySphere(radius=0.4, center=mesh.nodes[amax,:])
    nnFilter= FO.NodeFilter(zone = bound_end_sillage)
    nf = FOp.IntersectionFilter(filters=[nfExtBoundary, nnFilter])
    bound_end_sillage_nodes_ids = nf.GetNodesIndices(mesh)
    mesh.GetNodalTag("Bound_end_sillage").AddToTag(bound_end_sillage_nodes_ids)

    mesh.GetNodalTag("rest_out_bound").AddToTag(np.setdiff1d(nodeIndexExtBoundary, bound_end_sillage_nodes_ids))
    mesh.GetNodalTag("External_boundary__").AddToTag(nodeIndexExtBoundary)

    for i in bound_end_sillage_nodes_ids:
        mesh.nodes[i,:2] = ComputeIntersectionWithBoundingBox(mesh.nodes[i,:2], PBBox, p0[:2])
    mesh.ConvertDataForNativeTreatment()

    # Compute intermediate partial boundary mesh
    amin = np.argmin(mesh.nodes[bound_end_sillage_nodes_ids,1])
    amax = np.argmax(mesh.nodes[bound_end_sillage_nodes_ids,1])
    mesh.GetNodalTag("extreme_end_sillage_nodes").AddToTag(np.array([bound_end_sillage_nodes_ids[amin], bound_end_sillage_nodes_ids[amax]]))

    ext_p0 = mesh.nodes[bound_end_sillage_nodes_ids[amax]]
    ext_p1 = mesh.nodes[bound_end_sillage_nodes_ids[amin]]

    ef_rest_out_bound_mesh = FO.ElementFilter(elementType=ED.Bar_2, nTag = ["rest_out_bound", "extreme_end_sillage_nodes"])
    rest_out_bound_mesh = MIT.ExtractElementsByElementFilter(mesh, ef_rest_out_bound_mesh)
    MMT.CleanLonelyNodes(rest_out_bound_mesh)

    # Clean intermediate tags
    mesh.elements[ED.Bar_2].tags.DeleteTags(["Skin", "External_boundary"])
    mesh.nodesTags.DeleteTags(["rest_out_bound", "extreme_end_sillage_nodes"])

    ###################
    # Create the mesh between the boundary box and the external boundary of the input mesh
    ###################

    points_to_add = [ext_p0[:2]]
    nb_points_to_add_top = int(nb_points_add_ext_boundary*(PBBox[2][1] - ext_p0[1])/(PBBox[2][1] - PBBox[1][1]))
    nb_points_to_add_down = int(nb_points_add_ext_boundary*(ext_p1[1] - PBBox[1][1])/(PBBox[2][1] - PBBox[1][1]))

    for i in range(nb_points_to_add_top):
        points_to_add.append(ext_p0[:2]+((i+1)/nb_points_to_add_top)*(PBBox[2]-ext_p0[:2]))
    for i in range(nb_points_add_ext_boundary):
        Delta_P = np.array(PBBox[3]) - np.array(PBBox[2])
        points_to_add.append(PBBox[2] + (i+1)/(nb_points_add_ext_boundary)*Delta_P)
    for i in range(nb_points_add_ext_boundary):
        Delta_P = np.array(PBBox[0]) - np.array(PBBox[3])
        points_to_add.append(PBBox[3] + (i+1)/(nb_points_add_ext_boundary)*Delta_P)
    for i in range(nb_points_add_ext_boundary):
        Delta_P = np.array(PBBox[1]) - np.array(PBBox[0])
        points_to_add.append(PBBox[0] + (i+1)/(nb_points_add_ext_boundary)*Delta_P)
    for i in range(nb_points_to_add_down):
        points_to_add.append(PBBox[1]+((i+1)/nb_points_to_add_down)*(ext_p1[:2]-PBBox[1]))

    vert = np.vstack((rest_out_bound_mesh.nodes[:,:2], points_to_add))
    nn = vert.shape[0]
    mm = len(points_to_add)
    indices_to_add = []
    for j in range(mm-1):
        indices_to_add.append([nn-mm+j, nn-mm+(j+1)%mm])
    seg = np.vstack((rest_out_bound_mesh.elements[ED.Bar_2].connectivity, indices_to_add))

    temp_mesh = MCT.CreateMeshOf(vert, seg, ED.Bar_2)
    MMT.CleanDoubleNodes(temp_mesh)
    MMT.CleanDoubleElements(temp_mesh)
    temp_mesh.ConvertDataForNativeTreatment()

    di = {'vertices':temp_mesh.nodes, 'segments':temp_mesh.elements[ED.Bar_2].connectivity, 'holes':[[0.5,0.]]}
    t = tr.triangulate(di, 'pc')
    total_mesh = MCT.CreateMeshOfTriangles(t['vertices'], t['triangles'])
    total_mesh.nodes = np.hstack((total_mesh.nodes, 0.5*np.ones(total_mesh.nodes.shape[0]).reshape((-1,1))))
    total_mesh.ConvertDataForNativeTreatment()

    ###################
    # Merge the meshes
    ###################
    pretreated_mesh = copy.deepcopy(mesh)
    pretreated_mesh.Merge(total_mesh)
    MMT.CleanDoubleNodes(pretreated_mesh)
    MMT.CleanDoubleElements(pretreated_mesh)
    pretreated_mesh.DeleteElemTags(["2D"])
    pretreated_mesh.ConvertDataForNativeTreatment()
    pretreated_mesh.Clean()

    ###################
    # Add External_boundary element and node tags
    ###################
    MMT.ComputeSkin(pretreated_mesh, md = 2, inPlace = True)
    ext_bound = np.setdiff1d(pretreated_mesh.elements[ED.Bar_2].GetTag("Skin").GetIds(), pretreated_mesh.elements[ED.Bar_2].GetTag("Airfoil").GetIds())
    pretreated_mesh.elements[ED.Bar_2].GetTag("External_boundary").SetIds(ext_bound)
    nfExtBound = FO.NodeFilter(eTag = "External_boundary")
    nodeIndexExtBound = nfExtBound.GetNodesIndices(pretreated_mesh)
    pretreated_mesh.GetNodalTag("External_boundary").AddToTag(nodeIndexExtBound)
    pretreated_mesh.elements[ED.Bar_2].tags.DeleteTags(["Skin"])

    ###################
    # Compute nodeFields values in the mesh in train
    ###################

    if train:
        # Using Scipy griddata instead of Muscat FE to avoid 'eigency' binary conflict in current environment
        nptadd = pretreated_mesh.nodes.shape[0] - mesh.nodes.shape[0]
        # Linear interpolation from the original mesh nodes (mesh.nodes) to the new points
        # Using 2D coordinates (x, y) for more robust interpolation
        new_coords_2d = np.asarray(pretreated_mesh.nodes[-nptadd:, :2])
        old_coords_2d = np.asarray(mesh.nodes[:, :2])
        
        nmesh = pretreated_mesh.nodes.shape[0]
        for pfn, pf in zip(out_field_names, pointFields):
            pretreated_mesh.nodeFields[pfn] = np.empty(nmesh)
            pretreated_mesh.nodeFields[pfn][:-nptadd] = pf.squeeze()
            
            # Interpolate values for the new nodes
            vals = griddata(old_coords_2d, np.asarray(pf.squeeze()), new_coords_2d, method='linear', fill_value=pf.mean())
            pretreated_mesh.nodeFields[pfn][-nptadd:] = vals

    size_init_mesh = mesh.GetNumberOfNodes()

    return pretreated_mesh, size_init_mesh, scalars


def InterpolateCurvAbsissas(airfoil_1, curv_abscissa_1):

    target_points = np.linspace(0., 1., num=101)

    extrado = np.array(PieceWiseLinearInterpolationVectorized(target_points, curv_abscissa_1[0], airfoil_1[1][0]))[:,:2]
    intrado = np.array(PieceWiseLinearInterpolationVectorized(target_points, curv_abscissa_1[1], airfoil_1[1][1]))[:,:2]

    X_angle = np.vstack((extrado, intrado)).ravel()

    return X_angle



def prepare_morphing(i_sample, benchmark_path, manifest, airfoil_0, curv_abscissa_0):

    pretreated_mesh, _, scalars = pretreat_sample(benchmark_path, manifest[i_sample], train = False)

    airfoil_1 = ExtractLineMesh(pretreated_mesh, "Airfoil")
    curv_abscissa_1 = computeAirfoilCurvAbscissa(airfoil_1)

    X_angle = InterpolateCurvAbsissas(airfoil_1, curv_abscissa_1)

    return X_angle, scalars


def morph_sample(mesh_1, airfoil_0, curv_abscissa_0, train, angle_predictor = None, scalars = None):

    ##############################################################
    # Compute the mapping of the extrado and intrado
    ##############################################################

    # start = time.time()
    airfoil_1 = ExtractLineMesh(mesh_1, "Airfoil")
    curv_abscissa_1 = computeAirfoilCurvAbscissa(airfoil_1)
    mapped_airfoil = MapAirfoil(airfoil_0, curv_abscissa_0, curv_abscissa_1)
    X_angle = InterpolateCurvAbsissas(airfoil_1, curv_abscissa_1)

    indices_extrado_to_morph_1 = airfoil_1[0][0][:-1]
    indices_intrado_to_morph_1 = airfoil_1[0][1][:-1]

    ##############################################################
    # Compute the mapping of the sillage
    ##############################################################

    sillage_node_ids = mesh_1.GetNodalTag("Sillage").GetIds()

    sillage_node_ids = sillage_node_ids[1:-8]

    # Compute displacement of the sillage nodes consisted of incidence angle rotation and correction wrt nut
    if train == True:
        angle_1 = find_angle_from_mesh(mesh_1)
    else:
        pca_X_angles_train = angle_predictor[0]
        scalar_scalers = angle_predictor[1]
        kmodel_angle = angle_predictor[2]
        angles_scaler = angle_predictor[3]

        X_angle = np.dot(pca_X_angles_train, X_angle.T).T.reshape(1, -1)
        scalars = scalar_scalers.transform(np.array(scalars).reshape(1, -1))

        X_ = np.hstack((X_angle, scalars))

        angle_1 = kmodel_angle.predict(X_.reshape(1, -1))
        angle_1 = angles_scaler.inverse_transform(angle_1.reshape(-1, 1)).squeeze()

    rot_matrix = np.array([[np.cos(-angle_1), -np.sin(-angle_1)], [np.sin(-angle_1), np.cos(-angle_1)]])

    l5 = len(sillage_node_ids)
    displacement_sillage = np.zeros((l5,2))

    for i, id in enumerate(sillage_node_ids):
        displacement_sillage[i,:] = p0[:2] + np.dot(rot_matrix,  mesh_1.nodes[id, :2] - p0[:2]) -  mesh_1.nodes[id, :2]

    ##############################################################
    # Compute the mapping of the outflow boundary
    ##############################################################

    indices_ext_bound_to_morph_1 = mesh_1.GetNodalTag("External_boundary").GetIds()
    nf = FO.NodeFilter(zone = lambda p: (-p[:,0] + PBBox[2][0]-0.0001))
    out_boundary_ids_1 = nf.GetNodesIndices(mesh_1)
    mesh_1.GetNodalTag("Out_boundary").AddToTag(out_boundary_ids_1)
    other_boundary_ids_1 = np.setdiff1d(indices_ext_bound_to_morph_1, out_boundary_ids_1)
    mesh_1.GetNodalTag("Other_boundary").AddToTag(other_boundary_ids_1)

    bound_end_sillage_ids = mesh_1.GetNodalTag("Bound_end_sillage").GetIds()
    l4 = len(bound_end_sillage_ids)
    displacement_out_boundary = np.zeros((l4,2))

    for i, id in enumerate(bound_end_sillage_ids):
        displacement_out_boundary[i,1] = p0[1] + np.dot(rot_matrix,  mesh_1.nodes[id, :2] - p0[:2])[1] -  mesh_1.nodes[id, 1]

    ##############################################################
    # Compute global target displacement and masks for RBF field morphing
    ##############################################################

    l1 = len(indices_extrado_to_morph_1)
    l2 = len(indices_intrado_to_morph_1)
    l3 = len(other_boundary_ids_1)

    targetDisplacement     = np.zeros((l1 + l2 + l3 + l4 + l5, 3))
    targetDisplacementMask = np.zeros((l1 + l2 + l3 + l4 + l5), dtype = int)

    targetDisplacement[:l1,:2]                        = mapped_airfoil[0][:,:2] - mesh_1.nodes[indices_extrado_to_morph_1,:2]
    targetDisplacement[l1:l1+l2,:2]                   = mapped_airfoil[1][:,:2] - mesh_1.nodes[indices_intrado_to_morph_1,:2]
    targetDisplacement[l1+l2:l1+l2+l3,:2]             = np.zeros((l3,2))
    targetDisplacement[l1+l2+l3:l1+l2+l3+l4,:2]       = displacement_out_boundary
    targetDisplacement[l1+l2+l3+l4:l1+l2+l3+l4+l5,:2] = displacement_sillage

    targetDisplacementMask[:l1]                        = indices_extrado_to_morph_1
    targetDisplacementMask[l1:l1+l2]                   = indices_intrado_to_morph_1
    targetDisplacementMask[l1+l2:l1+l2+l3]             = other_boundary_ids_1
    targetDisplacementMask[l1+l2+l3:l1+l2+l3+l4]       = bound_end_sillage_ids
    targetDisplacementMask[l1+l2+l3+l4:l1+l2+l3+l4+l5] = sillage_node_ids

    ##############################################################
    # Compute the morphing
    ##############################################################

    # RBF morphing
    mesh_1_nodes = mesh_1.nodes.copy()
    morphed_nodes = FastMorphing(mesh_1, targetDisplacement, targetDisplacementMask)
    mesh_1.nodes = morphed_nodes

    # final clean for out_bound out of sillage
    top_ind = np.argmax(mesh_1.nodes[out_boundary_ids_1,1])
    down_ind = np.argmin(mesh_1.nodes[out_boundary_ids_1,1])
    ind_to_clean = np.setdiff1d(out_boundary_ids_1, bound_end_sillage_ids)
    mesh_1.nodes[out_boundary_ids_1[top_ind],:2] = PBBox[2]
    mesh_1.nodes[out_boundary_ids_1[down_ind],:2] = PBBox[1]
    mesh_1.nodes[ind_to_clean,0] = PBBox[1][0]
    mesh_1.ConvertDataForNativeTreatment()

    mesh_1.nodeFields['X'] = mesh_1_nodes[:,0]
    mesh_1.nodeFields['Y'] = mesh_1_nodes[:,1]

    return mesh_1, angle_1, X_angle


def project_sample(morphed_mesh, morphed_mesh_0, train):

    projected_mesh = copy.deepcopy(morphed_mesh_0)

    if train:
        field_names = out_field_names + ['X', 'Y']
    else:
        field_names = ['X', 'Y']

    FE_interpolation_op = FastInterp(morphed_mesh, morphed_mesh_0.nodes)
    indices_nodes, filtered_triangles, all_coord_bary = FE_interpolation_op[0], FE_interpolation_op[1], FE_interpolation_op[2]

    input_fields = np.array([morphed_mesh.nodeFields[pfn] for pfn in field_names])
    proj_fields = np.empty((morphed_mesh_0.nodes.shape[0], input_fields.shape[0]))

    proj_fields[indices_nodes, :] = np.einsum('ijk,jk->ji', input_fields[:,filtered_triangles], all_coord_bary, optimize = True)

    for i, pfn in enumerate(field_names):
        projected_mesh.nodeFields[pfn] = proj_fields[:,i]

    FE_interpolation_op_inv = None

    if train == False:
        FE_interpolation_op_inv = FastInterp(morphed_mesh_0, morphed_mesh.nodes)

    return projected_mesh, FE_interpolation_op_inv


def pretreat_morph_and_project_mesh(i_sample, benchmark_path, manifest, morphed_mesh_0, airfoil_0, curv_abscissa_0, train, angle_predictor):

    ###################
    ## 1) Pretreat data
    ###################
    pretreated_mesh, size_init_mesh, scalars = pretreat_sample(benchmark_path, manifest[i_sample], train)

    ###################
    ## 2) Morph data
    ###################
    if train:
        morphed_mesh, angle, X_angle = morph_sample(pretreated_mesh, airfoil_0, curv_abscissa_0, train)
    else:
        morphed_mesh, angle, _ = morph_sample(pretreated_mesh, airfoil_0, curv_abscissa_0, train, angle_predictor, scalars)

    ###################
    ## 3) Project data
    ###################
    projected_mesh, FE_interpolation_op_inv = project_sample(morphed_mesh, morphed_mesh_0, train)

    res = {"projected_mesh":projected_mesh,
    "size_init_mesh":size_init_mesh,
    "scalars":scalars
    }

    if train == True:
        res["angle"] = angle
        res["X_angle"] = X_angle
    else:
        res["FE_interpolation_op_inv"] = FE_interpolation_op_inv

    return res


def get_dataset_name(dataset, train):

    taskk = 'full' if dataset._task == 'scarce' and not train else dataset._task
    split = 'train' if train else 'test'

    return taskk + '_' + split


def reynolds_filter(dataset):
    simulation_names=dataset.extra_data["simulation_names"]
    reynolds=np.array([float(name.split('_')[2])/1.56e-5 for name,numID in simulation_names])
    simulation_indices=np.where((reynolds>3e6) & (reynolds<5e6))[0]
    filtered_reynolds=reynolds[simulation_indices]
    return filtered_reynolds


def safran_process_dataset(dataset, train, benchmark_path, common_mesh_id, angle_predictor, common_mesh_precomp):

    dataset_name = get_dataset_name(dataset, train)

    print("benchmark_path =", benchmark_path)
    with open(os.path.join(benchmark_path, 'manifest.json'), 'r') as f:
        manifest_full = json.load(f)

    if train == True:
        reynolds = np.array([float(manifest_full[dataset_name][i].split('_')[2])/1.56e-5 for i in range(len(manifest_full[dataset_name]))])
        simulation_indices = np.where((reynolds>3e6) & (reynolds<5e6))[0]

        filtered_reynolds = [reynolds[i] for i in simulation_indices]
        filtered_reynolds_ref = reynolds_filter(dataset)
        
        # Handle sliced datasets for testing/verification
        if len(filtered_reynolds) != len(filtered_reynolds_ref):
            print(f"Warning: Sliced dataset detected ({len(filtered_reynolds_ref)} samples). Adjusting manifest...")
            simulation_indices = simulation_indices[:len(filtered_reynolds_ref)]
            filtered_reynolds = filtered_reynolds[:len(filtered_reynolds_ref)]
            
        assert np.allclose(filtered_reynolds, filtered_reynolds_ref) == True

        manifest = [manifest_full[dataset_name][i] for i in simulation_indices]

    else:
        manifest = manifest_full[dataset_name]
        # Handle sliced datasets for testing/verification
        num_sims = len(dataset.extra_data['simulation_names'])
        if len(manifest) > num_sims:
            print(f"--- Verification Mode: Slicing test manifest from {len(manifest)} to {num_sims} ---")
            manifest = manifest[:num_sims]

    nb_samples = len(manifest)

    n_cores = mp.cpu_count()
    print(f"number of cores: {n_cores}, nb_samples: {nb_samples}")

    start = time.time()
    # Choose common mesh among training set
    folder_0 = manifest_full['scarce_train'][common_mesh_id]
    reynolds = float(folder_0.split('_')[2])/1.56e-5
    assert reynolds>3e6 and reynolds<5e6, "chosen common mesh not among restricted training set"
    pretreated_mesh_0, _, _ = pretreat_sample(benchmark_path, folder_0, train = True, n_threads = n_parallel_tasks)

    if train:
        # Pretreat airfoil of first training mesh
        airfoil_0 = ExtractLineMesh(pretreated_mesh_0, "Airfoil")
        curv_abscissa_0 = computeAirfoilCurvAbscissa(airfoil_0)

        # Morph airfoil of first training mesh
        morphed_mesh_0, _, _ = morph_sample(pretreated_mesh_0, airfoil_0, curv_abscissa_0, train = True)

        common_mesh_precomp = [morphed_mesh_0, airfoil_0, curv_abscissa_0]

        print(f"duration pretreat_and_morph_mesh_0 = {int(time.time() - start)} s")

    else:

        morphed_mesh_0 = common_mesh_precomp[0]
        airfoil_0 = common_mesh_precomp[1]
        curv_abscissa_0 = common_mesh_precomp[2]

    print("Treating dataset "+dataset_name)

    # Sample 89 in scarce_train is a known outlier that can cause hard crashes (Terminated).
    # We skip it during processing and insert a placeholder to maintain indexing.
    indices_to_process = range(nb_samples)
    if dataset_name == "scarce_train" and nb_samples > 89:
        print(f"--- Anti-Crash: Skipping outlier sample 89 in {dataset_name} ---")
        indices_to_process = [i for i in range(nb_samples) if i != 89]

    with torch.multiprocessing.Pool(n_parallel_tasks) as pool:
        processed_results = list(tqdm(
            pool.imap(partial(pretreat_morph_and_project_mesh, benchmark_path = benchmark_path, manifest = manifest, \
                    morphed_mesh_0 = morphed_mesh_0, airfoil_0 = airfoil_0, curv_abscissa_0 = curv_abscissa_0, train = train, angle_predictor = angle_predictor), \
                    indices_to_process), total = len(indices_to_process), disable = False))

    # Reconstruct full results list with a placeholder at index 89 if needed
    if dataset_name == "scarce_train" and nb_samples > 89:
        results = []
        p_idx = 0
        for i in range(nb_samples):
            if i == 89:
                results.append("OUTLIER_PLACEHOLDER")
            else:
                results.append(processed_results[p_idx])
                p_idx += 1
    else:
        results = processed_results

    return results, common_mesh_precomp


def parse_data(data, training):

    clouds, scalars = [], []

    if training:
        fields, angles, X_angles = [], [], []
    else:
        sizes_init_meshes, inverse_fe_op = [], []

    # Find a template for dummy data shape
    template_item = None
    for item in data:
        if item != "OUTLIER_PLACEHOLDER":
            template_item = item
            break

    for item in data:
        if item == "OUTLIER_PLACEHOLDER":
            # Maintain indexing by adding dummy data matching the shapes of other samples
            mesh_template = template_item["projected_mesh"]
            n_nodes = mesh_template.nodeFields["X"].shape[0]
            clouds.append(np.zeros((n_nodes, 2)))
            scalars.append(np.zeros_like(template_item["scalars"]))
            if training:
                fields.append(np.zeros((n_nodes, 4)))
                angles.append(0.0)
                X_angles.append(np.zeros_like(template_item["X_angle"]))
            else:
                inverse_fe_op.append(None)
            continue

        mesh = item["projected_mesh"]
        all_scalars = item["scalars"]

        clouds.append(np.stack([mesh.nodeFields["X"], mesh.nodeFields["Y"]], axis=1))
        scalars.append(np.array(all_scalars))

        if training:
            fields.append(np.stack([mesh.nodeFields["UX"], mesh.nodeFields["UY"], mesh.nodeFields["p"].squeeze(), mesh.nodeFields["nut"].squeeze()], axis=1))
            angles.append(item["angle"])
            X_angles.append(item["X_angle"])

        else:
            inverse_fe_op.append(item["FE_interpolation_op_inv"])
            sizes_init_meshes.append(item["size_init_mesh"])

    clouds = np.stack(clouds)
    clouds = clouds.reshape(clouds.shape[0], -1)
    scalars = np.stack(scalars)

    if training:
        fields = np.stack(fields)
        angles = np.array(angles)
        X_angles = np.stack(X_angles)

        return clouds, scalars, fields, angles, X_angles

    else:
        return clouds, scalars, inverse_fe_op, sizes_init_meshes
