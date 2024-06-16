import numpy as np
import scipy.linalg
from scipy.sparse import csc_matrix, coo_matrix, linalg, bmat, diags
from scipy.sparse.linalg import spsolve, splu
from tqdm import tqdm
import os
from contextlib import redirect_stdout, redirect_stderr


def accumarray(indices, values):
    output = np.zeros((np.max(indices) + 1), dtype=values.dtype)
    indFlat = indices.flatten()
    valFlat = values.flatten()
    # for index in range(indFlat.shape[0]):
    #     output[indFlat[index]] += valFlat[index]
    np.add.at(output, indFlat, valFlat)

    return output


def load_off_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the vertices and faces from the OFF file
    num_vertices, num_faces, _ = map(int, lines[1].split())

    vertices = np.array([list(map(float, line.split())) for line in lines[2:2 + num_vertices]])
    faces = np.array([list(map(int, line.split()))[1:] for line in lines[2 + num_vertices:]])

    return vertices, faces


def compute_areas_normals(vertices, faces):
    face_vertices = vertices[faces]

    # Compute vectors on the face
    vectors1 = face_vertices[:, 1, :] - face_vertices[:, 0, :]
    vectors2 = face_vertices[:, 2, :] - face_vertices[:, 0, :]

    # Compute face normals using cross product
    normals = np.cross(vectors1, vectors2)
    faceAreas = 0.5 * np.linalg.norm(normals, axis=1)

    normals /= (2.0 * faceAreas[:, np.newaxis])
    return normals, faceAreas


def visualize_boundary_Edges(ps_mesh, vertices, boundEdges):
    boundVertices = vertices[boundEdges]

    boundVertices = boundVertices.reshape(2 * boundVertices.shape[0], 3)
    curveNetIndices = np.arange(0, boundVertices.shape[0])
    curveNetIndices = curveNetIndices.reshape(int(len(curveNetIndices) / 2), 2)
    ps_net = ps.register_curve_network("boundary edges", boundVertices, curveNetIndices)

    return ps_net


def createEH(edges, halfedges):
    # Create dictionaries to map halfedges to their indices
    halfedges_dict = {(v1, v2): i for i, (v1, v2) in enumerate(halfedges)}
    # reversed_halfedges_dict = {(v2, v1): i for i, (v1, v2) in enumerate(halfedges)}

    EH = np.zeros((len(edges), 2), dtype=int)

    for i, (v1, v2) in enumerate(edges):
        # Check if the halfedge exists in the original order
        if (v1, v2) in halfedges_dict:
            EH[i, 0] = halfedges_dict[(v1, v2)]
        # Check if the halfedge exists in the reversed order
        if (v2, v1) in halfedges_dict:
            EH[i, 1] = halfedges_dict[(v2, v1)]

    return EH


def compute_edge_list(vertices, faces, sortBoundary=False):
    halfedges = np.empty((3 * faces.shape[0], 2))
    for face in range(faces.shape[0]):
        for j in range(3):
            halfedges[3 * face + j, :] = [faces[face, j], faces[face, (j + 1) % 3]]

    edges, firstOccurence, numOccurences = np.unique(np.sort(halfedges, axis=1), axis=0, return_index=True,
                                                     return_counts=True)
    edges = halfedges[np.sort(firstOccurence)]
    edges = edges.astype(int)
    halfedgeBoundaryMask = np.zeros(halfedges.shape[0])
    halfedgeBoundaryMask[firstOccurence] = 2 - numOccurences
    edgeBoundMask = halfedgeBoundaryMask[np.sort(firstOccurence)]

    boundEdges = edges[edgeBoundMask == 1, :]
    boundVertices = np.unique(boundEdges).flatten()

    # EH = [np.where(np.sort(halfedges, axis=1) == edge)[0] for edge in edges]
    # EF = []

    EH = createEH(edges, halfedges)
    EF = np.column_stack((EH[:, 0] // 3, (EH[:, 0] + 2) % 3, EH[:, 1] // 3, (EH[:, 1] + 2) % 3))

    if (sortBoundary):
        loop_order = []
        loopEdges = boundEdges.tolist()
        current_node = boundVertices[0]  # Start from any node
        visited = set()
        while True:
            loop_order.append(current_node)
            visited.add(current_node)
            next_nodes = [node for edge in loopEdges for node in edge if
                          current_node in edge and node != current_node and node not in visited]
            if not next_nodes:
                break
            next_node = next_nodes[0]
            loopEdges = [edge for edge in loopEdges if
                          edge != (current_node, next_node) and edge != (next_node, current_node)]
            current_node = next_node
            current_node = next_node

        boundVertices = np.array(loop_order)

    return halfedges, edges, edgeBoundMask, boundVertices, EH, EF


def compute_angle_defect(vertices, faces, boundVertices):
    # TODO: complete
    angles = np.zeros(faces.shape)
    for i, face in tqdm(enumerate(faces), 
                        desc='Computing angle defect', 
                        total=faces.shape[0]):
        v1, v2, v3 = vertices[face]
        angles[i, 0] = np.arccos(np.dot(v2 - v1, v3 - v1) / 
                                 (np.linalg.norm(v2 - v1) * np.linalg.norm(v3 - v1)))
        angles[i, 1] = np.arccos(np.dot(v1 - v2, v3 - v2) / 
                                 (np.linalg.norm(v1 - v2) * np.linalg.norm(v3 - v2)))
        angles[i, 2] = np.pi - angles[i, 0] - angles[i, 1]
    
    angles = accumarray(
        faces, 
        angles
    )
    boundVerticesMask = np.zeros(vertices.shape[0])
    boundVerticesMask[boundVertices] = 1
    
    G = (2 - boundVerticesMask) * np.pi - angles
    
    # for i, vertex in tqdm(enumerate(vertices), 
    #                       desc='Computing angle defect', 
    #                       total=vertices.shape[0]):
        
    #     sum_angles = (2 - int(i in boundVertices)) * np.pi
            
    #     faces_i = faces[np.any(faces == i, axis=1)]
    #     for face in faces_i:
    #         v2, v3 = vertices[face[face != i]]
    #         angle = np.arccos(np.dot(v2-vertex, v3-vertex) / (np.linalg.norm(v2-vertex)*np.linalg.norm(v3-vertex)))
    #         sum_angles -= angle
                
    #     G[i] = sum_angles
    return G


def compute_mean_curvature_normal(vertices, faces, faceNormals, L, vorAreas):
    #TODO: complete
    MCNormal = L @ vertices / (2 * vorAreas[:, np.newaxis])
    MC = np.zeros((vertices.shape[0]))
    #-------------Vertex normals and MC--------------#
    vertexNormals = np.stack(
        [
            accumarray(faces.flatten(), np.repeat(faceNormals[:, 0], 3)), 
            accumarray(faces.flatten(), np.repeat(faceNormals[:, 1], 3)), 
            accumarray(faces.flatten(), np.repeat(faceNormals[:, 2], 3))
        ], 
        axis=1
    )
    
    vertexNormals = vertexNormals/np.linalg.norm(vertexNormals, axis=1)[:, np.newaxis]
    MC = np.sign(np.sum(MCNormal*vertexNormals, axis=1)) * np.linalg.norm(MCNormal, axis=1)
    #------------------------------------------------#
    return MCNormal, MC, vertexNormals


def compute_laplacian(vertices, faces, edges, edgeBoundMask, EF, onlyArea=False):
    #TODO: complete

    #-------------Voroni area--------------#
    faceAreas = np.zeros((faces.shape[0]))
    for i, face in tqdm(enumerate(faces),
                        desc='Computing face areas',
                        total=faces.shape[0]):
        v1, v2, v3 = vertices[face]
        faceAreas[i] = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
        
    vorAreas = accumarray(
        faces, 
        np.repeat(faceAreas, 3)
    )
    vorAreas /= 3
    
    # vorAreas = np.zeros((vertices.shape[0]))
    # for i, vertex in tqdm(enumerate(vertices), 
    #                       desc='Computing Voroni area', 
    #                       total=vertices.shape[0]):
            
    #     faces_i = faces[np.any(faces == i, axis=1)]
    #     for face in faces_i:
    #         v2, v3 = vertices[face[face != i]]
    #         area = 0.5 * np.linalg.norm(np.cross(v2 - vertex, v3 - vertex))
            
    #         vorAreas[i] += area
    # vorAreas /= 3
    
    if onlyArea:
        return vorAreas
    #--------------------------------------#
    
    #-------------Cot-weight Laplacian--------------#
    row = []
    col = []
    data = []
    for i in range(edges.shape[0]):
        for j in range(2):
            row.append(i)
            col.append(edges[i, j])
            data.append((-1)**(j+1))
    
    # Sparse differential matrix
    d0 = csc_matrix((data, (row, col)), shape=(edges.shape[0], vertices.shape[0]))
    
    diag_data = np.zeros((edges.shape[0]))
    for i in tqdm(range(edges.shape[0]), 
                  desc='Computing cot-weight Laplacian'):
        v1, v2 = vertices[edges[i, :]]
        
        face_j, j, face_l, l = EF[i]
        
        vj = vertices[faces[face_j, j]]
        anglej = np.arccos(np.dot(v1-vj, v2-vj) / 
                           (np.linalg.norm(v1-vj)*np.linalg.norm(v2-vj)))
        diag_data[i] += 1/np.tan(anglej)
        
        if edgeBoundMask[i]==0:
            vl = vertices[faces[face_l, l]]
            
            d1 = v1-v2
            d2 = v1-vl
            
            # Avoid division by zero
            while np.linalg.norm(d1)==0 or np.linalg.norm(d2)==0:
                d1 += 1e-6
                d2 += 1e-6
                
            anglel = np.arccos(np.dot(d1, d2) / 
                            (np.linalg.norm(d1)*np.linalg.norm(d2)))
            diag_data[i] += 1/np.tan(anglel)
            
    W = diags(diag_data/2, format='csc')
    
    L = d0.T @ W @ d0
    #-----------------------------------------------#
    return L, vorAreas, d0, W


def mean_curvature_flow(faces, boundVertices, currVertices, L, vorAreas, flowRate, isExplicit):
    #TODO: complete
    if isExplicit:
        v_iter = lambda v: v - flowRate * ((L @ v) / vorAreas[:, np.newaxis])
    else:
        v_iter = lambda v: splu(diags(vorAreas, format='csc') + flowRate * L).solve(vorAreas[:, np.newaxis] * v)
    
    newVertices = v_iter(currVertices)
        
    if len(boundVertices)>0:
        newVertices[boundVertices] = currVertices[boundVertices]
    else:
        # Scale the computed vertices
        with open(os.devnull, 'w') as f, redirect_stdout(f), redirect_stderr(f):
            newAreas = compute_laplacian(newVertices, faces, None, None, None, True)
            
        newVertices = newVertices * np.sqrt(np.sum(vorAreas)/np.sum(newAreas))

    
    return newVertices #stub


def compute_boundary_embedding(vertices, boundVertices, r):
    #TODO: complete
    verBound = vertices[list(boundVertices)+[boundVertices[0]]]
    lenBound = np.linalg.norm(np.diff(verBound, axis=0), axis=1)
    lenTotal = np.sum(lenBound)
    
    angle_sector = 2*np.pi * lenBound/lenTotal
    
    UVB = r * np.stack(
        [
            np.cos(np.cumsum(angle_sector)), 
            np.sin(np.cumsum(angle_sector))
        ], 
        axis=1
    )
    
    return UVB


def compute_tutte_embedding(vertices, d0, W, boundVertices, boundUV):
    #TODO: complete
    inMask = np.ones(vertices.shape[0]).astype(bool)
    inMask[boundVertices] = 0
    
    d0B = d0[:, boundVertices]
    d0I = d0[:, inMask]
    
    rhs = -d0I.T @ W @ d0B @ boundUV
    lhs = d0I.T @ W @ d0I
    
    UVI = spsolve(lhs, rhs)
    
    # UV = np.concatenate((boundUV, UVI), axis=0)
    UV = np.zeros((vertices.shape[0], 2))
    UV[boundVertices] = boundUV
    UV[inMask] = UVI
    
    return UV
