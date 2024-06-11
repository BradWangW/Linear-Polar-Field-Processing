import numpy as np
import scipy.linalg
from scipy.sparse import lil_matrix, csc_matrix, coo_matrix, linalg, bmat, diags
from scipy.sparse.linalg import spsolve, splu
from tqdm import tqdm
import os
from contextlib import redirect_stdout, redirect_stderr


def compute_planes_F(V, F):
    '''
    Compute the basis vectors and the normal of the plane of each face.
    '''
    V1 = V[F[:, 0]]
    V2 = V[F[:, 1]]
    V3 = V[F[:, 2]]
    
    # Two basis vectors of the plane
    U = V2 - V1
    V = V3 - V1
    
    # Check parallelism
    para = np.all(np.cross(U, V) == 0, axis=1)
    if np.any(para):
        raise ValueError(f'The face(s) {F[np.where(para)]} is degenerate.')
    
    # Normal vector of the plane
    normals = np.cross(V2 - V1, V3 - V1)
    normals = normals / np.linalg.norm(normals, axis=1)[:, None]
    
    return U, V, normals

def compute_projection(U, V, normals, posi):
    '''
    Compuite the projection of a point onto the planes defined by the basis vectors and the normals.
    '''
    # posi:(3), normals: (N, 3), U: (N, 3), V: (N, 3)
    posi_normal = np.dot(normals, posi[:, None]) * normals
    
    posi_plane = posi - posi_normal
    
    X = np.dot(posi_plane, U.T)
    Y = np.dot(posi_plane, V.T)
    
    return X + 1j * Y


def extended_mesh(V, E, F):
    '''
    Construct the extended mesh from the input mesh, which consists of 
        face-faces (first |F| faces), edge-faces (following |E| faces), and vertex-faces (last |V| faces), 
        twin edges (first 2|E| edges) and combinatorial edges (last {#boundary vertices + sum_{v}degree_v} edges), 
        and the corresponding vertices.
    '''
    
    V_extended = []
    E_twin = []
    E_comb = []
    F_f = []
    F_e = []
    F_v = []
    
    V_map = {v:[] for v in range(V.shape[0])}
    
    # Face-faces and twin edges
    for f in F:
        f_f = []
        
        for v in f:
            extended_index = len(V_extended)
            
            V_map[v].append(extended_index)
            
            # Extended vertices
            V_extended.append(V[v].tolist())
            f_f.append(extended_index)
            
        # Twin edges
        for i in range(3):
            E_twin.append([f_f[-i-1], f_f[-((i+1) % 3)-1]])
        
        # Face-faces
        F_f.append(f_f)
            
    # Edge-faces
    for e in E:
        extended_indices_v1 = V_map[e[0]]
        extended_indices_v2 = V_map[e[1]]
        
        # In a triangle mesh, two faces can share at most one edge, 
        # so when extracting the twin edges that encompass both extended_vertices of v1 and v2, 
        # only two twin edges will be found, such that they are the twin edges of the edge e.
        e_twin = np.array(E_twin)[
            np.all(np.isin(E_twin, extended_indices_v1 + extended_indices_v2), axis=1)
        ]
        
        # Check if the twin edges are parallel or opposite
        pairing = np.isin(e_twin, extended_indices_v1)
        
        # If parallel, reverse the second twin edge
        if np.all(pairing[0] == pairing[1]):
            F_e.append(e_twin[0].tolist() + e_twin[1, ::-1].tolist())
        # If opposite, keep the twin edges as they are
        elif np.all(pairing[0] == pairing[1][::-1]):
            F_e.append(e_twin.flatten().tolist())
        # If the twin edges are not parallel or opposite, something went wrong
        else:
            raise ValueError('Twin edges are not parallel or opposite.')
        
    # Auxiliary function for vertex-faces
    def sort_neighbours(neighbours, v):
        '''
        Sort the neighbour triangles of a vertex in counter-clockwise order.
        '''
        v1 = V[v]
        
        faces = np.array(F_f)[neighbours]
        v2s = np.mean(np.array(V_extended)[faces], axis=1)
        
        angles = np.arccos(np.sum(v1 * v2s, axis=1) / (np.linalg.norm(v1) * np.linalg.norm(v2s, axis=1)))
        
        # for neighbour in neighbours:
        #     # Compute the angles based on the centroid of the triangle
        #     face = F_f[neighbour]
        #     v2 = np.mean(np.array(V_extended)[face], axis=0)
            
        #     angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            
        #     angles.append(angle)
            
        order = np.argsort(angles)
        
        return order
    
    # Vertex-faces and combinatorial edges
    for v, extended_indices in V_map.items():
        # Since each extended_vertex belongs to only one face-face, 
        # and they are added to the extended sets in the same order,
        # the counter-clockwise order of the face-faces is preserved for extended_indices.
        neighbours = np.any(np.isin(F_f, extended_indices), axis=1)
        
        order = sort_neighbours(neighbours, v)
        
        sorted_indices = np.array(extended_indices)[order]
        
        # Vertex-faces if the vertex is not on the boundary
        if len(extended_indices) > 2:
            F_v.append(sorted_indices.tolist())
        
        # Combinatorial edges
        for i in range(len(sorted_indices)):
            E_comb.append([extended_indices[i], extended_indices[(i+1) % len(sorted_indices)]])
    
    # Return np arrays for everything but the vertex-faces as they have variable length
    return (np.array(V_extended), np.array(E_twin), np.array(E_comb), 
            np.array(F_f), np.array(F_e), F_v)
    
    
def compute_twin_edges(V_extended, E_twin, F_f, singularities, indices):
    
    U, V, normals = compute_planes_F(V_extended, F_f)
    
    for singularity, index in zip(singularities, indices):
        # Project the singularity and the extended vertices onto the face planes
        Z1 = singularity - V_extended[E_twin[:, 0]]
        Z2 = singularity - V_extended[E_twin[:, 1]]
        
        Z1_proj = compute_projection(U, V, normals, Z1)
        Z2_proj = compute_projection(U, V, normals, Z2)
        
        thetas = index * np.arccos(
            ((singularity - V_extended[E_twin[:, 0]]) * np.conjugate(singularity - V_extended[E_twin[:, 1]])) / 
            (np.linalg.norm(singularity - V_extended[E_twin[:, 0]]) * np.linalg.norm(singularity - V_extended[E_twin[:, 1]]))
        )
        
    return thetas


def compute_comb_edges(V_extended, E_comb, Fe, Fv, singularities, indices, GV):
    num_F = len(Fe + Fv)
    num_E = len(E_comb)
    Fe = np.array(Fe)
    Fv = np.array(Fv)
    
    d1_FE = lil_matrix((num_F, num_E))
    
    
    I_F = lil_matrix((num_F, 1))
    for singularity, index in zip(singularities, indices):
        
        in_Fe = np.all((V_extended[Fe[:, 0]] > singularity) * (V_extended[Fe[:, 1]] < singularity), axis=1)
        if np.any(in_Fe):
            I_F[np.where(in_Fe)[0]] = index
            continue
            
        for i, fv in enumerate(Fv):
            if V_extended[fv[0]] == singularity:
                I_F[len(Fe) + i] = index
                break
            
    
    
    
    G_F = coo_matrix([0] * len(Fe) + GV, shape=(num_F, 1))
    
    
        
        
        
    


V = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
])
E = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]])
F = np.array([[0, 1, 2], [2, 3, 0], [0, 1, 3], [1, 2, 3]])
singularities = [[1, 0, 0]]
indices = [1]

compute_planes_F(V, F)

V_extended, E_twin, E_comb, F_f, F_e, F_v = extended_mesh(V, E, F)
# print(V_extended, '\n', E_twin, '\n', E_comb, '\n', F_f, '\n', F_e, '\n', F_v)
# print(len(V_extended), len(E_twin), len(E_comb), len(F_f), len(F_e), len(F_v))

thetas = compute_twin_edges(V_extended, E_twin, [np.array([1, 0, 0])], [1], 1)
print(thetas.shape)
            
