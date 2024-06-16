import numpy as np
import scipy.linalg
import polyscope as ps
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix, coo_matrix, linalg, bmat, diags
from scipy.sparse.linalg import spsolve, splu, lsqr
from tqdm import tqdm
import os
from contextlib import redirect_stdout, redirect_stderr
from DAFunctions import load_off_file, compute_areas_normals, compute_laplacian, compute_mean_curvature_normal, compute_edge_list, compute_angle_defect
from scipy.optimize import minimize


def compute_planes_F(V, F):
    '''
    Compute the basis vectors and the normal of the plane of each face.
        Input:
            V: (N, 3) array of vertices
            F: (M, 3) array of faces
        Output:
            B1: (M, 3) array of the first basis vector of each face
            B2: (M, 3) array of the second basis vector of each face
            normals: (M, 3) array of the normal vector of each face
    '''
    V1 = V[F[:, 0]]
    V2 = V[F[:, 1]]
    V3 = V[F[:, 2]]
    
    # Two basis vectors of the plane
    B1 = V2 - V1
    B2 = V3 - V1
    
    # Check parallelism
    para = np.all(np.cross(B1, B2) == 0, axis=1)
    if np.any(para):
        raise ValueError(f'The face(s) {F[np.where(para)]} is degenerate.')
    
    # Normal vector of the plane
    normals = np.cross(B1, B2)
    normals = normals / np.linalg.norm(normals, axis=1)[:, None]
    
    return B1, B2, normals

def compute_projection(B1, B2, normals, posis):
    '''
    Compuite the projection of a point onto the planes defined by the basis vectors and the normals.
        Input:
            B1: (M, 3) array of the first basis vector of each face
            B2: (M, 3) array of the second basis vector of each face
            normals: (M, 3) array of the normal vector of each face
            posis: (N, 3) array of the points to be projected
        Output:
            X: (M, N) array of the x-coordinates of the projected points
            Y: (M, N) array of the y-coordinates of the projected points
    '''
    posis_normal = np.dot(normals, posis.T) * normals.T
    
    posis_plane = posis - posis_normal
    
    X = np.dot(B1, posis_plane.T)
    Y = np.dot(B2, posis_plane.T)
    
    return X + 1j * Y


def extended_mesh(V, E, F):
    '''
    Construct the extended mesh from the input mesh, which consists of 
        the extended vertices, twin edges and combinatorial edges,
        face-faces, edge-faces, and vertex-faces.
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
        
        # Avoid division by zero
        while np.any(np.linalg.norm(v2s, axis=1) == 0) or np.linalg.norm(v1) == 0:
            v1 += 1e-6
            v2s += 1e-6
        
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


def construct_d1(E_twin, E_comb, F_f, F_e, F_v):
    '''
        Compute the first derivative matrix of the face-edge operator.
    '''
    E = np.concatenate([E_twin, E_comb])
    
    d1 = lil_matrix((len(F_f) + len(F_e) + len(F_v), len(E)))
    
    # Supposing the faces are counter-clockwise oriented,
    for i, f in enumerate(F_f.tolist() + F_e.tolist() + F_v):
        for j, e in enumerate(E):
            if e[0] in f and e[1] in f:
                if e[0] == f[0]:
                    d1[i, j] = 1
                elif e[1] == f[0]:
                    d1[i, j] = -1
                else:
                    d1[i, j] = 0
            else:
                d1[i, j] = 0
                
    return d1.tocsr()
    

def is_in_face(V, F, posi):
    '''
        Input:
            V: (N, 3) array of vertices
            F: (M, 3) array of faces
            posi: (3,) array of the point to be checked
        Output:
            False if the point is not in any face, 
            the index of the face if the point is in a face
    '''
    _, _, normals = compute_planes_F(V, F)
    
    # Filter the faces whose plane the point is in
    is_in_plane = np.where(np.sum(normals * (posi - V[F[:, 0]]), axis=1) == 0)[0]
    
    if len(is_in_plane) == 0:
        return False
    else:
        # Check if the point is in the triangle
        candidate_faces = []
        for i in is_in_plane:
            f = F[i]
            v1 = V[f[0]]
            v2 = V[f[1]]
            v3 = V[f[2]]
            
            # Compute the barycentric coordinates
            A = np.array([
                [v1[0] - v3[0], v2[0] - v3[0]],
                [v1[1] - v3[1], v2[1] - v3[1]]
            ])
            b = np.array([
                posi[0] - v3[0],
                posi[1] - v3[1]
            ])
            
            x = np.linalg.solve(A, b)
            
            if np.all(x > 0) and np.sum(x) < 1:
                candidate_faces.append(i)
        
        if len(candidate_faces) == 1:
            return candidate_faces
        elif len(candidate_faces) > 1:
            raise ValueError('The point is in more than one face.')
        else:
            return False


def compute_thetas(VEF_extended, singularities, indices, G_V):
    
    V_extended, E_twin, E_comb, F_f, F_e, F_v = VEF_extended
    
    E_extended = np.concatenate([E_twin, E_comb])
    
    num_F = len(F_f) + len(F_e) + len(F_v)
    num_E = len(E_extended)
    
    d1_full = construct_d1(E_twin, E_comb, F_f, F_e, F_v)
    
    G_F_full = np.zeros(num_F)
    G_F_full[-len(F_v):] = G_V
    
    I_F_e = np.zeros(len(F_e))
    I_F_v = np.zeros(len(F_v))
    F_singular = []
    for singularity, index in zip(singularities, indices):
        
        # Check if the singularity is on a (original) edge, vertex, or face
        in_F_e = np.all((V_extended[F_e[:, 0]] > singularity) * (V_extended[F_e[:, 1]] < singularity), axis=1)
        in_F_v = np.all(V_extended[[f_v[0] for f_v in F_v]] == singularity, axis=1)
        
        if np.any(in_F_e):
            I_F_e[np.where(in_F_e)[0]] = index
        elif np.any(in_F_v):
            I_F_v[np.where(in_F_v)[0]] = index
            continue
        else:
            candidate_faces = is_in_face(V_extended, F_f, singularity)
            
            if candidate_faces:
                F_singular.append((candidate_faces[0], singularity, index))
            else:
                raise ValueError(f'The singularity {singularity} is not in any face.')
    
    I_F_full = np.concatenate([np.zeros(len(F_f)), I_F_e, I_F_v])
    
    
    # Compute the sets of thetas for each face singularity
    Thetas = np.zeros((num_E, len(F_singular)))
    constraints = []
    mask_removed_E = np.ones((num_E, len(F_singular)), dtype=bool)
    for i, (f_singular, singularity, index) in enumerate(F_singular):
        # Compute the thetas for the singular face
        e_f = np.all(np.isin(E_extended, F_f[f_singular]), axis=1)
        
        B1, B2, normals = compute_planes_F(V_extended, F_f[f_singular][None, :])

        V1 = singularity - V_extended[E_extended[e_f, 0]]
        V2 = singularity - V_extended[E_extended[e_f, 1]]
        
        Z1 = compute_projection(B1, B2, normals, V1)
        Z2 = compute_projection(B1, B2, normals, V2)
        
        Thetas[e_f, i] = index * np.arccos(
            ((Z1 * np.conjugate(Z2)) / 
            (np.linalg.norm(Z1) * np.linalg.norm(Z2))).real
        )[0]
        
        # Add constraints (trivial connection)
        mask_removed_f = np.ones(num_F, dtype=bool)
        mask_removed_f[f_singular] = False
        
        mask_removed_E[e_f, i] = False
        
        d1 = d1_full[mask_removed_f]
        d1 = d1[:, mask_removed_E[:, i]]
        # lhs = d1.tocoo()
        
        G_F = G_F_full[mask_removed_f]
        I_F = I_F_full[mask_removed_f]
        rhs = - G_F + 2 * np.pi * I_F
        
        constraints.append({
            'type': 'eq', 'fun': lambda x, i=i: d1.dot(x[i*(num_E-3):(i+1)*(num_E-3)]) - rhs
            })

        # thetas[f'Singularity {singularity}'][mask_removed_E[:, i]] = lsqr(lhs, rhs)[0]
        
    # Compute the thetas for the rest of the faces
    def objective(thetas):
        # Return ||sum_i thetas_i||^2
        return np.linalg.norm(np.sum(thetas.reshape(num_E-3, -1), axis=0))**2
    
    thetas_init = np.random.rand((num_E-3) * len(F_singular))
    
    result = minimize(objective, thetas_init, constraints=constraints)
    
    Thetas[mask_removed_E] = result.x
    
    thetas = np.sum(Thetas, axis=1)
        
    return thetas
        

def reconstruct_corners_from_thetas(v_init, z_init, VEF_extended, thetas):
    V_extended, E_twin, E_comb, _, _, _ = VEF_extended
    
    E_extended = np.concatenate([E_twin, E_comb])
    
    A = np.zeros((len(E_extended) + 1, len(V_extended)), dtype=complex)
    A[np.arange(len(E_extended)), E_extended[:, 0]] = np.exp(1j * thetas)
    A[np.arange(len(E_extended)), E_extended[:, 1]] = -1
    A[-1, v_init] = 1
    
    b = np.zeros(len(E_extended) + 1, dtype=complex)
    b[-1] = z_init
    
    Z = np.linalg.lstsq(A, b, rcond=None)[0]
    
    return Z



V = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1]
], dtype=float)
E = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]])
F = np.array([[0, 1, 2], [2, 3, 0], [0, 1, 3], [1, 2, 3]])
singularities = np.array([[0.8, 0.2, 0], [0.9, 0.1, 0]])
indices = [1, -1]

halfedges, E, edgeBoundMask, boundVertices, EH, EF = compute_edge_list(V, F)

_, vorAreas, _,_ = compute_laplacian(V, F, E, edgeBoundMask, EF, onlyArea=True)

angleDefect = compute_angle_defect(V, F, boundVertices)

G_V = angleDefect / vorAreas

VEF_extended = extended_mesh(V, E, F)
# print(V_extended, '\n', E_twin, '\n', E_comb, '\n', F_f, '\n', F_e, '\n', F_v)
# print(len(V_extended), len(E_twin), len(E_comb), len(F_f), len(F_e), len(F_v))

thetas = compute_thetas(VEF_extended, singularities, indices, G_V)

# print(thetas)

Z = reconstruct_corners_from_thetas(0, 1j+1, VEF_extended, thetas)

print(Z)

# if __name__ == '__main__':

#     ps.init()

#     ps_mesh = ps.register_surface_mesh("Input Mesh", V, F)
    
#     ps.register_point_cloud("singularity marker", singularities, enabled=True)

#     ps.show()
            
