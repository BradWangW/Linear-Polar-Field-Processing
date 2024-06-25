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
from Auxiliary import *


# Increase the recursion limit
np.set_printoptions(threshold=np.inf)


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
    for f in tqdm(F, desc='Constructing extended mesh 1/3'):
        f_f = []
        
        for v in f:
            extended_index = len(V_extended)
            
            V_map[v].append(extended_index)
            
            # Extended vertices
            V_extended.append(V[v].tolist())
            f_f.append(extended_index)
        
        # Face-faces
        F_f.append(f_f)
            
        # Twin edges
        E_twin += [
            [f_f[0], f_f[1]], [f_f[1], f_f[2]], [f_f[2], f_f[0]]
        ]
            
    # Edge-faces
    for e in tqdm(E, desc='Constructing extended mesh 2/3'):
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
        v1 = V[v].copy()
        
        faces = np.array(F_f)[neighbours]
        # Sort by the angle between the centroid of the face and the vertex
        v2s = np.mean(np.array(V_extended)[faces], axis=1).copy()
        
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
    for v, extended_indices in tqdm(V_map.items(), desc='Constructing extended mesh 3/3'):
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
        E_comb += [[sorted_indices[i], sorted_indices[(i+1) % len(sorted_indices)]] for i in range(len(sorted_indices))]
        
    # Return np arrays for everything but the vertex-faces as they have variable length
    # print('Extended mesh constructed. Lengths: ', len(V_extended), len(E_twin), len(E_comb), len(F_f), len(F_e), len(F_v))
    return (np.array(V_extended), np.array(E_twin), np.array(E_comb), 
            np.array(F_f), np.array(F_e), F_v)


def construct_d1(E_twin, E_comb, F_f, F_e, F_v):
    '''
        Compute the first derivative matrix of the face-edge operator.
    '''
    E_extended = np.concatenate([E_twin, E_comb])
    
    d1 = lil_matrix((len(F_f) + len(F_e) + len(F_v), len(E_extended)))
    
    # The edges of face faces are oriented f[0] -> f[1] -> f[2], 
    # so F_f[:, i:i+1%3] appears exactly the same in E_twin, 
    # thus index-searching is enough.
    for i in range(3):
        # Find the indices of the face edges in the edge list
        indices = find_indices(E_extended, np.stack([F_f[:, i], F_f[:, (i+1)%3]], axis=1))
        
        d1[np.arange(len(F_f)), indices] = 1
    
    # The edges of the edge faces are not oriented as the faces are,
    # so we check the orientation differences
    for i, f in enumerate(F_e):
        candidate_indices = np.where(np.all(np.isin(E_extended, f), axis=1))[0]
        E_f = E_extended[candidate_indices]
        # print(f, candidate_indices, E_f.shape)
        
        for index, e in zip(candidate_indices, E_f):
            # print(f, e)
            if (np.where(f == e[0])[0] == np.where(f == e[1])[0] - 1) or (f[-1] == e[0] and f[0] == e[1]):
                # print('normal')
                d1[len(F_f) + i, index] = 1
            elif np.where(f == e[0])[0] == np.where(f == e[1])[0] + 1 or (f[-1] == e[1] and f[0] == e[0]):
                # print('reversed')
                d1[len(F_f) + i, index] = -1
            else:
                raise ValueError(f'The edge {e} is not in the face {f}, or the edge face is wrongly defined.')
    
    # The edges of the vertex faces are counter-clockwise oriented, 
    # so similar to the face faces, index-searching is enough.
    for i, f in tqdm(enumerate(F_v), desc='Constructing d1', total=len(F_v)):
        # Find the indices of the combinatorial edges in the edge list
        E_f = np.array([
            [f[i], f[(i+1) % len(f)]] for i in range(len(f))
        ])
        
        indices = find_indices(E_extended, E_f)
        
        d1[len(F_f) + len(F_e) + i, indices] = 1
        
    # Debug
    # print('d1 constructed: ', d1.todense())
    # print(np.sum(d1.todense(), axis=1))

    return d1.tocsr()
    

def compute_thetas(VEF_extended, singularities, indices, G_V):
    
    V_extended, E_twin, E_comb, F_f, F_e, F_v = VEF_extended
    
    E_extended = np.concatenate([E_twin, E_comb])
    
    num_F = len(F_f) + len(F_e) + len(F_v)
    num_E = len(E_extended)
    
    d1_full = construct_d1(E_twin, E_comb, F_f, F_e, F_v)
    
    G_F_full = np.concatenate([np.zeros(len(F_f)), np.zeros(len(F_e)), G_V])
    
    I_F_e = np.zeros(len(F_e))
    I_F_v = np.zeros(len(F_v))
    I_F_f = np.zeros(len(F_f))
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
                print('!!!!! Dont forget the old trivial connection trial !!!!!')
                I_F_f[candidate_faces[0]] = index
            else:
                raise ValueError(f'The singularity {singularity} is not in any face.')
    
    I_F_full = np.concatenate([I_F_f, I_F_e, I_F_v])
    lhs = d1_full.tocoo()
    rhs = - G_F_full + 2 * np.pi * I_F_full
    
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
        
        Z1 = complex_projection(B1, B2, normals, V1)[0]
        Z2 = complex_projection(B1, B2, normals, V2)[0]
        
        # print(np.all(
        #     np.mod(np.angle(Z2) - np.angle(Z1) + np.pi, 2*np.pi) - np.pi == np.mod(np.arccos(
        #     ((Z1 * np.conjugate(Z2)) / 
        #     (np.abs(Z1) * np.abs(Z2))).real
        # ) + np.pi, 2*np.pi) - np.pi))
        
        # Thetas[e_f, i] = index * (np.angle(Z2) - np.angle(Z1))
        
        Thetas[e_f, i] = index * np.arccos(
            ((Z1 * np.conjugate(Z2)) / 
            (np.abs(Z1) * np.abs(Z2))).real
        )
        
        # print(Thetas)
        # print(np.sum(Thetas))
        # print(d1_full.dot(Thetas[:, i]), - G_F_full + 2 * np.pi * I_F_full)
        
        # Add constraints (trivial connection)
        mask_removed_f = np.ones(num_F, dtype=bool)
        mask_removed_f[f_singular] = False
        
        mask_removed_E[e_f, i] = False
        
        # d1 = d1_full[mask_removed_f]
        # d1 = d1[:, mask_removed_E[:, i]]
        # lhs = d1.tocoo()
        
        # G_F = G_F_full[mask_removed_f]
        # I_F = I_F_full[mask_removed_f]
        # rhs = - G_F + 2 * np.pi * I_F
        
        # Debug
        # print(G_F)
        # print(I_F)
        # print(rhs)
        
        # constraints.append({
        #     'type': 'eq', 'fun': lambda x, i=i: lhs.dot(x[i*(num_E-3):(i+1)*(num_E-3)]) - rhs
        #     })
        
        constraints.append({
            'type': 'eq', 'fun': lambda x, i=i: lhs.dot(x[i*num_E:(i+1)*num_E]) - rhs
            })
        
        constraints.append({
            'type': 'eq', 'fun': lambda x, i=i: x[i*num_E:(i+1)*num_E][e_f] - Thetas[e_f, i]
        })
        
        Thetas[:, i], _, _, r1norm = lsqr(lhs, rhs)[:4]
        print('Error for theta: ', r1norm)
        
    # Compute the thetas for the rest of the faces
    # def objective(thetas):
    #     # Return ||sum_i thetas_i||^2
    #     # return np.linalg.norm(np.sum(thetas.reshape(num_E-3, -1), axis=1))**2
    #     return np.linalg.norm(thetas)**2
    
    # thetas_init = np.zeros(num_E * len(F_singular))
    
    # print('Optimising thetas...')
    # opt = {'disp':True,'maxiter':10}
    # print(constraints)
    # result = minimize(objective, thetas_init, constraints=constraints, options=opt) 
    # # result = minimize(objective, thetas_init, constraints=constraints) 
    
    # print(result.x)
    
    # Thetas = result.x.reshape(num_E, -1)
    
    # print(thetas)
    # print(d1_full.dot(thetas), - G_F_full + 2 * np.pi * I_F_full)
    
    # print(max(thetas), min(thetas))
    # print(np.sum(thetas))
    # print(np.sum(thetas > 0.5), np.sum(thetas < 0.01))
    # print(E_twin)
    # print(Thetas)
    return Thetas


def compute_face_pair_rotation(VEF_extended):
    V_extended, _, E_comb, F_f, F_e, _ = VEF_extended
    
    rotations = np.zeros(E_comb.shape[0])
    # The ith element in F_e and E represent the same edge
    for f_e in F_e:
        # Recall each f_e is formed as v1 -> v2 -> v2 -> v1
        # so that v1 -> v2 is a twin edge and v1 -> v1 is a combinatorial edge
        e1_comb = np.all(np.sort(E_comb, axis=1) == np.sort(f_e[[0, 3]]), axis=1)
        e2_comb = np.all(np.sort(E_comb, axis=1) == np.sort(f_e[[1, 2]]), axis=1)

        vec_e = V_extended[f_e[1]] - V_extended[f_e[0]]
        
        f1_f = np.where(np.any(np.isin(F_f, f_e[0]), axis=1))[0]
        f2_f = np.where(np.any(np.isin(F_f, f_e[3]), axis=1))[0]
        
        B1, B2, normals = compute_planes_F(V_extended, F_f[[f1_f, f2_f]][:, 0])
        f1 = F_f[f1_f]
        f2 = F_f[f2_f]
        # print(B1, B2)
    
        U = complex_projection(B1, B2, normals, vec_e[None, :])
        # print(U)
        U[U > 0] = U[U > 0] / np.abs(U[U > 0])
        rotation = np.angle(U[1]) - np.angle(U[0])
        # rotation = np.arccos((U[0] * np.conjugate(U[1])).real)
        # print(np.mod(rotation + np.pi, 2*np.pi) - np.pi)
        
        if np.all(E_comb[e1_comb] == f_e[[0, 3]]):
            rotations[e1_comb] = rotation
        elif np.all(E_comb[e1_comb] == f_e[[3, 0]]):
            rotations[e1_comb] = -rotation
        else:
            raise ValueError(f'{E_comb[e1_comb]} and {f_e[[0, 3]]} do not match.')
        
        if np.all(E_comb[e2_comb] == f_e[[1, 2]]):
            rotations[e2_comb] = rotation
        elif np.all(E_comb[e2_comb] == f_e[[2, 1]]):
            rotations[e2_comb] = -rotation
        else:
            raise ValueError(f'{E_comb[e2_comb]} and {f_e[[1, 2]]} do not match.')
        
    return np.mod(rotations + np.pi, 2*np.pi) - np.pi
        
        

def reconstruct_corners_from_thetas(v_init, z_init, VEF_extended, thetas, face_pair_rotations=None):
    V_extended, E_twin, E_comb, _, _, _ = VEF_extended
    
    E_extended = np.concatenate([E_twin, E_comb])
    
    thetas[len(E_twin):, 0] += face_pair_rotations
    thetas = np.mod(thetas + np.pi, 2*np.pi) - np.pi
    print(thetas.shape)
    print(len(E_extended))
    
    A = lil_matrix((len(E_extended) + 1, len(V_extended)), dtype=complex)
    A[np.arange(len(E_extended)), E_extended[:, 0]] = np.exp(1j * thetas)
    A[np.arange(len(E_extended)), E_extended[:, 1]] = -1
    A[np.arange(len(E_comb)) + len(E_twin), E_comb[:, 0]] *= np.exp(1j * face_pair_rotations)
    A[-1, v_init] = 1
    # print(A)
    A = A.tocoo()
    
    b = np.zeros(len(E_extended) + 1, dtype=complex)
    b[-1] = z_init
    
    # A = lil_matrix((len(E_extended), len(V_extended)), dtype=complex)
    # A[np.arange(len(E_extended)), E_extended[:, 0]] = np.exp(1j * thetas)
    # A[np.arange(len(E_extended)), E_extended[:, 1]] = -1
    # A = A.tocoo()
    
    # b = np.zeros(len(E_extended), dtype=complex)
    
    U, _, _, r1norm = lsqr(A, b)[:4]
    print('Error for corner vectors: ', r1norm)
    
    # normalise the vectors
    U[U > 0] = U[U > 0] / np.abs(U[U > 0])
    
    print(U)
    
    return U


def reconstruct_linear_from_corners(VEF_extended, U):
    V_extended, _, _, F_f, _, _ = VEF_extended
    
    # Compute the complex representation of the vertices on their face faces
    A = lil_matrix((len(F_f)*4, len(F_f)*4), dtype=complex)
    
    for i, f in tqdm(enumerate(F_f), desc='Reconstructing linear field coefficients', total=len(F_f)):
        B1, B2, normals = compute_planes_F(V_extended, f[None, :])
        
        # Compute the complex representation of the vertices on the face face
        Zf = complex_projection(B1, B2, normals, V_extended[f])[0]
        Uf = U[f]
        
        prod = Zf * Uf
        
        # A[4*i:4*(i+1), 4*i:4*(i+1)] = np.array([
        #     [prod[0].imag, prod[0].real, Uf[0].imag, Uf[0].real],
        #     [prod[1].imag, prod[1].real, Uf[1].imag, Uf[1].real],
        #     [prod[2].imag, prod[2].real, Uf[2].imag, Uf[2].real], 
        #     [singularity_proj[i], singularity_proj[i], 1, 1]
        # ])
        
        A[4*i:4*(i+1), 4*i:4*(i+1)] = np.array([
            [prod[0].imag, prod[0].real, Uf[0].imag, Uf[0].real],
            [prod[1].imag, prod[1].real, Uf[1].imag, Uf[1].real],
            [prod[2].imag, prod[2].real, Uf[2].imag, Uf[2].real], 
            [1, 0, 0, 0]
        ])
        
        # A[3*i:3*(i+1), 4*i:4*(i+1)] = np.array([
        #     [prod[0].imag, prod[0].real, Uf[0].imag, Uf[0].real],
        #     [prod[1].imag, prod[1].real, Uf[1].imag, Uf[1].real],
        #     [prod[2].imag, prod[2].real, Uf[2].imag, Uf[2].real]
        # ])
    
    A = A.tocoo()
    
    b = np.zeros(len(F_f)*4, dtype=complex)
    b[np.arange(3, len(F_f)*4, 4)] = 1
    
    # b = np.zeros(len(F_f)*3, dtype=complex)
    # print(b)
            
    result, _, _, r1norm = lsqr(A, b)[:4]
    
    print('Error for linear field coefficients: ', r1norm)
    
    result = result.reshape(4, -1)
    
    coeffs = np.stack([result[0] + 1j * result[1], result[2] + 1j * result[3]], axis=1)

    return coeffs


def construct_linear_field(V, F, singularities, indices, v_init, z_init):
    
    E = obtain_E(F)
    
    G_V = compute_G_V(V, E, F)
    
    VEF_extended = extended_mesh(V, E, F)
    
    thetas = compute_thetas(VEF_extended, singularities, indices, G_V)
    
    face_pair_rotations = compute_face_pair_rotation(VEF_extended)
    
    Z = reconstruct_corners_from_thetas(v_init, z_init, VEF_extended, thetas, face_pair_rotations)
    
    coeffs = reconstruct_linear_from_corners(VEF_extended, Z)
    
    def linear_field(posis):
        B1 = np.zeros((len(posis), 3))
        B2 = np.zeros((len(posis), 3))
        normals = np.zeros((len(posis), 3))
        
        # From the way the face faces are constructed,
        # they are ordered the same way as the faces
        faces_involved = np.array([is_in_face(V, F, posi)[0] for posi in posis])
        
        B1, B2, normals = compute_planes_F(V, F[faces_involved])
        
        Z = complex_projection(B1, B2, normals, posis, diagonal=True)
        
        vectors_complex = coeffs[faces_involved, 0] * Z + coeffs[faces_involved, 1]
        
        vectors = B1 * vectors_complex.real[:, None] + B2 * vectors_complex.imag[:, None]
            
        return vectors
        
    return linear_field, Z







if __name__ == '__main__':

    # V, F = load_off_file(os.path.join('..', 'data', 'spherers.off'))
    # singularities = np.array([
    #     0.3 * V[F[0, 0]] + 0.3 * V[F[0, 1]] + 0.4 * V[F[0, 2]],
    #     V[256]
    # ])
    # indices = [1, 1]
    # v_init = 100
    # z_init = 1
    
    
    # A minimal triangulated tetrahedron
    V = np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    ], dtype=float)
    F = np.array([
        [0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]
    ])
    # singularities = np.array([[0.2, 0.2, 0], [0.4, 0.4, 0]])
    singularities = np.array([[1/3, 1/3, -1/3], [-1, -1, 1]])
    indices = [1, 1]
    v_init = 0
    z_init = 1
    
    
    # A minimal triangulated cube
    # V = np.array([
    #     [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 1, 0]
    # ], dtype=float)
    # F = np.array([
    #     [0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4], [2, 6, 5], [3, 4, 5], [3, 5, 6], [2, 7, 6], [2, 1, 7], [1, 0, 7], [0, 3, 7], [3, 6, 7]
    # ])
    # singularities = np.array([[0.2, 0.2, 0], [1, 1, 1]])
    # indices = [1, 1]
    # v_init = 0
    # z_init = 1
    
    # A mini sphere
    # V = np.array([
    #     [1.0, 0.0, 0.0],   
    #     [0.0, 1.0, 0.0],   
    #     [0.0, 0.0, 1.0], 
    #     [0.0, -1.0, 0.0],  
    #     [0.0, 0.0, -1.0],   
    #     [-1.0, 0.0, 0.0]
    # ])
    # F = np.array([
    #     [0, 1, 2], 
    #     [0, 2, 3], 
    #     [0, 3, 4], 
    #     [0, 4, 1], 
    #     [5, 1, 2], 
    #     [5, 2, 3], 
    #     [5, 3, 4], 
    #     [5, 4, 1], 
    # ])
    # singularities = np.array([[-1/3, -1/3, -1/3], [0, 0, 1]])
    # indices = [1, 1]
    # v_init = 0
    # z_init = 1

    field, Z = construct_linear_field(V, F, singularities, indices, v_init, z_init)

    points, vectors = sample_points_and_vectors(V, F, field, num_samples=30)

    # normalise the vectors
    vectors = vectors / np.linalg.norm(vectors, axis=1)[:, None]

    ps.init()
    ps_mesh = ps.register_surface_mesh("Input Mesh", V, F)
    ps_centroids = ps.register_point_cloud("Centroids", points, enabled=True)
    
    ps_centroids.add_vector_quantity('Samples', vectors, enabled=True)
    
    ps.register_point_cloud("singularity marker", singularities, enabled=True)

    ps.show()
            
