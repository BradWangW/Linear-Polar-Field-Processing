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


def complex_projection(B1, B2, normals, posis, diagonal=False):
    '''
    Compuite the complex-represented projection of a point 
    onto the planes defined by the basis vectors and the normals.
        Input:
            B1: (M, 3) array of the first basis vector of each face
            B2: (M, 3) array of the second basis vector of each face
            normals: (M, 3) array of the normal vector of each face
            posi: (N, 3) array of the points to be projected
        Output:
            X: (M, N, 3) array of the x-coordinates of the projected points
            Y: (M, N, 3) array of the y-coordinates of the projected points
    '''
    if diagonal:
        Z = []
        if B1.shape[0] == posis.shape[0]:
            for i, posi in enumerate(posis):
                posi_normal = np.dot(normals[i], posi) * normals[i]
                
                posi_plane = posi - posi_normal
                
                X = np.sum(B1[i] * posi_plane)
                Y = np.sum(B2[i] * posi_plane)
                
                Z.append(X + 1j * Y)
                
            return np.array(Z)
        else:
            raise ValueError('When diagonal is True, the number of faces must be equal to the number of points.')

    else:
        X = np.zeros((len(B1), len(posis)))
        Y = np.zeros((len(B1), len(posis)))
        
        for i, posi in enumerate(posis):
            posi_normal = np.dot(normals, posi)[:, np.newaxis] * normals
            
            posi_plane = posi[np.newaxis, :] - posi_normal
            
            X[:, i] = np.sum(B1 * posi_plane, axis=1)
            Y[:, i] = np.sum(B2 * posi_plane, axis=1)
            
        Z = X + 1j * Y
            
        return Z


def obtain_E(F):
    E = np.concatenate([
        F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]
    ])
    E = np.unique(np.sort(E, axis=1), axis=0)
    
    return E


def compute_G_V(V, E, F):
    _, E, edgeBoundMask, boundVertices, _, EF = compute_edge_list(V, F)

    vorAreas = compute_laplacian(V, F, E, edgeBoundMask, EF, onlyArea=True)

    angleDefect = compute_angle_defect(V, F, boundVertices)

    # return angleDefect / vorAreas
    return angleDefect


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
    return (np.array(V_extended), np.array(E_twin), np.array(E_comb), 
            np.array(F_f), np.array(F_e), F_v)


def find_indices(A, B):
    '''
    Find the indices of rows in B as they appear in A
    '''
    # Convert A and B to tuples for easy comparison
    A_tuples = [tuple(row) for row in A]
    B_tuples = [tuple(row) for row in B]

    # Create a dictionary to map rows in A to their indices
    A_dict = {row: idx for idx, row in enumerate(A_tuples)}

    # Find the indices of rows in B as they appear in A
    indices = [A_dict[row] for row in B_tuples]

    return np.array(indices)


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
        indices = find_indices(E_twin, np.stack([F_f[:, i], F_f[:, (i+1)%3]], axis=1))
        
        d1[np.arange(len(F_f)), indices] = 1
    
    # The edges of the edge faces are not oriented as the faces are,
    # so we check the orientation differences
    for i, f in enumerate(F_e):
        candidate_indices = np.where(np.all(np.isin(E_extended, f), axis=1))[0]
        E_f = E_extended[candidate_indices]
        # print(f, candidate_indices, E_f.shape)
        
        for index, e in zip(candidate_indices, E_f):
            if np.where(f == e[0])[0] < np.where(f == e[1])[0]:
                d1[len(F_f) + i, index] = 1
            elif np.where(f == e[0])[0] > np.where(f == e[1])[0]:
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
    is_in_plane = np.where(np.abs(np.sum(normals * (posi[np.newaxis, :] - V[F[:, 0]]), axis=1)) < 1e-15)[0]
    
    if len(is_in_plane) == 0:
        print(np.min(np.abs(np.sum(normals * (posi[np.newaxis, :] - V[F[:, 0]]), axis=1))))
        raise ValueError(f'The point {posi} is not in any plane.')
    else:
        # Check if the point is in the triangle
        candidate_faces = []
        for i in is_in_plane:
            f = F[i]
            v1 = V[f[0]]
            v2 = V[f[1]]
            v3 = V[f[2]]
            
            v2v1 = v2 - v1
            v3v1 = v3 - v1
            posi_v1 = posi - v1
            
            dot00 = np.dot(v3v1, v3v1)
            dot01 = np.dot(v3v1, v2v1)
            dot02 = np.dot(v3v1, posi_v1)
            dot11 = np.dot(v2v1, v2v1)
            dot12 = np.dot(v2v1, posi_v1)
            
            inv_denom = dot00 * dot11 - dot01 * dot01
            
            u = (dot11 * dot02 - dot01 * dot12) / inv_denom
            v = (dot00 * dot12 - dot01 * dot02) / inv_denom
            
            if (u >= 0) and (v >= 0) and (u + v <= 1):
                candidate_faces.append(i)
        
        if len(candidate_faces) == 1:
            return candidate_faces
        elif len(candidate_faces) > 1:
            raise ValueError(f'The point {posi} is in more than one face.')
        else:
            raise ValueError(f'The point is {posi} not in any face.') 


def compute_thetas(VEF_extended, singularities, indices, G_V):
    
    V_extended, E_twin, E_comb, F_f, F_e, F_v = VEF_extended
    
    E_extended = np.concatenate([E_twin, E_comb])
    
    num_F = len(F_f) + len(F_e) + len(F_v)
    num_E = len(E_extended)
    
    d1_full = construct_d1(E_twin, E_comb, F_f, F_e, F_v)
    
    G_F_full = np.concatenate([np.zeros(len(F_f)), np.zeros(len(F_e)), G_V])
    print(G_F_full.shape)
    
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
        
        Z1 = complex_projection(B1, B2, normals, V1)
        Z2 = complex_projection(B1, B2, normals, V2)
        
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
        lhs = d1.tocoo()
        
        G_F = G_F_full[mask_removed_f]
        I_F = I_F_full[mask_removed_f]
        rhs = - G_F + 2 * np.pi * I_F
        print(G_V)
        print(np.sort(rhs))
        
        constraints.append({
            'type': 'eq', 'fun': lambda x, i=i: lhs.dot(x[i*(num_E-3):(i+1)*(num_E-3)]) - rhs
            })
        
        # Thetas[mask_removed_E[:, i], i] = lsqr(lhs, rhs)[0]
        
    # Compute the thetas for the rest of the faces
    def objective(thetas):
        # Return ||sum_i thetas_i||^2
        return np.linalg.norm(np.sum(thetas.reshape(num_E-3, -1), axis=0))**2
    
    thetas_init = np.zeros((num_E-3) * len(F_singular))
    
    print('Optimising thetas...')
    opt = {'disp':True,'maxiter':1}
    result = minimize(objective, thetas_init, constraints=constraints, options=opt) 
    
    Thetas[mask_removed_E] = result.x
    
    thetas = np.sum(Thetas, axis=1)
    
    print(max(thetas), min(thetas))
    print(np.sum(thetas))
    print(np.sum(thetas > 0.5), np.sum(thetas < 0.01))
        
    return thetas
        

def reconstruct_corners_from_thetas(v_init, z_init, VEF_extended, thetas):
    V_extended, E_twin, E_comb, _, _, _ = VEF_extended
    
    E_extended = np.concatenate([E_twin, E_comb])
    
    A = lil_matrix((len(E_extended) + 1, len(V_extended)), dtype=complex)
    A[np.arange(len(E_extended)), E_extended[:, 0]] = np.exp(1j * thetas)
    A[np.arange(len(E_extended)), E_extended[:, 1]] = -1
    A[-1, v_init] = 1
    A = A.tocoo()
    
    b = np.zeros(len(E_extended) + 1, dtype=complex)
    b[-1] = z_init
    
    # A = lil_matrix((len(E_extended), len(V_extended)), dtype=complex)
    # A[np.arange(len(E_extended)), E_extended[:, 0]] = np.exp(1j * thetas)
    # A[np.arange(len(E_extended)), E_extended[:, 1]] = -1
    # A = A.tocoo()
    
    # b = np.zeros(len(E_extended), dtype=complex)
    
    U, _, _, r1norm = lsqr(A, b)[:4]
    print(r1norm)
    
    # normalise the vectors
    U = U / np.abs(U)
    
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
            
    result = lsqr(A, b)[0].reshape(4, -1)
    
    coeffs = np.stack([result[0] + 1j * result[1], result[2] + 1j * result[3]], axis=1)

    return coeffs


def construct_linear_field(V, F, singularities, indices, v_init, z_init):
    
    E = obtain_E(F)
    
    G_V = compute_G_V(V, E, F)
    
    VEF_extended = extended_mesh(V, E, F)
    
    thetas = compute_thetas(VEF_extended, singularities, indices, G_V)
    
    Z = reconstruct_corners_from_thetas(v_init, z_init, VEF_extended, thetas)
    
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
        
    return linear_field


def sample_points_and_vectors(V, F, field, num_samples=3):
    points = []
    for face in tqdm(F, desc='Sampling points and vectors', total=len(F)):
        for j in range(num_samples):
            for k in range(num_samples - j - 1):
                # Barycentric coordinates
                u = (j+1) / (num_samples + 1)
                v = (k+1) / (num_samples + 1)
                w = 1 - u - v
                
                # Interpolate to get the 3D point in the face
                points.append(u * V[face[0]] + v * V[face[1]] + w * V[face[2]])
                
    points = np.array(points)
    
    vectors = field(points)
    
    return points, vectors



# V = np.array([
#     [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 1, 0]
# ], dtype=float)
# F = np.array([
#     [0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4], [2, 6, 5], [3, 4, 5], [3, 5, 6], [2, 7, 6], [2, 1, 7], [1, 0, 7], [0, 3, 7], [3, 6, 7]
# ])
# singularities = np.array([[0.2, 0.2, 0], [0.4, 0.4, 0]])
# indices = [1, -1]
# v_init = 0
# z_init = 1j+1

if __name__ == '__main__':

    V, F = load_off_file(os.path.join('..', 'data', 'spherers.off'))
    
    # singularities = np.array([
    #     0.3 * V[F[0, 0]] + 0.3 * V[F[0, 1]] + 0.4 * V[F[0, 2]],
    #     0.4 * V[F[100, 0]] + 0.3 * V[F[100, 1]] + 0.3 * V[F[100, 2]]
    # ])
    # indices = [1, -1]
    
    singularities = np.array([
        0.3 * V[F[0, 0]] + 0.3 * V[F[0, 1]] + 0.4 * V[F[0, 2]]
    ])
    indices = [1]
    
    v_init = 0
    z_init = 1j+1

    field = construct_linear_field(V, F, singularities, indices, v_init, z_init)

    points, vectors = sample_points_and_vectors(V, F, field, num_samples=10)

    # normalise the vectors
    # vectors = vectors / np.linalg.norm(vectors, axis=1)[:, None]

    ps.init()
    ps_mesh = ps.register_surface_mesh("Input Mesh", V, F)
    ps_centroids = ps.register_point_cloud("Centroids", points, enabled=True)
    
    ps_centroids.add_vector_quantity('Samples', vectors, enabled=True)
    
    ps.register_point_cloud("singularity marker", singularities, enabled=True)

    ps.show()
            
