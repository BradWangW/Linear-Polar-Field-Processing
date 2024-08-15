import numpy as np
from tqdm import tqdm
from collections import defaultdict

def load_off_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the vertices and faces from the OFF file
    num_vertices, num_faces, _ = map(int, lines[1].split())

    vertices = np.array([list(map(float, line.split())) for line in lines[2:2 + num_vertices]])
    faces = np.array([list(map(int, line.split()))[1:] for line in lines[2 + num_vertices:]])

    return vertices, faces

def accumarray(indices, values):
    '''
    Accumulate values into an array using the indices.
    '''
    output = np.zeros((np.max(indices) + 1), dtype=values.dtype)
    indFlat = indices.flatten()
    valFlat = values.flatten()
    np.add.at(output, indFlat, valFlat)

    return output
        
def obtain_E(F):
    '''
    Obtain the edge list from the face list.
    '''
    E = np.concatenate([
        F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]
    ])
    E = np.unique(np.sort(E, axis=1), axis=0)
    
    return E

import numpy as np

def compute_cot_weights(V, E, F):
    # Number of vertices and edges
    n_edges = E.shape[0]
    
    # Initialize an array to hold the cotangent weights
    cot_weights = np.zeros(n_edges)
    
    # Helper function to calculate the cotangent of the angle opposite to a given edge
    def cotangent(a, b, c):
        ba = V[b] - V[a]
        ca = V[c] - V[a]
        cosine_angle = np.dot(ba, ca) / (np.linalg.norm(ba) * np.linalg.norm(ca))
        sine_angle = np.linalg.norm(np.cross(ba, ca)) / (np.linalg.norm(ba) * np.linalg.norm(ca))
        return cosine_angle / sine_angle

    # Create a dictionary to map edges to their corresponding cotangent weights
    edge_cot_map = {}

    for face in F:
        # The three vertices of the face
        i, j, k = face
        
        # Compute the cotangent weights for each angle in the triangle
        cot_alpha_ij = cotangent(i, j, k)
        cot_beta_ij = cotangent(j, k, i)
        cot_gamma_ij = cotangent(k, i, j)

        # Add these weights to the edges
        edges = [(i, j), (j, k), (k, i)]
        cots = [cot_gamma_ij, cot_alpha_ij, cot_beta_ij]
        
        for edge, cot in zip(edges, cots):
            sorted_edge = tuple(sorted(edge))
            if sorted_edge in edge_cot_map:
                edge_cot_map[sorted_edge] += cot
            else:
                edge_cot_map[sorted_edge] = cot

    # Populate the cotangent weights array
    for idx, edge in enumerate(E):
        sorted_edge = tuple(sorted(edge))
        if sorted_edge in edge_cot_map:
            cot_weights[idx] = edge_cot_map[sorted_edge] / 2.0  # divide by 2 as per the cotangent weight definition

    return cot_weights


def compute_V_boundary(F):
    # Build edge-to-face mapping
    E_to_F = defaultdict(list)

    for index_f, f in enumerate(F):
        for i in range(len(f)):
            e = tuple(sorted([f[i], f[(i + 1) % len(f)]]))
            E_to_F[e].append(index_f)

    # Identify boundary edges
    E_boundary = []

    for edge, faces in E_to_F.items():
        if len(faces) == 1:
            E_boundary.append(edge)

    # Extract boundary vertices
    V_boundary = np.unique(np.array(E_boundary).flatten())
    
    return V_boundary
 
def compute_angle_defect(V, F, V_boundary):
    angles = np.zeros(F.shape)
    
    # Compute the angles of each face
    for i, face in tqdm(enumerate(F), 
                        desc='Computing angle defect', 
                        total=F.shape[0],
                        leave=False):
        v1, v2, v3 = V[face]
        angles[i, 0] = np.arccos(np.dot(v2 - v1, v3 - v1) / 
                                (np.linalg.norm(v2 - v1) * np.linalg.norm(v3 - v1)))
        angles[i, 1] = np.arccos(np.dot(v1 - v2, v3 - v2) / 
                                (np.linalg.norm(v1 - v2) * np.linalg.norm(v3 - v2)))
        angles[i, 2] = np.pi - angles[i, 0] - angles[i, 1]
    
    # Accumulate the angles of each vertex
    angles = accumarray(
        F, 
        angles
    )
    
    # For interior (boundary) vertices, the angle defect is 2pi - sum angles (pi - sum angles)
    boundVerticesMask = np.zeros(V.shape[0])
    if len(V_boundary) > 0:
        boundVerticesMask[V_boundary] = 1
    
    G_V = (2 - boundVerticesMask) * np.pi - angles
    return G_V

def compute_planes(V, F):
    ''' 
    Compute the orthonormal basis vectors and the normal of the plane of each face.
        Input:
            V: (N, 3) array of vertices
            F: (M, 3) array of faces
        Output:
            B1: (M, 3) array of the first basis vector of each face
            B2: (M, 3) array of the second basis vector of each face
            normals: (M, 3) array of the normal vector of each face
    '''
    V_F = V[F]

    # Two basis vectors of the plane
    B1 = V_F[:, 1, :] - V_F[:, 0, :]
    B2 = V_F[:, 2, :] - V_F[:, 0, :]

    # Check parallelism
    para = np.all(np.cross(B1, B2) == 0, axis=1)
    if np.any(para):
        raise ValueError(f'The face(s) {F[np.where(para)]} is degenerate.')
    
    # Face normals of the planes
    normals = np.cross(B1, B2)
    normals = normals / np.linalg.norm(normals, axis=1)[:, None]
    
    # Correct for orthogonal basis vectors
    B2 = np.cross(normals, B1)
    
    # Normalise the basis vectors
    B1 = B1 / np.linalg.norm(B1, axis=1)[:, None]
    B2 = B2 / np.linalg.norm(B2, axis=1)[:, None]
    
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
        Output: (non-diagonal)
            Z: (M, N) array of the complex-represented projection of each point
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
            
            # Debug
            # print(f'Projection of {posi} to {normals} is {posi_plane}')
            # print(B1, B2, np.dot(B1, B2.T))
            
            X[:, i] = np.sum(B1 * posi_plane, axis=1)
            Y[:, i] = np.sum(B2 * posi_plane, axis=1)
            
        Z = X + 1j * Y
            
        return Z

def sort_neighbours( V, F, v, neighbours=None):
    '''
    Sort the neighbour faces of a vertex in counter-clockwise order.
    '''
    
    if neighbours is None:
        neighbours = np.any(F == v, axis=1)
        
    v1 = V[v].copy()
    F_neighbour = F[neighbours]
    
    # Compute the centroids of the neighbour faces
    v2s = np.mean(V[F_neighbour], axis=1).copy()
    
    # Sort by the angles between the centroids and the vertex
    epsilon = 1e-10
    angles = np.arccos(np.sum(v1 * v2s, axis=1) / ((np.linalg.norm(v1) + epsilon) * (np.linalg.norm(v2s, axis=1) + epsilon)))
    order = np.argsort(angles)
    
    return order

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

def compute_barycentric_coordinates(v1, v2, v3, posi):
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
    w = 1 - u - v
    
    return u, v, w
 

def is_in_face(V, F, posi, include_EV=False):
    '''
        Input:
            V: (N, 3) array of vertices
            F: (M, 3) array of faces
            posi: (3,) array of the point to be checked
        Output:
            False if the point is not in any face, 
            the index of the face if the point is in a face
    '''
    _, _, normals = compute_planes(V, F)
    
    # Filter the faces whose plane the point is in
    is_in_plane = np.where(np.abs(np.sum(normals * (posi[np.newaxis, :] - V[F[:, 0]]), axis=1)) < 1e-6)[0]
    
    if len(is_in_plane) == 0:
        # print('Error to the nearest face: ', np.min(np.abs(np.sum(normals * (posi[np.newaxis, :] - V[F[:, 0]]), axis=1))))
        # raise ValueError(f'The point {posi} is not in any plane.')
        return False
    else:
        # Check if the point is in the triangle
        candidate_faces = []
        for i in is_in_plane:
            f = F[i]
            v1 = V[f[0]]
            v2 = V[f[1]]
            v3 = V[f[2]]
            
            u, v, _ = compute_barycentric_coordinates(v1, v2, v3, posi)
            
            # If include_EV is True, the point is considered to be in the face if it is on the edge or the vertex
            # in other words, all adjacent faces are considered
            if include_EV:
                bool0 = (u >= 0) and (v >= 0) and (u + v <= 1)
            # if false, the point must be strictly inside the face
            else:
                bool0 = (u > 1e-6) and (v > 1e-6) and (u + v < 1 - 1e-6)

            if bool0:
                candidate_faces.append(i)
        
        if len(candidate_faces) == 1:
            if include_EV:
                return candidate_faces
            else:
                return candidate_faces[0]
        elif len(candidate_faces) > 1:
            if include_EV:
                return candidate_faces
            else:
                raise ValueError(f'The point {posi} is in more than one face.')
        else:
            return False

def normalise(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)

def rotation_matrix(axis, theta):
    """
    Generate a rotation matrix to rotate around a specified axis by theta radians.
    Uses the Rodrigues' rotation formula.
    """
    axis = normalise(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def compute_unfolded_vertex(a, b, c, d):
    """
    Find the coordinates of d' after rotating d around the axis bc so that the plane (b, c, d') 
    lies in the same plane as (a, b, c) and does not overlap.
    """
    # Vectors for the faces
    n1 = np.cross(b - a, c - a) # Normal vector to (a, b, c)
    n2 = np.cross(b - d, c - d) # Normal vector to (b, c, d)
    
    # Normalize the normals
    n1 = normalise(n1)
    n2 = normalise(n2)
    
    # Rotation axis (normalized)
    u = normalise(c - b)
    
    # Angle between the normals
    cos_theta = np.dot(n1, n2)
    sin_theta = np.linalg.norm(np.cross(n1, n2))
    theta = np.arctan2(sin_theta, cos_theta)
    
    angles = [theta, np.pi - theta, -theta, -np.pi + theta]
    furthest_point = None
    furthest_distance = 0
    
    for angle in angles:
        R = rotation_matrix(u, angle)
        d_prime = b + np.dot(R, d - b)
        
        distance = np.linalg.norm(d_prime - a)
        
        if distance > furthest_distance:
            furthest_distance = distance
            furthest_point = d_prime
    
    return furthest_point
