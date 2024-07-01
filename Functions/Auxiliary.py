import numpy as np
from tqdm import tqdm
from collections import defaultdict

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


def obtain_E(F):
    '''
    Obtain the edge list from the face list.
    '''
    E = np.concatenate([
        F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]
    ])
    E = np.unique(np.sort(E, axis=1), axis=0)
    
    return E


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
            
            # If include_EV is True, the point is considered to be in the face if it is on the edge or the vertex
            # in other words, all adjacent faces are considered
            if include_EV:
                bool0 = (u >= 0) and (v >= 0) and (u + v <= 1)
            # if false, the point must be strictly inside the face
            else:
                bool0 = (u > 0) and (v > 0) and (u + v < 1)

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


def sample_points_and_vectors(V, F, field, num_samples=3):
    points = []
    for face in tqdm(F, desc='Sampling points and vectors', 
                     total=len(F), leave=False):
        for j in range(num_samples):
            for k in range(num_samples - j - 1):
                # Barycentric coordinates
                u = (j+1) / (num_samples + 1)
                v = (k+1) / (num_samples + 1)
                w = 1 - u - v
                
                # Interpolate to get the 3D point in the face
                points.append(u * V[face[0]] + v * V[face[1]] + w * V[face[2]])
                
    points = np.array(points)

    posis, vectors = field(points)
    
    return posis, vectors
            
