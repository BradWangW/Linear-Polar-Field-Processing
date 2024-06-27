import numpy as np
from tqdm import tqdm
from collections import defaultdict
from scipy.sparse import lil_matrix
from Auxiliary import accumarray, find_indices, is_in_face



class Triangle_mesh():
    '''
    A class for processing triangle meshes by vertex and edge extending approach.
    '''
    def __init__(self, V, F):
        self.V = V
        self.F = F
        self.E = self.obtain_E(F)
        
        self.V_boundary = self.compute_V_boundary(F)
        
        self.G_V = self.compute_angle_defect(V, F, self.V_boundary)
        
    def initialise_field_processing(self):
        self.construct_extended_mesh()
        self.construct_d1_extended()
        
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
    
    def compute_angle_defect(self):
        angles = np.zeros(self.F.shape)
        
        # Compute the angles of each face
        for i, face in tqdm(enumerate(self.F), 
                            desc='Computing angle defect', 
                            total=self.F.shape[0]):
            v1, v2, v3 = self.V[face]
            angles[i, 0] = np.arccos(np.dot(v2 - v1, v3 - v1) / 
                                    (np.linalg.norm(v2 - v1) * np.linalg.norm(v3 - v1)))
            angles[i, 1] = np.arccos(np.dot(v1 - v2, v3 - v2) / 
                                    (np.linalg.norm(v1 - v2) * np.linalg.norm(v3 - v2)))
            angles[i, 2] = np.pi - angles[i, 0] - angles[i, 1]
        
        # Accumulate the angles of each vertex
        angles = accumarray(
            self.F, 
            angles
        )
        
        # For interior (boundary) vertices, the angle defect is 2pi - sum angles (pi - sum angles)
        boundVerticesMask = np.zeros(self.V.shape[0])
        boundVerticesMask[self.V_boundary] = 1
        
        G_V = (2 - boundVerticesMask) * np.pi - angles
        return G_V
      
    def sort_neighbours(V, F, v, neighbours=None):
        '''
        Sort the neighbour faces of a vertex in counter-clockwise order.
        '''
        if not neighbours:
            neighbours = np.any(F == v, axis=1)
            
        v1 = V[v].copy()
        F_neighbour = F[neighbours]
        
        # Compute the centroids of the neighbour faces
        v2s = np.mean(V[F_neighbour], axis=1).copy()
        
        # Avoid division by zero
        while np.any(np.linalg.norm(v2s, axis=1) == 0) or np.linalg.norm(v1) == 0:
            v1 += 1e-6
            v2s += 1e-6
        
        # Sort by the angles between the centroids and the vertex
        angles = np.arccos(np.sum(v1 * v2s, axis=1) / (np.linalg.norm(v1) * np.linalg.norm(v2s, axis=1)))
        order = np.argsort(angles)
        
        return order
    
    def construct_extended_mesh(self):
        '''
        Construct the extended mesh from the input mesh, which consists of 
            the extended vertices, twin edges and combinatorial edges,
            face-faces, edge-faces, and vertex-faces.
        '''
        
        # sum_degree = np.sum(np.count_nonzero(self.E == v, axis=1) for v in range(self.V.shape[0]))
        
        V_extended = []
        E_twin = np.zeros((self.F.shape[0] * 3, 2), dtype=int)
        E_comb = []
        F_f = np.zeros((self.F.shape[0], 3), dtype=int)
        F_e = []
        F_v = []
        
        # Mapping from original vertices to (corresponding) extended vertices
        V_map = {v:[] for v in range(self.V.shape[0])}
        
        # Construct face-faces and twin edges
        # Loop over original faces
        for i, f in tqdm(enumerate(self.F), desc='Constructing face-faces and twin edges'):
            f_f = []
            
            # For each vertex in the original face
            for v in f:
                # Add the coordinate
                V_extended.append(self.V[v])
                
                # Index of the extended vertex
                index_extended = len(V_extended)
                f_f.append(index_extended)
                
                V_map[v].append(index_extended)

            # Add the face-face
            F_f[i] = f_f
            
            # Add the twin edges
            E_twin[i * 3:i * 3 + 3] = np.stack([f_f, np.roll(f_f, -1)], axis=1)
            
        # Construct edge-faces
        for i, e in tqdm(enumerate(self.E), desc='Constructing edge-faces'):
            indices1_extended = V_map[e[0]]
            indices2_extended = V_map[e[1]]
            
            # In a triangle mesh, two faces share at most one edge,
            # so when extracting the twin edges that encompass both extended_vertices of v1 and v2,
            # Either one or two edges will be found.
            e_twin = E_twin[
                np.all(
                    np.isin(E_twin, indices1_extended + indices2_extended), 
                    axis=1
                )
            ]
            
            # If two edges are found, the edge is an interior edge
            # in which case the 4 vertices give an edge-face
            if e_twin.shape[0] == 2:
                # Check if the twin edges are aligned or opposite
                pairing = np.isin(e_twin, indices1_extended)
                
                # If aligned, reverse the second twin edge to make a proper face
                if np.all(pairing[0] == pairing[1]):
                    F_e.append(e_twin[0].tolist() + e_twin[1, ::-1].tolist())
                # If opposite, the twin edges are already proper faces
                elif np.all(pairing[0] == pairing[1][::-1]):
                    F_e.append(e_twin[0].tolist() + e_twin[1].tolist())
                else:
                    raise ValueError('The twin edges are not aligned or opposite.')
                
            # If one edge is found, the edge is a boundary edge, 
            # in which case no edge-face is formed
            elif e_twin.shape[0] == 1:
                pass
            else:
                raise ValueError(f'Wrong number of twin edges found: {e_twin}.')
                
        # Construct vertex-faces and combinatorial edges
        for v, indices_extended in tqdm(V_map.items(), desc='Constructing vertex-faces and combinatorial edges'):
            # Find the neighbours of the vertex
            neighbours = np.any(np.isin(F_f, indices_extended), axis=1)
            
            # Sort the neighbours in counter-clockwise order
            order = self.sort_neighbours(self.V, self.F, v, neighbours)
            indices_sorted = np.array(indices_extended)[order]
            
            # If the vertex is not on the boundary
            if len(indices_extended) > 2:
                # Add the vertex-face
                F_v.append(indices_sorted.tolist())
                
                # Construct the combinatorial edges
                E_comb += np.stack([
                    indices_sorted, 
                    np.roll(indices_sorted, -1)
                ], axis=1).tolist()

            # If the vertex is on the boundary
            elif len(indices_extended) == 2:
                # Add the (one) combinatorial edge
                E_comb.append(indices_sorted)
            else:
                raise ValueError(f'Wrong number of extended vertices found for {v}: {indices_extended}.')
            
        self.V_extended = np.array(V_extended)
        
        self.E_twin = E_twin
        self.E_comb = np.array(E_comb)
        self.E_extended = np.concatenate([E_twin, E_comb])
        
        self.F_f = F_f
        self.F_e = np.array(F_e)
        self.F_v = F_v
        
        self.V_map = V_map
        self.G_F = np.concatenate([
            np.zeros(len(self.F_f) + len(self.F_e)),
            self.G_V
        ])
        
    def construct_d1_extended(self):
        d1 = lil_matrix((len(self.F_f) + len(self.F_e) + len(self.F_v), len(self.E_extended)))
        
        # The edges of face faces are oriented f[0] -> f[1] -> f[2], 
        # so F_f[:, i:i+1%3] appears exactly the same in E_twin, 
        # thus index-searching is enough.
        for i in range(3):
            # Find the indices of the face edges in the edge list
            indices = find_indices(self.E_extended, np.stack([self.F_f[:, i], self.F_f[:, (i+1)%3]], axis=1))

            d1[np.arange(len(self.F_f)), indices] = 1

        # The edges of the edge faces are not oriented as the faces are,
        # so we check the orientation alignment
        for i, f in enumerate(self.F_e):
            # Find the indices of the face edges in the edge list
            indices = np.where(np.all(np.isin(self.E_extended, f), axis=1))[0]
            E_f = self.E_extended[indices]
            
            for index, e in zip(indices, E_f):
                # If the edge is aligned with the face, the orientation is positive
                if (np.where(f == e[0])[0] == np.where(f == e[1])[0] - 1) or (f[-1] == e[0] and f[0] == e[1]):
                    d1[len(self.F_f) + i, index] = 1
                # If the edge is opposite to the face, the orientation is negative
                elif np.where(f == e[0])[0] == np.where(f == e[1])[0] + 1 or (f[-1] == e[1] and f[0] == e[0]):
                    d1[len(self.F_f) + i, index] = -1
                else:
                    raise ValueError(f'The edge {e} is not in the face {f}, or the edge face is wrongly defined.')

        # The edges of the vertex faces are oriented counter-clockwisely, 
        # so similar to the face faces, we can directly index-search.
        for i, f in tqdm(enumerate(self.F_v), desc='Constructing d1'):
            # Find the indices of the combinatorial edges in the edge list
            indices = find_indices(
                self.E_extended, 
                np.stack([f, np.roll(f, -1)], axis=1)
            )

            d1[len(self.F_f) + len(self.F_e) + i, indices] = 1
            
        self.d1 = d1.tocsr()
        
    def compute_thetas(self, singularities=None, indices=None):
        # Construct the index array and filter singular faces
        I_F_f = np.zeros(len(self.F_f))
        I_F_e = np.zeros(len(self.F_e))
        I_F_v = np.zeros(len(self.F_v))
        F_singular = []
        
        for singularity, index in zip(singularities, indices):
            
            # Check if the singularity is in an edge or vertex
            in_F_e = np.all(
                (self.V_extended[self.F_e[:, 0]] > singularity) * (self.V_extended[self.F_e[:, 1]] < singularity), 
                axis=1
            )
            in_F_v = np.all(
                self.V_extended[[f_v[0] for f_v in self.F_v]] == singularity,
                axis=1
            )
            
            # If the singularity is in an edge or vertex, assign the index
            if np.any(in_F_e):
                I_F_e[in_F_e] = index
            elif np.any(in_F_v):
                I_F_v[in_F_v] = index
            # If the singularity is in a face, it contributes a set of thetas
            else:
                F_candidate = is_in_face(self.V_extended, self.F_f, singularity)
                
                if F_candidate:
                    F_singular.append((F_candidate[0], singularity, index))
                else:
                    raise ValueError(f'The singularity {singularity} is not in any face, edge or vertex.')
            
        self.I_F = np.concatenate([I_F_f, I_F_e, I_F_v])
        
        # Compute the set of thetas for each face singularity
        # by constrained optimisation using the KKT conditions
        Thetas = np.zeros((len(self.E_extended), len(F_singular)))
        
        constraints = []
        
        

Field - dictate a,b,c,d - thetas - rsconstruct U - reconstruct coefficients