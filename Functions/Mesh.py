import numpy as np
from tqdm import tqdm
from scipy.sparse import lil_matrix, eye, bmat
from Functions.Auxiliary import accumarray, find_indices, is_in_face, compute_planes, complex_projection, obtain_E, compute_V_boundary
from scipy.sparse.linalg import lsqr
import cvxopt



class Triangle_mesh():
    '''
    A class for processing triangle meshes by vertex and edge extending approach.
    '''
    def __init__(self, V, F):
        self.V = V
        self.F = F
        self.E = obtain_E(F)
        
        self.V_boundary = compute_V_boundary(F)
        
        self.B1, self.B2, self.normals = compute_planes(V, F)
        
        self.G_V = self.compute_angle_defect()
        
    def initialise_field_processing(self):
        self.construct_extended_mesh()
        self.construct_d1_extended()
        self.compute_face_pair_rotation()
        
    def compute_angle_defect(self):
        angles = np.zeros(self.F.shape)
        
        # Compute the angles of each face
        for i, face in tqdm(enumerate(self.F), 
                            desc='Computing angle defect', 
                            total=self.F.shape[0],
                            leave=False):
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
        if len(self.V_boundary) > 0:
            boundVerticesMask[self.V_boundary] = 1
        
        G_V = (2 - boundVerticesMask) * np.pi - angles
        return G_V
      
    def sort_neighbours(self, V, F, v, neighbours=None, extended=True):
        '''
        Sort the neighbour faces of a vertex in counter-clockwise order.
        '''
        # if extended:
        #     V = self.V_extended
        #     F = self.F_f
        # else:
        #     V = self.V
        #     F = self.F
        
        if neighbours is None:
            neighbours = np.any(F == v, axis=1)
            
        v1 = V[v].copy()
        F_neighbour = F[neighbours]
        
        # Compute the centroids of the neighbour faces
        v2s = np.mean(V[F_neighbour], axis=1).copy()
        
        # Avoid division by zero
        # while np.any(np.linalg.norm(v2s, axis=1) == 0) or np.linalg.norm(v1) == 0:
        #     v1 += 1e-6
        #     v2s += 1e-6
        
        # Sort by the angles between the centroids and the vertex
        epsilon = 1e-10
        angles = np.arccos(np.sum(v1 * v2s, axis=1) / ((np.linalg.norm(v1) + epsilon) * (np.linalg.norm(v2s, axis=1) + epsilon)))
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
        for i, f in tqdm(enumerate(self.F), 
                         desc='Constructing face-faces and twin edges',
                         leave=False,
                         total=self.F.shape[0]):
            f_f = []
            
            # For each vertex in the original face
            for v in f:
                # Add the coordinate
                V_extended.append(self.V[v])
                
                # Index of the extended vertex
                index_extended = len(V_extended) - 1
                f_f.append(index_extended)
                
                V_map[v].append(index_extended)

            # Add the face-face
            F_f[i] = f_f
            
            # Add the twin edges
            for j, k in np.stack([np.arange(3), np.roll(np.arange(3), -1)], axis=1):
                E_twin[i * 3 + j] = [f_f[j], f_f[k]]
            
        V_extended = np.array(V_extended)
            
        # Construct edge-faces
        for i, e in tqdm(enumerate(self.E), 
                         desc='Constructing edge-faces',
                         leave=False,
                         total=self.E.shape[0]):
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
        for v, indices_extended in tqdm(V_map.items(), 
                                        desc='Constructing vertex-faces and combinatorial edges',
                                        leave=False):
            # Find the neighbours of the vertex
            neighbours = np.any(self.F == v, axis=1)
            F_neighbours = self.F[neighbours]
            
            mask = ~np.isin(F_neighbours, v)
            E_neighbours = np.sort(F_neighbours[mask].reshape(-1, 2), axis=1)
            
            # Sort the neighbours so that they form a Hamiltonian cycle
            order = [0]
            E_neighbours_in_cycle = np.zeros((len(E_neighbours), 2), dtype=int)
            E_neighbours_in_cycle[0] = E_neighbours[0]
            
            # After considered, the edge is changed to [-1, -1] to avoid re-consideration
            E_neighbours[0] = [-1, -1]
            
            for i in range(1, len(E_neighbours)):
                next_edge = np.where(
                    np.any(E_neighbours == E_neighbours_in_cycle[i-1, 1], axis=1)
                )[0][0]
                order.append(next_edge)
                
                # If the next edge is head-to-tail with the last edge, keep the order
                if E_neighbours[next_edge, 0] == E_neighbours_in_cycle[i-1, 1]:
                    E_neighbours_in_cycle[i] = E_neighbours[next_edge]
                # If the next edge is tail-to-tail with the last edge, reverse the order
                elif E_neighbours[next_edge, 0] != E_neighbours_in_cycle[i-1, 1]:
                    E_neighbours_in_cycle[i] = E_neighbours[next_edge][::-1]
                    
                E_neighbours[next_edge] = [-1, -1]
                
            indices_sorted = [indices_extended[i] for i in order]
            
            # Only if the vertex is adjacent to > 2 faces,
            # the vertex-face is constructed
            # Otherwise, only one combinatorial edge is formed
            if len(indices_extended) > 2:
                # Add the vertex-face
                F_v.append(indices_sorted)
                
                # Construct the combinatorial edges
                E_comb += np.stack([
                    indices_sorted, 
                    np.roll(indices_sorted, -1)
                ], axis=1).tolist()

            # If the vertex is adjacent to <= 2 faces
            elif len(indices_extended) == 2:
                # Add the (one) combinatorial edge
                E_comb.append(indices_sorted)
            else:
                raise ValueError(f'Wrong number of extended vertices found for {v}: {indices_extended}.')
            
        self.V_extended = V_extended
        
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
        '''
            Construct the incidence matrix d1 for the extended mesh.
        '''
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
        for i, f in tqdm(enumerate(self.F_v), 
                         desc='Constructing d1', 
                         total=len(self.F_v), 
                         leave=False):
            # Find the indices of the combinatorial edges in the edge list
            indices = find_indices(
                self.E_extended, 
                np.stack([f, np.roll(f, -1)], axis=1)
            )

            d1[len(self.F_f) + len(self.F_e) + i, indices] = 1
        
        # print(d1.shape)
        # # print(d1_arr, d1_arr[:-1])
        # print(np.linalg.matrix_rank(d1.toarray()))
        # print(np.linalg.matrix_rank(d1[:-1].toarray()))
            
        self.d1 = d1
        
    def compute_face_pair_rotation(self):
        pair_rotations = np.zeros(self.E_comb.shape[0])
        
        # The ith element in F_e and E represent the same edge
        for f_e in tqdm(self.F_e, 
                        desc='Computing face pair rotations', 
                        total=len(self.F_e), 
                        leave=False):
            # Recall each f_e is formed as v1 -> v2 -> v2 -> v1
            # so that v1 -> v2 is a twin edge and v1 -> v1 is a combinatorial edge
            e1_comb = np.all(np.sort(self.E_comb, axis=1) == np.sort(f_e[[0, 3]]), axis=1)
            e2_comb = np.all(np.sort(self.E_comb, axis=1) == np.sort(f_e[[1, 2]]), axis=1)

            vec_e = self.V_extended[f_e[1]] - self.V_extended[f_e[0]]
            
            f1_f = np.where(np.any(np.isin(self.F_f, f_e[0]), axis=1))[0]
            f2_f = np.where(np.any(np.isin(self.F_f, f_e[3]), axis=1))[0]
            
            B1, B2, normals = compute_planes(self.V_extended, self.F_f[[f1_f, f2_f]][:, 0])
            f1 = self.F_f[f1_f]
            f2 = self.F_f[f2_f]
        
            U = complex_projection(B1, B2, normals, vec_e[None, :])
            # print(U)
            # U[U > 0] = U[U > 0] / np.abs(U[U > 0])
            # rotation = np.arccos(
            #     np.real(np.conjugate(U[0]) * U[1]) / (np.abs(U[0]) * np.abs(U[1]))
            # )
            rotation = np.angle(U[1]) - np.angle(U[0])
            # rotation = np.arccos((U[0] * np.conjugate(U[1])).real)
            # print(np.mod(rotation + np.pi, 2*np.pi) - np.pi)
            
            if np.all(self.E_comb[e1_comb] == f_e[[0, 3]]):
                pair_rotations[e1_comb] = rotation
            elif np.all(self.E_comb[e1_comb] == f_e[[3, 0]]):
                pair_rotations[e1_comb] = -rotation
            else:
                raise ValueError(f'{self.E_comb[e1_comb]} and {f_e[[0, 3]]} do not match.')
            
            if np.all(self.E_comb[e2_comb] == f_e[[1, 2]]):
                pair_rotations[e2_comb] = rotation
            elif np.all(self.E_comb[e2_comb] == f_e[[2, 1]]):
                pair_rotations[e2_comb] = -rotation
            else:
                raise ValueError(f'{self.E_comb[e2_comb]} and {f_e[[1, 2]]} do not match.')
            
        self.pair_rotations = pair_rotations
        
        
    def compute_thetas(self, singularities=None, indices=None):
        '''
            Compute the set of thetas for each face singularity 
            by constrained optimisation using the KKT conditions.
        '''
        if len(singularities) != len(indices):
            raise ValueError('The number of singularities and the number of indices do not match.')
            
        # Construct the index array and filter singular faces
        I_F_f = np.zeros(len(self.F_f))
        I_F_e = np.zeros(len(self.F_e))
        I_F_v = np.zeros(len(self.F_v))
        self.F_singular = []
        self.singularities_F = []
        self.indices_F = []
        
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
                
                if F_candidate != None:
                    self.F_singular.append(F_candidate)
                    self.singularities_F.append(singularity)
                    self.indices_F.append(index)
                else:
                    raise ValueError(f'The singularity {singularity} is not in any face, edge or vertex.')
            
        self.I_F = np.concatenate([I_F_f, I_F_e, I_F_v])
        self.b = -self.G_F + 2 * np.pi * self.I_F
        
        # Compute the set of thetas for each face singularity
        # by constrained optimisation using the KKT conditions
        Thetas = np.zeros((len(self.E_extended), len(self.F_singular)))
        mask_removed_E = np.ones((len(self.E_extended), len(self.F_singular)), dtype=bool)
        
        # Independent quantities for quadratic programming
        Q_block = eye(len(self.E_extended) - 3)
        Q = bmat([
            [Q_block] * len(self.F_singular)
        ] * len(self.F_singular), format='coo')
        c = np.zeros((len(self.E_extended) - 3) * len(self.F_singular))
        
        # Quantities for quadratic programming dependent on the singularities
        dim_E_block = (
            (len(self.F_f) + len(self.F_e) + len(self.F_v) - 1), 
            (len(self.E_extended) - 3)
        )
        E = lil_matrix(
            (dim_E_block[0] * len(self.F_singular), dim_E_block[1] * len(self.F_singular)), 
            dtype=float
        )
        d = np.zeros(dim_E_block[0] * len(self.F_singular))
        
        for i, (f_singular, singularity, index) in tqdm(enumerate(zip(self.F_singular, self.singularities_F, self.indices_F)), 
                                               desc = 'Computing thetas', 
                                               total=len(self.F_singular), 
                                               leave=False):
            # Find the face containing the singularity
            e_f = np.all(np.isin(self.E_extended, self.F_f[f_singular]), axis=1)
            # print('Edges of the singular face:', np.where(e_f))
            
            b1, b2, normal = self.B1[f_singular], self.B2[f_singular], self.normals[f_singular]
            
            V1 = singularity - self.V_extended[self.E_extended[e_f, 0]]
            V2 = singularity - self.V_extended[self.E_extended[e_f, 1]]
            
            Z1 = complex_projection(b1[None, :], b2[None, :], normal[None, :], V1)
            Z2 = complex_projection(b1[None, :], b2[None, :], normal[None, :], V2)
            
            # rotations = (index * (np.angle(Z2) - np.angle(Z1))).squeeze()
            # rotations = np.mod(rotations + np.pi, 2 * np.pi) - np.pi
            rotations = index * np.arccos(
                np.real(np.conjugate(Z1) * Z2) / (np.abs(Z1) * np.abs(Z2))
            ).squeeze()
            
            Thetas[e_f, i] = rotations
            
            # print('Rotation and Thetas shapes: ', rotations.shape, Thetas[e_f, i].shape)
            
            # Masks for truncating the matrix blocks
            mask_removed_f = np.ones(len(self.F_f) + len(self.F_e) + len(self.F_v), dtype=bool)
            mask_removed_f[f_singular] = False
            mask_removed_E[e_f, i] = False
            
            d1_i = self.d1[mask_removed_f]
            d1_i = d1_i[:, mask_removed_E[:, i]]
            # d1_i = d1_i[:-1]
            E[dim_E_block[0] * i:dim_E_block[0] * (i + 1), dim_E_block[1] * i:dim_E_block[1] * (i + 1)] = d1_i.toarray()
            
            b1 = self.b.copy()
            # For the other edge faces involving one of the computed edges, 
            # the rhs of the system needs to minus the rotation of that edge
            for j in range(3):
                e_involved = np.where(e_f)[0][j]
                
                f_involved = len(self.F_f) + np.where(
                    np.sum(np.isin(self.F_e, self.E_extended[e_f][j]), axis=1) == 2
                )[0][0]
                
                affect_in_d1 = -self.d1[f_involved, e_involved]
                
                b1[f_involved] = affect_in_d1 * rotations[j]
            
            b1 = b1[mask_removed_f]
            # b1 = b1[:-1]
            d[dim_E_block[0] * i:dim_E_block[0] * (i + 1)] = b1
            
        # Define the system to solve the quadratic programming problem
        KKT_lhs = bmat([
            [Q, E.T],
            [E, np.zeros((dim_E_block[0] * len(self.F_singular), dim_E_block[0] * len(self.F_singular)))]
        ], format='coo')
        KKT_rhs = np.concatenate([-c, d])
        
        # Solve the quadratic programming problem
        solution, _, itn, r1norm = lsqr(KKT_lhs, KKT_rhs)[:4]
        # print(solution)
        
        # Extract the thetas
        for i in range(len(self.F_singular)):
            Thetas[mask_removed_E[:, i], i] = solution[i * (len(self.E_extended) - 3):(i + 1) * (len(self.E_extended) - 3)]

        print(f'Theta computation iteration and residual: {itn}, {r1norm}.')

        # -----------------------------------------------------------------------------------#
        # Using package cvxopt to solve the quadratic programming problem
        # Verified: gave the same result
        # Q = cvxopt.matrix(Q.toarray())
        # c = cvxopt.matrix(c)
        # E = cvxopt.matrix(E.toarray())
        # d = cvxopt.matrix(d)

        # # Define G and h for no inequality constraints
        # G = cvxopt.matrix(np.zeros(dim_E))
        # h = cvxopt.matrix(np.zeros(dim_E[0]))

        # # Solve the quadratic program
        # solution = cvxopt.solvers.qp(Q, c, G, h, E, d)

        # # Extract the optimal solution
        # print(solution['x'])
        # -----------------------------------------------------------------------------------#
        
        return Thetas
    
    def reconstruct_corners_from_thetas(self, Thetas, v_init, z_init, Thetas_include_pairface=False):
        '''
            Reconstruct the corner values from the thetas.
            Input:
                Thetas: (num_E, num_singularities) array of thetas
                v_init: initial value for the corner in the vertex face
                z_init: initial value for the corner in the edge face
            Output:
                Us: (num_V_extended, num_singularities) complex array of corner values
        '''
        if not Thetas_include_pairface:
            for j in range(Thetas.shape[1]):
                Thetas[len(self.E_twin):, j] += self.pair_rotations
                
            # Thetas[len(self.E_twin):] += self.pair_rotations[:, None]
            Thetas = np.mod(Thetas + np.pi, 2 * np.pi) - np.pi
            
        Us = np.zeros((len(self.V_extended), Thetas.shape[1]), dtype=complex)
        itns = np.zeros(Thetas.shape[1])
        r1norms = np.zeros(Thetas.shape[1])
        
        for i in range(Thetas.shape[1]):
            thetas = Thetas[:, i]
            
            lhs = lil_matrix((len(self.E_extended) + 1, len(self.V_extended)), dtype=complex)
            lhs[np.arange(len(self.E_extended)), self.E_extended[:, 0]] = np.exp(1j * thetas)
            lhs[np.arange(len(self.E_extended)), self.E_extended[:, 1]] = -1
            lhs[-1, v_init] = 1
            
            rhs = np.zeros(len(self.E_extended) + 1, dtype=complex)
            rhs[-1] = z_init / np.abs(z_init)
            
            Us[:, i], _, itns[i], r1norms[i] = lsqr(lhs.tocsr(), rhs)[:4]
            
        print(f'Corner reconstruction iterations and residuals',
              np.stack([itns, r1norms], axis=1))
        # print(Us)
        
        return Us
    
    def reconstruct_linear_from_corners(self, Us, singularities_F=None, six_equations=False):
        '''
            Reconstruct the coefficients of the linear field from the corner values.
            Input:
                Us: (num_F, num_singularities) complex array of corner values
                singularities_F: (num_singularities, 3) array of singularities in the faces
            Output:
                coeffs: (num_F, num_singularities, 2) complex array of coefficients for the linear fields
        '''
        if singularities_F is None:
            singularities_F = self.singularities_F
            
        if Us.shape[1] != len(singularities_F):
            raise ValueError('The number of singularities and the number of sets of corner values do not match.')

        coeffs = np.zeros((len(self.F_f), Us.shape[1], 2), dtype=complex)
        itns = np.zeros(Us.shape[1])
        r1norms = np.zeros(Us.shape[1])
        
        for i, singularity in tqdm(enumerate(singularities_F), 
                                   desc='Reconstructing linear field coefficients', 
                                   total=len(singularities_F), 
                                   leave=False):
            U = Us[:, i]
            
            # Find the face containing the singularity
            f_singular = is_in_face(self.V_extended, self.F_f, singularity)
            
            if f_singular == None:
                raise ValueError(f'The singularity {singularity} is not in any face.')
            
            # Construct the linear system for the coefficients of the linear field
            if six_equations:
                # Specific for fitting both Re(az+b)Im(u) = Im(az+b)Re(u)
                # and Re(az+b) + Im(az+b) = Re(u) + Im(u) for all 3 corners (6 equations)
                lhs = lil_matrix((len(self.F_f) * 6, len(self.F_f) * 4), dtype=float)
                rhs = np.zeros(len(self.F_f) * 6, dtype=float)
            else:
                # General for fitting 4 equations
                lhs = lil_matrix((len(self.F_f) * 4, len(self.F_f) * 4), dtype=float)
                rhs = np.zeros(len(self.F_f) * 4, dtype=float)

            for j, f in tqdm(enumerate(self.F_f), 
                             desc=f'Constructing the linear system for singularity {singularity}',
                             total=len(self.F_f),
                             leave=False):
                b1 = self.B1[j][None, :]; b2 = self.B2[j][None, :]; normal = self.normals[j][None, :]
                
                # Compute the complex representation of the vertices on the face face
                Zf = complex_projection(b1, b2, normal, self.V_extended[f] - self.V_extended[f[0]])[0]
                
                Uf = U[f]
                prod = np.conjugate(Uf) * Zf

                # Linearity enables linear combination
                centre_z = np.mean(Zf)
                centre_u = np.mean(Uf)
                centre_prod = np.conjugate(centre_u) * centre_z

                if six_equations:
                    # Corners by both Re(az+b)Im(u) = Im(az+b)Re(u)
                    # and Re(az+b) + Im(az+b) = Re(u) + Im(u)

                    # If the face is singular, the last row explicitly specifies 
                    # the singularity (zero point of the field)
                    if j == f_singular:
                        zc = complex_projection(
                            b1, b2, normal, 
                            np.array([singularity - self.V_extended[f[0]]])
                        )[0, 0]
                        last_row = [zc.real + zc.imag, zc.real - zc.imag, 1, 1]
                    # If the face is not singular, the last row aligns the first corner 
                    # value up to +-sign and scale, as for the other two corners
                    else:
                        last_row = [prod[2].imag, prod[2].real, -Uf[2].imag, Uf[2].real]
                    
                    # Construct the linear system (blocks of 6x4 in the matrix)
                    lhs[6*j:6*(j+1), 4*j:4*(j+1)] = np.array([
                        [Zf[0].real + Zf[0].imag, Zf[0].real - Zf[0].imag, 1, 1],
                        [Zf[1].real + Zf[1].imag, Zf[1].real - Zf[1].imag, 1, 1],
                        [Zf[2].real + Zf[2].imag, Zf[2].real - Zf[2].imag, 1, 1],
                        [prod[0].imag, prod[0].real, -Uf[0].imag, Uf[0].real],
                        [prod[1].imag, prod[1].real, -Uf[1].imag, Uf[1].real],
                        last_row
                    ], dtype=float)
                    rhs[6*j:6*(j+1)] = [
                        Uf[0].real + Uf[0].imag,
                        Uf[1].real + Uf[1].imag,
                        Uf[2].real + Uf[2].imag,
                        0,
                        0,
                        0
                    ]

                else:
                    #-------------------------------------------------------------------------# 
                    # Corners by Re(az+b)Im(u) = Im(az+b)Re(u)
                    # No more information is given

                    # If the face is singular, the last row explicitly specifies 
                    # the singularity (zero point of the field)
                    if j == f_singular:
                        zc = complex_projection(
                            b1, b2, normal, 
                            np.array([singularity - self.V_extended[f[0]]])
                        )[0, 0]
                            
                        lhs[4*j:4*(j+1), 4*j:4*(j+1)] = np.array([
                            [zc.real, -zc.imag, 1, 0],
                            [zc.imag, zc.real, 0, 1],
                            [Zf[0].real, -Zf[0].imag, 1, 0],
                            [Zf[0].imag, Zf[0].real, 0, 1]
                        ], dtype=float)
                        
                        rhs[4*j:4*(j+1)] = [
                            0, 0, Uf[0].real, Uf[0].imag
                        ]
                    # If the face is not singular, the last row aligns the first corner 
                    # value up to +-sign and scale, as for the other two corners
                    else:
                        lhs[4*j:4*(j+1), 4*j:4*(j+1)] = np.array([
                            [prod[0].imag, prod[0].real, -Uf[0].imag, Uf[0].real],
                            [prod[1].imag, prod[1].real, -Uf[1].imag, Uf[1].real],
                            [Zf[0].real, -Zf[0].imag, 1, 0],
                            [Zf[0].imag, Zf[0].real, 0, 1]
                        ], dtype=float)
                        
                        rhs[4*j:4*(j+1)] = [
                            0, 0, Uf[0].real, Uf[0].imag
                        ]
                    #-------------------------------------------------------------------------#
                    
                    #-------------------------------------------------------------------------# 
                    # # Corners by Re(az+b) + Im(az+b) = Re(u) + Im(u)
                    # # Centre by Re(az+b)Im(u) = Im(az+b)Re(u)

                    # # If the face is singular, the last row explicitly specifies 
                    # # the singularity (zero point of the field)
                    # if j == f_singular:
                    #     zc = complex_projection(
                    #         b1, b2, normal, 
                    #         np.array([singularity - self.V_extended[f[0]]])
                    #     )[0, 0]
                    #     first_row = [0, 0, 1, 0]
                    #     first_element = Uf[0].real
                    #     last_row = [zc.real + zc.imag, zc.real - zc.imag, 1, 1]
                    #     last_element = 0
                    # # If the face is not singular, the last row aligns the first corner 
                    # # value up to +-sign and scale, as for the other two corners
                    # else:
                    #     first_row = [Zf[0].real + Zf[0].imag, Zf[0].real - Zf[0].imag, 1, 1]
                    #     first_element = Uf[0].real + Uf[0].imag
                    #     last_row = [
                    #         centre_prod.imag, centre_prod.real, -centre_u.imag, centre_u.real
                    #         ]
                    #     last_element = 0

                    # # Construct the linear system (blocks of 4x4 in the matrix)
                    # lhs[4*j:4*(j+1), 4*j:4*(j+1)] = np.array([
                    #     first_row,
                    #     [Zf[1].real + Zf[1].imag, Zf[1].real - Zf[1].imag, 1, 1],
                    #     [Zf[2].real + Zf[2].imag, Zf[2].real - Zf[2].imag, 1, 1],
                    #     last_row
                    # ], dtype=float)
                    # rhs[4*j:4*(j+1)] = [
                    #     first_element,
                    #     Uf[1].real + Uf[1].imag,
                    #     Uf[2].real + Uf[2].imag,
                    #     last_element
                    # ]
                    #-------------------------------------------------------------------------#

                    #-------------------------------------------------------------------------#
                    # Corners by Re(az+b)Im(u) = Im(az+b)Re(u)
                    # Centre by Re(az+b) + Im(az+b) = Re(u) + Im(u)
                    # if j == f_singular:
                    #     zc = complex_projection(
                    #         b1, b2, normal, 
                    #         np.array([singularity - self.V_extended[f[0]]])
                    #     )[0, 0]
                    #     first_row = [0, 0, 1, 0]
                    #     first_element = Uf[0].real
                    #     last_row = [zc.real + zc.imag, zc.real - zc.imag, 1, 1]
                    #     last_element = 0
                    # else:
                    #     first_row = [prod[0].imag, prod[0].real, -Uf[0].imag, Uf[0].real]
                    #     first_element = 0
                    #     last_row = [
                    #         centre_z.real + centre_z.imag, centre_z.real - centre_z.imag, 1, 1
                    #         ]
                    #     last_element = centre_u.real + centre_u.imag
                    # lhs[4*j:4*(j+1), 4*j:4*(j+1)] = np.array([
                    #     first_row,
                    #     [prod[1].imag, prod[1].real, -Uf[1].imag, Uf[1].real],
                    #     [prod[2].imag, prod[2].real, -Uf[2].imag, Uf[2].real],
                    #     last_row
                    # ], dtype=float)
                    # rhs[4*j:4*(j+1)] = [
                    #     first_element,
                    #     0,
                    #     0,
                    #     last_element
                    # ]
                    #-------------------------------------------------------------------------#

                # 
                # if j == f_singular:
                #     zc = complex_projection(
                #         b1, b2, normal, 
                #         np.array([singularity - self.V_extended[f[0]]])
                #     )[0, 0]
                #     last_line = [zc.real + zc.imag, zc.real - zc.imag, 1, 1]
                #     # last_line = [zc, zc * 1j, 1, 1j]

                # else:
                #     last_line = [prod[0].imag, prod[0].real, -Uf[0].imag, Uf[0].real]
                
                # 
                # lhs[4*j:4*(j+1), 4*j:4*(j+1)] = np.array([
                #     [0, 0, 1, 0],
                #     [prod[1].imag, prod[1].real, -Uf[1].imag, Uf[1].real],
                #     [prod[2].imag, prod[2].real, -Uf[2].imag, Uf[2].real], 
                #     last_line
                # ], dtype=float)
                # rhs[4*j] = Uf[0].real

                # lhs[4*j:4*(j+1), 4*j:4*(j+1)] = np.array([
                #     [prod[0].imag, prod[0].real, -Uf[0].imag, Uf[0].real],
                #     [prod[1].imag, prod[1].real, -Uf[1].imag, Uf[1].real],
                #     [prod[2].imag, prod[2].real, -Uf[2].imag, Uf[2].real],
                #     [1, 0, 0, 0]
                # ], dtype=float)
                # rhs[4*j] = 1
            
            result, _, itns[i], r1norms[i] = lsqr(lhs.tocsr(), rhs)[:4]
            
            result = result.reshape(-1, 4)
            
            coeffs[:, i, 0] = result[:, 0] + 1j * result[:, 1]
            coeffs[:, i, 1] = result[:, 2] + 1j * result[:, 3]

        print(f'Linear field reconstruction iterations and residuals',
              np.stack([itns, r1norms], axis=1))
        
        return coeffs
    
    def define_linear_field(self, coeffs, indices_F=None):
        '''
            Define the linear field from the coefficients.
            Input:
                coeffs: (num_F, num_singularities, 2) complex array of coefficients for the linear fields
                indices_F: (num_singularities, ) array of indices for the singularities
            Output:
                linear_field: function of the linear field
        '''
        if indices_F is None:
            indices_F = self.indices_F
            
        if coeffs.shape[1] != len(indices_F):
            raise ValueError('The number of singularities and the number of sets of coefficients do not match.')
        
        indices_F = np.array(indices_F)

        def linear_field(posis):

            # Find the faces where the points are located
            # For points on vertices or edges, all adjacent faces are considered
            posis_extended = []
            F_involved = []
            for posi in posis:
                f_involved = is_in_face(self.V_extended, self.F_f, posi, include_EV=True)

                if len(f_involved) > 1:
                    posis_extended += [posi] * len(f_involved)
                    F_involved += f_involved
                else:
                    posis_extended.append(posi)
                    F_involved.append(f_involved)

            posis_extended = np.array(posis_extended)
            F_involved = np.array(F_involved).flatten()
            
            # Compute the linear field
            B1 = self.B1[F_involved]; B2 = self.B2[F_involved]; normals = self.normals[F_involved]
            
            Z = complex_projection(
                B1, B2, normals, 
                posis_extended - self.V_extended[self.F_f[F_involved, 0]], 
                diagonal=True
            )
            
            vectors_complex = np.prod(
                (coeffs[F_involved, :, 0] * Z[:, None] + coeffs[F_involved, :, 1]) ** indices_F[None, :],
                axis=1
            )

            vectors = B1 * vectors_complex.real[:, None] + B2 * vectors_complex.imag[:, None]

            return posis_extended, vectors
        
        return linear_field
    
    def vector_field(self, singularities, indices, v_init, z_init, six_eq_fit_linear=True):
        self.initialise_field_processing()
        
        Thetas = self.compute_thetas(singularities=singularities, indices=indices)
        
        Us = self.reconstruct_corners_from_thetas(Thetas, v_init, z_init)
        
        coeffs = self.reconstruct_linear_from_corners(Us, six_equations=six_eq_fit_linear)
        
        field = self.define_linear_field(coeffs)
        
        return field
    
    def vector_field_from_truth(self, coeffs_truth, singularities, indices, six_eq_fit_linear=True):
        '''
            Field - dictate a,b,c,d - thetas - reconstruct U - reconstruct coefficients
            Input: 
                coeffs: (num_F, num_singularities_F, 2) complex array of coefficients for the linear fields
            Output:
                field: function of the reconstructed vector field
                field_truth: function of the truth vector field
        '''
        self.initialise_field_processing()

        F_singular = []
        singularities_F = []
        indices_F = []

        if singularities is not None:
            for singularity, index in zip(singularities, indices):
                # If the singularity is in a face, it contributes a set of thetas
                F_candidate = is_in_face(self.V_extended, self.F_f, singularity)
                
                if F_candidate is not False:
                    F_singular.append(F_candidate)
                    singularities_F.append(singularity)
                    indices_F.append(index)

        F_singular = np.array(F_singular)
        singularities_F = np.array(singularities_F)
        indices_F = np.array(indices_F)

        if coeffs_truth.shape[1] != len(singularities_F):
            raise ValueError('The number of face singularities and the number of sets of coefficients do not match.')

        U_truth = np.zeros(len(self.V_extended), dtype=complex)
        
        for i, f in enumerate(self.F_f):
            b1 = self.B1[i][None, :]; b2 = self.B2[i][None, :]; normal = self.normals[i][None, :]

            V_f = self.V_extended[f]

            Z_f = complex_projection(b1, b2, normal, V_f - V_f[0])[0]

            # coeffs_truth[i, :, 0]: (num_singularities_F, )
            # Z_f: (3, )
            for j in range(3):
                U_truth[f[j]] = np.prod(
                    (coeffs_truth[i, :, 0] * Z_f[j] + coeffs_truth[i, :, 1]) ** indices_F
                )
        U_truth = U_truth / np.abs(U_truth)

        Theta = np.zeros((len(self.E_extended), 1))
        Theta[:, 0] = np.angle(U_truth[self.E_extended[:, 1]]) - np.angle(U_truth[self.E_extended[:, 0]])
        # Theta = np.mod(Theta + np.pi, 2 * np.pi) - np.pi

        Us = self.reconstruct_corners_from_thetas(Theta, v_init=0, z_init=U_truth[0], Thetas_include_pairface=True)

        # print(np.stack([U_truth, Us[:, 0]], axis=1))

        coeffs = self.reconstruct_linear_from_corners(Us, singularities_F, six_equations=six_eq_fit_linear) 

        # print('Truth and reconstructed coefficients: \n', np.stack([coeffs_truth, coeffs], axis=1))
        
        field_truth = self.define_linear_field(coeffs_truth, indices_F)

        field = self.define_linear_field(coeffs, indices_F)
        
        return field_truth, field




