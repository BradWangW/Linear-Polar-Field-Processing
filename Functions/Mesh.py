import numpy as np
from tqdm import tqdm
from scipy.sparse import lil_matrix, eye, bmat
from Functions.Auxiliary import (accumarray, find_indices, is_in_face, compute_planes, 
                                 complex_projection, obtain_E, compute_V_boundary, 
                                 compute_unfolded_vertex, compute_barycentric_coordinates, 
                                 compute_angle_defect)
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
        
        self.G_V = compute_angle_defect(V, F, self.V_boundary)
        
        self.B1, self.B2, self.normals = compute_planes(V, F)
        
    def initialise_field_processing(self):
        self.construct_extended_mesh()
        self.construct_d1_extended()
        self.compute_face_pair_rotation()
    
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
        
        # Mapping for efficient construction of d1
        F_v_map_E_comb = {}
        
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
                F_v_map_E_comb[len(F_v)] = np.arange(len(indices_sorted)) + len(E_comb)
                    
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
        self.F_v_map_E_comb = F_v_map_E_comb
        
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
        for i, f in tqdm(enumerate(self.F_e),
                         desc='Constructing d1 (edge faces)',
                         total=len(self.F_e),
                         leave=False):
            # Find the indices of the face edges in the edge list
            indices = np.where(np.all(np.isin(self.E_extended, f), axis=1))[0]
            E_f = self.E_extended[indices]
            
            for index, e in zip(indices, E_f):
                # If the edge is aligned with the face, the orientation is positive
                if (np.where(f == e[0])[0] == np.where(f == e[1])[0] - 1) or (f[-1] == e[0] and f[0] == e[1]):
                    d1[len(self.F_f) + i, index] = 1
                # If the edge is opposite to the face, the orientation is negative
                elif (np.where(f == e[0])[0] == np.where(f == e[1])[0] + 1) or (f[-1] == e[1] and f[0] == e[0]):
                    d1[len(self.F_f) + i, index] = -1
                else:
                    raise ValueError(f'The edge {e} is not in the face {f}, or the edge face is wrongly defined.')

        # The edges of the vertex faces are oriented counter-clockwisely (same as the vertex-faces)
        # so we directly use the mapping from the vertex faces to the combinatorial edges
        for i, f in tqdm(enumerate(self.F_v), 
                         desc='Constructing d1 (vertex faces)', 
                         total=len(self.F_v), 
                         leave=False):
            d1[
                len(self.F_f) + len(self.F_e) + i, 
                self.F_v_map_E_comb[i] + len(self.E_twin)
            ] = 1
            
        self.d1 = d1
        
    def compute_face_pair_rotation(self):
        '''
            Compute the rotation between the pair of faces sharing an edge.
        '''
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
            # f1 = self.F_f[f1_f]
            # f2 = self.F_f[f2_f]
        
            U = complex_projection(B1, B2, normals, vec_e[None, :])
            
            rotation = np.angle(U[1]) - np.angle(U[0])
            
            # The rotation is positive if the edge is aligned with the face
            if np.all(self.E_comb[e1_comb] == f_e[[0, 3]]):
                pair_rotations[e1_comb] = rotation
            # The rotation is negative if the edge is opposite to the face
            elif np.all(self.E_comb[e1_comb] == f_e[[3, 0]]):
                pair_rotations[e1_comb] = -rotation
            else:
                raise ValueError(f'{self.E_comb[e1_comb]} and {f_e[[0, 3]]} do not match.')
            
            # The rotation is positive if the edge is aligned with the face
            if np.all(self.E_comb[e2_comb] == f_e[[1, 2]]):
                pair_rotations[e2_comb] = rotation
            # The rotation is negative if the edge is opposite to the face
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
        self.I_F = np.zeros(len(self.F_f) + len(self.F_e) + len(self.F_v))
        
        Theta = np.zeros(len(self.E_extended))
        self.F_singular = []
        self.singularities_f = {}
        self.indices_f = {}
        
        mask_removed_f = np.ones(len(self.F_f) + len(self.F_e) + len(self.F_v), dtype=bool)
        mask_removed_e = np.ones(len(self.E_extended), dtype=bool)
        
        rhs_correction = np.zeros(len(self.F_f) + len(self.F_e) + len(self.F_v))
        
        # Function to compute rotations in singular faces and 
        # update the system for trivial connections
        def deal_singularity(f, singularity, index, in_face=False):
            if f not in self.F_singular:
                self.F_singular.append(f)
                self.singularities_f[f] = [singularity]
                self.indices_f[f] = [index]
            else:
                self.singularities_f[f].append(singularity)
                self.indices_f[f].append(index)
            
            # Find the edges of the face containing the singularity
            e_f = np.all(np.isin(self.E_extended, self.F_f[f]), axis=1)
        
            b1, b2, normal = self.B1[f], self.B2[f], self.normals[f]
            
            V1 = singularity - self.V_extended[self.E_extended[e_f, 0]]
            V2 = singularity - self.V_extended[self.E_extended[e_f, 1]]
            
            Z1 = complex_projection(b1[None, :], b2[None, :], normal[None, :], V1)
            Z2 = complex_projection(b1[None, :], b2[None, :], normal[None, :], V2)
            
            # If the singularity is inside the face
            if in_face:
                rotations = index * np.arccos(
                    np.real(np.conjugate(Z1) * Z2) / (np.abs(Z1) * np.abs(Z2))
                ).squeeze()
            # If the singularity is on the vertex/edge
            else:
                rotations = index * (np.angle(Z2) - np.angle(Z1)).squeeze()
                rotations = np.mod(rotations + np.pi, 2*np.pi) - np.pi
                
                if np.abs(np.abs(np.sum(rotations)) - 2*np.pi) < 1e-6:
                    rotations[np.argmax(np.abs(rotations))] *= -1
                if np.abs(np.sum(rotations)) > 1e-6:
                    raise ValueError(f'The total rotation {np.sum(rotations)} is not zero.')
            
            if np.any(np.linalg.norm(V1, axis=1) < 1e-6):
                v_singular = np.where(np.linalg.norm(V1, axis=1) < 1e-6)[0]
                rotations[v_singular] = 0
                rotations[v_singular-1] = 0
            
            Theta[e_f] += rotations            
            
            mask_removed_f[f] = False
            mask_removed_e[e_f] = False
            
            # For the other edge faces involving one of the computed edges, 
            # the rhs of the system needs to minus the rotation of that edge
            for j in range(3):
                e_involved = np.where(e_f)[0][j]
                
                f_involved = len(self.F_f) + np.where(
                    np.sum(np.isin(self.F_e, self.E_extended[e_f][j]), axis=1) == 2
                )[0][0]
                
                affect_in_d1 = -self.d1[f_involved, e_involved]
                
                rhs_correction[f_involved] += affect_in_d1 * rotations[j]
        
        for singularity, index in tqdm(zip(singularities, indices), 
                                       desc='Processing singularities and computing thetas', 
                                       total=len(singularities), 
                                       leave=False):
            
            in_F_v = np.where(
                np.all(np.isclose(self.V, singularity[None, :]), axis=1)
            )[0]
            
            # Check if the singularity is in an edge
            vec1 = self.V_extended[self.F_e[:, 0]] - singularity[None, :]
            vec2 = self.V_extended[self.F_e[:, 1]] - singularity[None, :]
            dot = np.einsum('ij,ij->i', vec1, vec2)
            obtuse = dot < 0
            parallel = np.isclose(
                np.abs(dot), np.linalg.norm(vec1, axis=1) * np.linalg.norm(vec2, axis=1)
            )
            in_F_e = np.where(parallel * obtuse)[0]
            
            # Check if the singularity is in a face
            in_F_f = is_in_face(self.V_extended, self.F_f, singularity)
            
            # If the singularity is in a vertex, it gives the thetas for the incident faces
            if len(in_F_v) == 1:
                # Loop over the faces incident to the vertex
                for f in np.where(np.any(np.isin(self.F, in_F_v), axis=1))[0]:
                    deal_singularity(f, singularity, index)
                
                for e_comb in self.F_v_map_E_comb[in_F_v[0]]:
                    f_e = np.where(
                        np.sum(np.isin(self.F_e, self.E_comb[e_comb]), axis=1) == 2
                    )[0]
                    
                    for idx in [[0, 3], [1, 2]]:
                        e_sub = np.where(
                            np.all(np.isin(self.E_extended, self.F_e[f_e][0, idx]), axis=1)
                        )[0]
                        mask_removed_e[e_sub] = False
                    
                    mask_removed_f[len(self.F_f) + f_e] = False
                
                mask_removed_f[len(self.F_f) + len(self.F_e) + in_F_v[0]] = False
                
            # If the singularity is in an edge, it gives the thetas for the incident faces
            elif len(in_F_e) == 1:
                # Loop over the two faces incident to the edge
                for i in range(2):
                    e = self.F_e[in_F_e[0]][2*i:2*(i+1)]
                    f = np.where(
                        np.sum(np.isin(self.F_f, e), axis=1) == 2
                    )[0][0]
                    
                    deal_singularity(f, singularity, index)
                    
                    comb = [[0, 3], [1, 2]][i]
                    e_comb = np.where(
                        np.all(np.isin(self.E_comb, self.F_e[in_F_e[0]][comb]), axis=1)
                    )[0]
                    mask_removed_e[e_comb + len(self.E_twin)] = False
                
                mask_removed_f[len(self.F_f) + in_F_e[0]] = False
                    
            # If the singularity is in a face, it gives thetas for the face
            elif in_F_f is not False:
                # Obtain the neighbour faces of the face containing the singularity
                # and the unfolded locations of the singularity on those faces
                deal_singularity(in_F_f, singularity, index, in_face=True)
                
                for i in range(3):
                    common_edge = np.stack([self.F[in_F_f], np.roll(self.F[in_F_f], -1)], axis=1)[i]
                    f_neighbour = np.where(np.sum(np.isin(self.F, common_edge), axis=1) == 2)[0]
                    f_neighbour = f_neighbour[f_neighbour != in_F_f][0]
                    
                    v_far = self.V[np.setdiff1d(self.F[f_neighbour], common_edge)].squeeze()
                    
                    singularity_unfolded = compute_unfolded_vertex(
                        v_far, self.V[common_edge[0]], self.V[common_edge[1]], singularity
                    )
                    
                    deal_singularity(f_neighbour, singularity_unfolded, index)
                            
            else:
                raise ValueError(f'The singularity {singularity} is not in any face, edge or vertex.')
                
        # Independent quantities for quadratic programming
        Q = eye(np.sum(mask_removed_e), format='lil')
        c = np.zeros(np.sum(mask_removed_e))
        
        # Add more penalty for the combinatorial edges to reduce jumps
        Q[len(self.E_twin):, len(self.E_twin):] *= 5
        Q = Q.tocoo()
        
        # Quantities for quadratic programming dependent on the singularities
        E = self.d1[mask_removed_f][:, mask_removed_e]
        d = (2 * np.pi * self.I_F - self.G_F + rhs_correction)[mask_removed_f]
            
        # Define the system to solve the quadratic programming problem
        KKT_lhs = bmat([
            [Q, E.T],
            [E, np.zeros((E.shape[0], E.shape[0]))]
        ], format='coo')
        KKT_rhs = np.concatenate([-c, d])
        
        # Solve the quadratic programming problem
        solution, _, itn, r1norm = lsqr(KKT_lhs, KKT_rhs)[:4]
        
        Theta[mask_removed_e] = solution[:np.sum(mask_removed_e)]
        
        print(f'Theta computation iteration and residual: {itn}, {r1norm}.')
        
        print(f'Total combinatorial rotations: {np.sum(np.abs(Theta[len(self.E_twin):]))}.')
        
        return Theta
    
    def reconstruct_corners_from_thetas(self, Theta, v_init, z_init, Theta_include_pairface=False):
        '''
            Reconstruct the corner values from the thetas.
            Input:
                Theta: (num_E, ) array of thetas
                v_init: initial value for the corner in the vertex face
                z_init: initial value for the corner in the edge face
            Output:
                Us: (num_V_extended, num_singularities) complex array of corner values
        '''
        if not Theta_include_pairface:
            Theta_with_pairface = Theta.copy()
            Theta_with_pairface[len(self.E_twin):] += self.pair_rotations
            # Theta = np.mod(Theta + np.pi, 2 * np.pi) - np.pi
            
        lhs = lil_matrix((len(self.E_extended) + 1, len(self.V_extended)), dtype=complex)
        lhs[np.arange(len(self.E_extended)), self.E_extended[:, 0]] = np.exp(1j * Theta_with_pairface)
        lhs[np.arange(len(self.E_extended)), self.E_extended[:, 1]] = -1
        lhs[-1, v_init] = 1
        
        rhs = np.zeros(len(self.E_extended) + 1, dtype=complex)
        rhs[-1] = z_init / np.abs(z_init)
        
        U, _, itn, r1norm = lsqr(lhs.tocsr(), rhs)[:4]
        
        U = U / np.abs(U)
            
        print(f'Corner reconstruction iterations and residuals',
              itn, r1norm)
        
        return U
    
    def subdivide_faces_over_pi(self, Theta, U):
        # Edges with >pi rotations cannot be handled by linear field, 
        # so subdivisions are needed
        self.F_over_pi = []
        num_subdivisions_f = {}
        
        Theta_over_pi = np.abs(Theta) > np.pi
        print('Number of (twin-edge) thetas over pi: ', np.sum(Theta_over_pi[:len(self.E_twin)]))
        
        # Loop over the edges with >pi rotations and find the faces to subdivide
        for e in np.where(Theta_over_pi[:len(self.E_twin)])[0]:
            f = np.where(
                np.sum(np.isin(self.F_f, self.E_twin[e]), axis=1) == 2
            )[0][0]
            
            # Only non-singular faces can be subdivided
            if f not in self.F_singular:
                # the number of subdivisions is based on the edge with the largest rotation
                if f not in self.F_over_pi:
                    self.F_over_pi.append(f)
                    num_subdivisions_f[f] = np.abs(Theta[e]) // np.pi + 1
                else:
                    if num_subdivisions_f[f] < np.abs(Theta[e]) // np.pi + 1:
                        num_subdivisions_f[f] = np.abs(Theta[e]) // np.pi + 1
        
        self.U_subdivided = {}
        self.F_subdivided = {}
        self.V_subdivided = {}
            
        # Loop over the faces to subdivide and perform the subdivision
        for f in self.F_over_pi:
            N = int(num_subdivisions_f[f])
            
            # Note the face is oriented v1 -> v2 -> v3
            # so the rotation is along v1 -> v2 -> v3
            V_f = self.V_extended[self.F_f[f]]
            U_f = U[self.F_f[f]]
            
            # Obtain the signs of the rotations in Theta
            signs = np.sign(Theta[np.where(np.all(np.isin(self.E_twin, self.F_f[f]), axis=1))[0]])
            
            # Rotations u1 -> u2, u2 -> u3, u3 -> u1
            rotations = (np.roll(np.angle(U_f), -1) - np.angle(U_f)).squeeze()
            rotations = np.mod(rotations + np.pi, 2*np.pi) - np.pi + 2 * np.pi * (N - 1) * signs
            
            self.F_subdivided[f] = []
            self.V_subdivided[f] = []
            self.U_subdivided[f] = []

            # Step 2: Create the edge points
            for i in range(N + 1):
                for j in range(N + 1 - i):
                    u = i / N
                    v = j / N
                    w = (N - i - j) / N
                    
                    self.V_subdivided[f].append(u * V_f[0] + v * V_f[1] + w * V_f[2])
                    self.U_subdivided[f].append(U_f[2] * np.exp(-1j * rotations[1] * v) * np.exp(-1j * rotations[0] * u))
            
            # Step 3: Create the list of triangles
            for i in range(N):
                for j in range(N - i):
                    # Calculate indices for the top triangle
                    idx1 = (i * (N + 1) - (i * (i - 1)) // 2) + j
                    idx2 = idx1 + 1
                    idx3 = idx1 + (N + 1 - i)
                    self.F_subdivided[f].append([idx1, idx2, idx3])
                    if j < N - i - 1:
                        # Calculate indices for the bottom triangle
                        idx4 = idx2
                        idx5 = idx3
                        idx6 = idx3 + 1
                        self.F_subdivided[f].append([idx4, idx5, idx6])
            
            self.F_subdivided[f] = np.array(self.F_subdivided[f])
            self.V_subdivided[f] = np.array(self.V_subdivided[f])
            self.U_subdivided[f] = np.array(self.U_subdivided[f])
    
    def reconstruct_linear_from_corners(self, U):
        '''
            Reconstruct the coefficients of the linear field from the corner values.
            Input:
                Us: (num_F) complex array of corner values
                singularities_f: (num_singularities, 3) array of singularities in the faces
            Output:
                coeffs: (num_F, 2) complex array of coefficients for the linear fields
        '''
        coeffs = np.zeros((len(self.F_f), 2), dtype=complex)
        coeffs_singular = {}
        coeffs_subdivided = {}
        total_err = 0
        mean_itn = 0

        for i, f in tqdm(enumerate(self.F_f),
                         desc=f'Reconstructing linear field coefficients',
                         total=len(self.F_f),
                         leave=False):
            b1 = self.B1[i][None, :]; b2 = self.B2[i][None, :]; normal = self.normals[i][None, :]
            
            # Compute the complex representation of the vertices on the face face
            Zf = complex_projection(b1, b2, normal, self.V_extended[f] - self.V_extended[f[0]])[0]
            
            Uf = U[f]
            prod = np.conjugate(Uf) * Zf
            
            z_centre = np.mean(Zf)
            U_centre = np.mean(Uf)
            U_centre = U_centre / np.abs(U_centre)

            # If the face is singular, the last row explicitly specifies 
            # the singularity (zero point of the field)
            if i in self.F_singular:
                coeffs_f = np.zeros((len(self.singularities_f[i]), 2), dtype=complex)
                sub_itn = 0
                
                # Divide the argument of the firsr corner by the number of singularities on that face
                # so that the first corner, after the multiplicative superposition,
                # aligns with the first corner of the face
                uf0 = np.exp(1j * np.angle(U_centre) / len(self.singularities_f[i]))
                for j, (singularity, index) in enumerate(zip(self.singularities_f[i], self.indices_f[i])):
                    zc = complex_projection(
                        b1, b2, normal, 
                        np.array([singularity - self.V_extended[f[0]]])
                    )[0, 0]
                    
                    if index == 1:
                        lhs = np.array([
                            [zc.real, -zc.imag, 1, 0],
                            [zc.imag, zc.real, 0, 1],
                            [z_centre.real, -z_centre.imag, 1, 0],
                            [z_centre.imag, z_centre.real, 0, 1]
                        ], dtype=float)
                    elif index == -1:
                        lhs = np.array([
                            [zc.real, zc.imag, 1, 0],
                            [-zc.imag, zc.real, 0, 1],
                            [z_centre.real, z_centre.imag, 1, 0],
                            [-z_centre.imag, z_centre.real, 0, 1]
                        ], dtype=float)
                    else:
                        raise ValueError('The field cannot handle face singularities with index > 1 or < -1 yet.')
                    
                    rhs = np.array([
                        0, 0, uf0.real, uf0.imag
                    ])
                    
                    result, _, itn, err = lsqr(lhs, rhs)[:4]
                    coeffs_f[j, 0] = result[0] + 1j * result[1]
                    coeffs_f[j, 1] = result[2] + 1j * result[3]
                    
                    total_err += err
                    sub_itn += itn/(len(self.F_f) * len(self.singularities_f[i]))
                
                coeffs_singular[i] = coeffs_f
                mean_itn += sub_itn
                
            # If the face has edge rotation > pi
            elif i in self.F_over_pi:
                v0 = self.V_extended[f[0]]
                coeffs_f = np.zeros((len(self.F_subdivided[i]), 2), dtype=complex)
                sub_itn = 0
                
                # For each subdivided face, the last row aligns the first corner
                for j, f_sub in enumerate(self.F_subdivided[i]):
                    Zf_sub = complex_projection(
                        b1, b2, normal, self.V_subdivided[i][f_sub] - v0
                    )[0]
                    
                    Uf_sub = self.U_subdivided[i][f_sub]
                    prod_sub = np.conjugate(Uf_sub) * Zf_sub
                    
                    lhs = np.array([
                        [prod_sub[0].imag, prod_sub[0].real, -Uf_sub[0].imag, Uf_sub[0].real],
                        [prod_sub[1].imag, prod_sub[1].real, -Uf_sub[1].imag, Uf_sub[1].real],
                        [Zf_sub[0].real, -Zf_sub[0].imag, 1, 0],
                        [Zf_sub[0].imag, Zf_sub[0].real, 0, 1]
                    ], dtype=float)
                    rhs = np.array([
                        0, 0, Uf_sub[0].real, Uf_sub[0].imag
                    ])
                    
                    result, _, itn, err = lsqr(lhs, rhs)[:4]
                    coeffs_f[j, 0] = result[0] + 1j * result[1]
                    coeffs_f[j, 1] = result[2] + 1j * result[3]
                    
                    total_err += err
                    sub_itn += itn/(len(self.F_f) * len(self.F_subdivided[i]))
                
                coeffs_subdivided[i] = coeffs_f
                mean_itn += sub_itn
                
            # If the face is not singular, the last row aligns the first corner 
            # value up to +-sign and scale, as for the other two corners
            else:
                lhs = np.array([
                    [prod[2].imag, prod[2].real, -Uf[2].imag, Uf[2].real],
                    [prod[1].imag, prod[1].real, -Uf[1].imag, Uf[1].real],
                    [Zf[0].real, -Zf[0].imag, 1, 0],
                    [Zf[0].imag, Zf[0].real, 0, 1]
                ], dtype=float)
                rhs = np.array([
                    0, 0, Uf[0].real, Uf[0].imag
                ])

                result, _, itn, err = lsqr(lhs, rhs)[:4]
                coeffs[i, 0] = result[0] + 1j * result[1]
                coeffs[i, 1] = result[2] + 1j * result[3]
                total_err += err
                mean_itn += itn/len(self.F_f)

        print(f'Linear field reconstruction mean iterations and total residuals',
              mean_itn, total_err)
        
        return coeffs, coeffs_singular, coeffs_subdivided
    
    def define_linear_field(self, coeffs, coeffs_singular, coeffs_subdivided):
        '''
            Define the linear field from the coefficients.
            Input:
                coeffs: (num_F, num_singularities, 2) complex array of coefficients for the linear fields
                indices_f: (num_singularities, ) array of indices for the singularities
            Output:
                linear_field: function of the linear field
        '''

        def linear_field(posis):

            # Find the faces where the points are located
            # For points on vertices or edges, all adjacent faces are considered
            # For points in faces with more than one singularities,
            # multiplicative fields are computed
            posis_extended = []
            F_involved = []
            idx_singular = []
            idx_subdivided = []
            
            for posi in tqdm(posis, 
                             desc='Computing the linear field at the points', 
                             total=len(posis), 
                             leave=False):
                f_involved = is_in_face(self.V_extended, self.F_f, posi, include_EV=True)
                
                for i, f in enumerate(f_involved):
                    if f in self.F_singular:
                        idx_singular.append(len(posis_extended) + i)
                    elif f in self.F_over_pi:
                        idx_subdivided.append(len(posis_extended) + i)
                
                posis_extended += [posi] * len(f_involved)
                F_involved += f_involved

            posis_extended = np.array(posis_extended)
            F_involved = np.array(F_involved).flatten()
            mask = np.zeros(len(posis_extended), dtype=bool)
            mask[idx_singular] = True
            mask[idx_subdivided] = True
            
            # Compute the linear field
            B1 = self.B1[F_involved]; B2 = self.B2[F_involved]; normals = self.normals[F_involved]
            
            Z = complex_projection(
                B1, B2, normals, 
                posis_extended - self.V_extended[self.F_f[F_involved, 0]], 
                diagonal=True
            )
            
            vectors_complex = np.zeros(len(posis_extended), dtype=complex)
            vectors_complex[~mask] = coeffs[F_involved[~mask], 0] * Z[~mask] + coeffs[F_involved[~mask], 1]
            
            for i in idx_singular:
                f = F_involved[i]
                prod = 1
                for j, index in enumerate(self.indices_f[f]):
                    coeff_singular = coeffs_singular[f][j]
                    if index == 1:
                        prod *= coeff_singular[0] * Z[i] + coeff_singular[1]
                    elif index == -1:
                        prod *= coeff_singular[0] * np.conjugate(Z[i]) + coeff_singular[1]
                    else:
                        raise ValueError('The field cannot handle face singularities with index > 1 or < 1 yet.')
                    
                vectors_complex[i] = prod
            
            for i in idx_subdivided:
                f = F_involved[i]
                
                f_sub_involved = is_in_face(self.V_subdivided[f], self.F_subdivided[f], posis_extended[i], include_EV=True)
                
                vectors_complex[i] = coeffs_subdivided[f][f_sub_involved, 0] * Z[i] + coeffs_subdivided[f][f_sub_involved, 1]
            
            vectors_complex = vectors_complex

            vectors = B1 * vectors_complex.real[:, None] + B2 * vectors_complex.imag[:, None]

            return posis_extended, vectors
        
        return linear_field
    
    def vector_field(self, singularities, indices, v_init, z_init):
        Theta = self.compute_thetas(singularities=singularities, indices=indices)
        
        U = self.reconstruct_corners_from_thetas(Theta, v_init, z_init)
        
        self.subdivide_faces_over_pi(Theta, U)
        
        coeffs, coeffs_singular, coeffs_subdivided = self.reconstruct_linear_from_corners(U)
            
        field = self.define_linear_field(coeffs, coeffs_singular, coeffs_subdivided)
        
        return field
    
    def vector_field_from_truth(self, coeffs_truth, singularities, indices):
        '''
            Field - dictate a,b,c,d - thetas - reconstruct U - reconstruct coefficients
            Input: 
                coeffs: (num_F, num_singularities_f, 2) complex array of coefficients for the linear fields
            Output:
                field: function of the reconstructed vector field
                field_truth: function of the truth vector field
        '''
        self.initialise_field_processing()

        F_singular = []
        singularities_f = []
        indices_f = []

        if singularities is not None:
            for singularity, index in zip(singularities, indices):
                # If the singularity is in a face, it contributes a set of thetas
                F_candidate = is_in_face(self.V_extended, self.F_f, singularity)
                
                if F_candidate is not False:
                    F_singular.append(F_candidate)
                    singularities_f.append(singularity)
                    indices_f.append(index)

        F_singular = np.array(F_singular)
        singularities_f = np.array(singularities_f)
        indices_f = np.array(indices_f)

        U_truth = np.zeros(len(self.V_extended), dtype=complex)
        
        for i, f in enumerate(self.F_f):
            b1 = self.B1[i][None, :]; b2 = self.B2[i][None, :]; normal = self.normals[i][None, :]

            V_f = self.V_extended[f]

            Z_f = complex_projection(b1, b2, normal, V_f - V_f[0])[0]

            # coeffs_truth[i, :, 0]: (num_singularities_f, )
            # Z_f: (3, )
            for j in range(3):
                U_truth[f[j]] = coeffs_truth[i, 0] * Z_f[j] + coeffs_truth[i, 1]
        U_truth = U_truth / np.abs(U_truth)

        Theta = np.angle(U_truth[self.E_extended[:, 1]]) - np.angle(U_truth[self.E_extended[:, 0]])
        # Theta = np.mod(Theta + np.pi, 2 * np.pi) - np.pi

        U = self.reconstruct_corners_from_thetas(Theta, v_init=0, z_init=U_truth[0], Theta_include_pairface=True)

        coeffs = self.reconstruct_linear_from_corners(U, singularities_f) 

        field_truth = self.define_linear_field(coeffs_truth, indices_f)

        field = self.define_linear_field(coeffs, indices_f)
        
        return field_truth, field

    def sample_points_and_vectors(self, field, num_samples=3, margin = 0.15, singular_detail=False, num_samples_detail=10, margin_detail=0.05):
        points = []
        margins = [margin] * len(self.F)
        nums_samples = [num_samples] * len(self.F)
        
        if singular_detail:
            for f in self.F_singular + self.F_over_pi:
                nums_samples[f] = num_samples_detail
                margins[f] = margin_detail
        
        for i, f in tqdm(enumerate(self.F), desc='Sampling points and vectors', 
                         total=len(self.F), leave=False):
            num_samples = nums_samples[i]
            margin = margins[i]
            for j in range(num_samples):
                for k in range(num_samples - j):
                    # Barycentric coordinates
                    u = margin + (j / (num_samples-1)) * (1 - 3 * margin)
                    v = margin + (k / (num_samples-1)) * (1 - 3 * margin)
                    w = 1 - u - v
                    
                    # Interpolate to get the 3D point in the face
                    points.append(
                        u * self.V[f[0]] + v * self.V[f[1]] + w * self.V[f[2]]
                    )
                    
        points = np.array(points)

        posis, vectors = field(points)
        
        return posis, vectors
    





