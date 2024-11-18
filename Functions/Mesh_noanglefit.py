import numpy as np
from tqdm import tqdm
from scipy.sparse import lil_matrix, eye, bmat, diags, vstack, hstack, csc_matrix, csr_matrix
from Functions.Auxiliary import (accumarray, find_indices, is_in_face, compute_planes, 
                                 complex_projection, obtain_E, compute_V_boundary, 
                                 compute_unfolded_vertex, compute_barycentric_coordinates, 
                                 compute_angle_defect, get_memory_usage, solve_real_modular_system)
from scipy.sparse.linalg import lsqr, spsolve, spilu, LinearOperator, eigsh, gmres, splu, svds
from scipy.optimize import minimize
from numpy.linalg import matrix_rank
import networkx as nx
import random
import polyscope as ps
import polyscope.imgui as psim
import time


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
        
        self.genus = (2 - (V.shape[0] - self.E.shape[0] + F.shape[0])) / 2
        
        self.B1, self.B2, self.normals = compute_planes(V, F)
        
        print('Genus of the mesh:', self.genus)
        
    def get_E_dual(self):
        '''
        Construct the dual edges of the self.
        '''
        E_dual = np.zeros((len(self.E), 2), dtype=int)
        
        # Construct the dual edges
        for i, edge in tqdm(enumerate(self.E), 
                            desc='Constructing dual edges',
                            total=len(self.E),
                            leave=False):
            faces = np.where(np.isin(self.F, edge).sum(axis=1) == 2)[0] 
            
            if len(faces) == 2:
                E_dual[i] = faces
            else:
                raise ValueError(f'Wrong number of faces found for edge {edge}: {faces}.')
        
        return E_dual

    def get_homology_basis(self):
        E_tuple = [tuple(e) for e in self.E]
        E_dual = self.get_E_dual()
        
        # Create a graph from the mesh edges
        G = nx.Graph()
        G.add_edges_from(E_tuple)
        
        T = nx.minimum_spanning_tree(G)
        T_arr = np.array(T.edges())
        
        E_included = np.any((self.E[:, None] == T_arr).all(-1) | 
                            (self.E[:, None] == T_arr[:, ::-1]).all(-1), axis=1)
        E_dual_tuple = [tuple(e) for e in E_dual[~E_included]]
        
        # Construct the dual graph, where the edges 
        # of the previous spanning tree are removed
        G_dual = nx.Graph()
        G_dual.add_edges_from(E_dual_tuple)
        T_dual = nx.minimum_spanning_tree(G_dual)
        T_dual_arr = np.array(T_dual.edges())
        
        E_dual_included = np.any((E_dual[:, None] == T_dual_arr).all(-1) | 
                                 (E_dual[:, None] == T_dual_arr[:, ::-1]).all(-1), axis=1)
        
        E_either_included = E_included | E_dual_included
        
        E_co = self.E[~E_either_included]
        
        if len(E_co) != 2*self.genus:
            raise ValueError(f"Expected {2*self.genus} non-contractible edges, but found {len(E_co)}")
        
        # List to store non-contractible cycles
        cycles = []
        G_H = []

        for cotree_edge in tqdm(E_co, 
                                desc="Finding non-contractible cycles", 
                                total=len(E_co),
                                leave=False):
            # Add the cotree edge back to form a cycle
            T.add_edge(*cotree_edge)
            
            cycle = nx.find_cycle(T, source=cotree_edge[0])
            
            # Find the cycle created by adding this edge
            cycles.append(cycle)
            
            # Remove the edge again to restore the tree
            T.remove_edge(*cotree_edge)

        return cycles
        
    def get_V_F_f_E_twin(self):
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
                self.V_extended.append(self.V[v])
                
                # Index of the extended vertex
                index_extended = len(self.V_extended) - 1
                f_f.append(index_extended)
                
                self.V_map[v].append(index_extended)

            # Add the face-face
            self.F_f[i] = f_f
            
            # Add the twin edges
            for j, k in np.stack([np.arange(3), np.roll(np.arange(3), -1)], axis=1):
                self.E_twin[i * 3 + j] = [f_f[j], f_f[k]]
                self.d1[i, i * 3 + j] = 1
                
                self.E_twin_dict[tuple([f_f[j], f_f[k]])] = i * 3 + j
            
        self.V_extended = np.array(self.V_extended)
    
    def get_F_e_E_comb(self):
        # metric M_G
        d_G = lil_matrix((len(self.E), len(self.E) * 4), dtype=complex)
        Area_inv = lil_matrix((len(self.E), len(self.E)), dtype=complex)
        
        # Construct edge-faces
        for i, e in tqdm(enumerate(self.E), 
                        desc='Constructing edge-faces and combinatorial edges',
                        leave=False,
                        total=self.E.shape[0]):
            indices1_extended = self.V_map[e[0]]
            indices2_extended = self.V_map[e[1]]
            
            # In a triangle mesh, two faces share at most one edge,
            # so when extracting the twin edges that encompass both extended_vertices of v1 and v2,
            # Either one or two edges will be found.
            e_twin = np.where(np.all(
                np.isin(self.E_twin, indices1_extended + indices2_extended), 
                axis=1
            ))[0]
            
            # If two edges are found, the edge is an interior edge
            # in which case the 4 vertices give an edge-face
            if len(e_twin) == 2:
                # Check if the twin edges are aligned or opposite
                pairing = np.isin(self.E_twin[e_twin], indices1_extended)
                
                # The twin edges need to be oppositely oriented
                if np.all(pairing[0] == pairing[1]):
                    raise ValueError('Detected aligned two edges for an edge face. \n All faces need to be consistently oriented.')

                # Twin-edge-related quantities
                edges = self.E_twin[e_twin]
                self.F_e[i] = edges[0].tolist() + edges[1].tolist()
                self.d1[len(self.F_f) + i, e_twin] = 1
                
                # Corresponding comb edges
                self.E_comb += [
                    [edges[0][1], edges[1][0]],
                    [edges[1][1], edges[0][0]]
                ]
                
                f = np.where(np.any(np.isin(self.F_f, edges[1][0]), axis=1))[0]
                g = np.where(np.any(np.isin(self.F_f, edges[0][0]), axis=1))[0]
                
                other_v_f = np.setdiff1d(self.F_f[f], [edges[1][1], edges[1][0]])[0]
                other_v_g = np.setdiff1d(self.F_f[g], [edges[0][0], edges[0][1]])[0]
                
                e_i_j, e_j_k, e_kf_if, e_k_l, e_l_i, e_ig_kg = self.E_twin_dict[tuple([edges[1][1], other_v_f])],\
                    self.E_twin_dict[tuple([other_v_f, edges[1][0]])],\
                    self.E_twin_dict[tuple([edges[1][0], edges[1][1]])],\
                    self.E_twin_dict[tuple([edges[0][1], other_v_g])],\
                    self.E_twin_dict[tuple([other_v_g, edges[0][0]])],\
                    self.E_twin_dict[tuple([edges[0][0], edges[0][1]])]
                    
                e_if_ig, e_kg_kf = len(self.E_twin) + len(self.E_comb) - 1, len(self.E_twin) + len(self.E_comb) - 2
            
                posi_l_unfolded = compute_unfolded_vertex(self.V_extended[other_v_f], self.V_extended[edges[1][1]], self.V_extended[edges[1][0]], self.V_extended[other_v_g])
                
                vec_ij = self.V_extended[other_v_f] - self.V_extended[edges[1][1]]
                vec_jk = self.V_extended[edges[1][0]] - self.V_extended[other_v_f]
                vec_ik = self.V_extended[edges[1][1]] - self.V_extended[edges[1][0]]
                vec_kl = posi_l_unfolded - self.V_extended[edges[0][1]]
                vec_li = self.V_extended[edges[0][0]] - posi_l_unfolded
                
                z_ij, z_jk, z_ik, z_kl, z_li = complex_projection(self.B1[f], self.B2[f], self.normals[f],
                                    [vec_ij, vec_jk, vec_ik, vec_kl, vec_li]).squeeze()
                
                Area_inv[i, i] = 2/(np.linalg.norm(np.cross(vec_ij, vec_jk)) + np.linalg.norm(np.cross(vec_kl, vec_li)))
                
                d_G[i, e_i_j] = - (z_ik + z_jk)/2
                d_G[i, e_j_k] = (z_ij + z_ik)/2
                d_G[i, e_kf_if] = (z_jk - z_ij)/2
                d_G[i, e_k_l] = (z_ik - z_li)/2
                d_G[i, e_l_i] = (-z_ik + z_kl)/2
                d_G[i, e_ig_kg] = (z_li - z_kl)/2
                d_G[i, e_if_ig] = z_ik * np.exp(-1j * np.pi/2)
                d_G[i, e_kg_kf] = z_ik * np.exp(1j * np.pi/2)
                
                self.d1[len(self.F_f) + i, len(self.E_twin) + len(self.E_comb) - 2] = 1
                self.d1[len(self.F_f) + i, len(self.E_twin) + len(self.E_comb) - 1] = 1

                vec_e = self.V_extended[edges[0][1]] - self.V_extended[edges[0][0]]
                
                f1_f = np.where(np.any(np.isin(self.F_f, edges[0][0]), axis=1))[0]
                f2_f = np.where(np.any(np.isin(self.F_f, edges[1][0]), axis=1))[0]
                
                B1, B2, normals = (self.B1[[f1_f, f2_f]].squeeze(), 
                                self.B2[[f1_f, f2_f]].squeeze(), 
                                self.normals[[f1_f, f2_f]].squeeze())
                
                self.U = complex_projection(B1, B2, normals, vec_e[None, :])
                
                rotation = np.angle(self.U[1] / self.U[0])
                
                self.pair_rotations += [
                    rotation, -rotation
                ]
                
            # If one edge is found, the edge is a boundary edge, 
            # in which case no edge-face is formed
            elif e_twin.shape[0] == 1:
                pass
            else:
                raise ValueError(f'Wrong number of twin edges found: {e_twin}.')
        
        self.E_extended = np.concatenate([self.E_twin, self.E_comb])
        M_G_complex = d_G.real.T @ Area_inv @ d_G.real
        self.M_G = bmat([
            [M_G_complex.real, -M_G_complex.imag],
            [M_G_complex.imag, M_G_complex.real]
        ], format='csc')

    def get_F_v(self):
        # Construct vertex-faces and combinatorial edges
        for v, indices_extended in tqdm(self.V_map.items(), 
                                        desc='Constructing vertex-faces',
                                        leave=False):
            # Find the neighbours of the vertex
            edges = np.where(
                np.all(
                    np.isin(self.E_extended, indices_extended),
                    axis=1
                )
            )[0]
            
            self.d1[len(self.F_f) + len(self.F_e) + len(self.F_v), edges] = 1
            
            E_comb_surround = self.E_extended[edges]
            
            # Sort the neighbours so that they form a Hamiltonian cycle
            order = np.arange(len(E_comb_surround))
            
            for i in range(1, len(E_comb_surround)):
                next_edge = np.where(
                    np.any(
                        np.isin(E_comb_surround, E_comb_surround[i-1][-1]),
                        axis=1
                    )
                )[0]
                next_edge = np.setdiff1d(next_edge, i-1)[0]
                
                if i != next_edge:
                    E_comb_surround[[i, next_edge]] = E_comb_surround[[next_edge, i]]
                    order[[i, next_edge]] = order[[next_edge, i]]
            
            indices_sorted = [indices_extended[i] for i in order]
            
            # Only if the vertex is adjacent to > 2 faces,
            # the vertex-face is constructed
            # Otherwise, only one combinatorial edge is formed
            if len(indices_extended) > 2:
                # Add the vertex-face
                self.F_v.append(indices_sorted)

            else:
                raise ValueError(f'Wrong number of extended vertices found for {v}: {indices_extended}. \n Boundary vertices are not supported.')
        
        self.d1 = self.d1.tocsr()
        
    def get_preconds(self):
        # Theta computation
        self.pair_rotations = np.array(self.pair_rotations).squeeze()
        self.G_F = np.concatenate([
            np.zeros(len(self.F_f) + len(self.F_e)),
            self.G_V
        ])
        
        d1_zero_imag = bmat([
            [self.d1, None],
            [None, eye(len(self.E_extended))]
        ], format='csc')
        
        self.A_Theta_KKT = bmat([
            [self.M_G.tocsr(), d1_zero_imag.T],
            [d1_zero_imag, None]
        ], format='csc')
        self.M_Theta_KKT = splu(self.A_Theta_KKT + 1e-8 * eye(self.A_Theta_KKT.shape[0], format='csc'))
        
        Q = eye(len(self.E_extended), format='lil')
        self.A_Theta_KKT_identity_metric = bmat([
            [Q, self.d1.T],
            [self.d1, np.zeros((self.d1.shape[0], self.d1.shape[0]))]
        ], format='csc')
        self.M_Theta_KKT_identity_metric = splu(self.A_Theta_KKT_identity_metric + 1e-8 * eye(self.A_Theta_KKT_identity_metric.shape[0], format='csc'))
        
        Q = eye(len(self.E_extended), format='lil') * 5
        Q[np.arange(len(self.E_twin)), np.arange(len(self.E_twin))] = 1
        self.A_Theta_KKT_punishing_comb = bmat([
            [Q, self.d1.T],
            [self.d1, np.zeros((self.d1.shape[0], self.d1.shape[0]))]
        ], format='csc')
        self.M_Theta_KKT_punishing_comb = splu(self.A_Theta_KKT_punishing_comb + 1e-8 * eye(self.A_Theta_KKT_identity_metric.shape[0], format='csc'))
        
        # Corner scale computation
        self.A_Corner_scale = lil_matrix((len(self.E_extended) + 1, len(self.V_extended)))
        self.A_Corner_scale[np.arange(len(self.E_extended)), self.E_extended[:, 0]] = 1
        self.A_Corner_scale[np.arange(len(self.E_extended)), self.E_extended[:, 1]] = -1
        self.A_Corner_scale[-1, 0] = 1
        
        A_Corner_scale_augmented = bmat([
            [None, self.A_Corner_scale],
            [self.A_Corner_scale.T, None]
        ], format='csc')
        
        self.lu_Corner_scale_augmented = splu(A_Corner_scale_augmented + 1e-8 * eye(A_Corner_scale_augmented.shape[0], format='csc'))
        
        # Linear coefficient computation
        A_Coeff = lil_matrix((len(self.F_f) * 6, len(self.F_f) * 6), dtype=float)
        for i, f in tqdm(enumerate(self.F_f),
                         desc=f'Pre-conditioning linear field coefficient system',
                         total=len(self.F_f),
                         leave=False):
            b1 = self.B1[i][None, :]; b2 = self.B2[i][None, :]; normal = self.normals[i][None, :]
            
            # Compute the complex representation of the vertices on the face face
            Zf = complex_projection(b1, b2, normal, self.V_extended[f] - self.V_extended[f[0]])[0]
            
            A_Coeff[6*i:6*(i+1), 6*i:6*(i+1)] = np.array([
                [Zf[0].real, -Zf[0].imag, Zf[0].real, Zf[0].imag, 1, 0],
                [Zf[0].imag, Zf[0].real, -Zf[0].imag, Zf[0].real, 0, 1],
                [Zf[1].real, -Zf[1].imag, Zf[1].real, Zf[1].imag, 1, 0],
                [Zf[1].imag, Zf[1].real, -Zf[1].imag, Zf[1].real, 0, 1],
                [Zf[2].real, -Zf[2].imag, Zf[2].real, Zf[2].imag, 1, 0],
                [Zf[2].imag, Zf[2].real, -Zf[2].imag, Zf[2].real, 0, 1]
            ], dtype=float)
        
        self.M_Coeff = splu(A_Coeff.tocsc())
    
    def construct_extended_mesh(self):
        '''
        Construct the extended mesh from the input mesh, which consists of 
            the extended vertices, twin edges and combinatorial edges,
            face-faces, edge-faces, and vertex-faces.
        '''
        
        self.V_extended = []
        self.E_twin = np.zeros((self.F.shape[0] * 3, 2), dtype=int)
        self.E_comb = []
        self.F_f = np.zeros((self.F.shape[0], 3), dtype=int)
        self.F_e = np.zeros((self.E.shape[0], 4), dtype=int)
        self.F_v = []
        
        self.d1 = lil_matrix((len(self.F) + len(self.E) + len(self.V), 4 * len(self.E)), dtype=float)
        self.pair_rotations = []
        self.V_map = {v:[] for v in range(self.V.shape[0])}
        self.E_twin_dict = {}
        
        for step in tqdm([self.get_V_F_f_E_twin, 
                          self.get_F_e_E_comb, 
                          self.get_F_v, 
                        #   self.construct_M_G,
                          self.get_preconds], 
                         desc='Constructing the extended mesh',
                         leave=True):
            step()
        
        self.I_F_pre = np.zeros(len(self.F_f) + len(self.F_e) + len(self.F_v))
        self.Corner_arg_pre = np.zeros(len(self.V_extended))        
        self.Corner_scale_pre = np.zeros(len(self.V_extended))
        
        self.Corner_arg_harmonic = self.lu_Corner_scale_augmented.solve(
            np.zeros(len(self.E_extended) + len(self.V_extended) + 1)
        )
        self.Corner_scale_harmonic = self.lu_Corner_scale_augmented.solve(
            np.concatenate([
                np.append(np.zeros(len(self.E_extended), dtype=float), 1), 
                np.zeros(len(self.V_extended), dtype=float)
            ])
        )[len(self.E_extended) + 1:]
        self.metric = 'G'
        self.metric_pre = None
        
    def process_singularities(self, singularities=None, indices=None):
        if len(singularities) != len(indices):
            raise ValueError('The number of singularities and the number of indices do not match.')
        if np.sum(indices) != 2 - 2*self.genus:
            raise ValueError('The sum of indices is not equal to 2 - 2g.')
        
        self.singularities = singularities[np.array(indices) != 0]
        self.indices = indices[np.array(indices) != 0]
        
        self.F_extended_singular = np.zeros(len(self.singularities), dtype=int)
        self.mask_removed_v = np.ones(len(self.V_extended), dtype=bool)
        self.I_F = np.zeros(len(self.F_f) + len(self.F_e) + len(self.F_v))
        self.E_singular = np.zeros((len(self.singularities), 2), dtype=int)
        
        self.F_f_singular = []
        
        for i, singularity in tqdm(enumerate(self.singularities), 
                                       desc='Processing singularities', 
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
                self.I_F[len(self.F_f) + len(self.F_e) + in_F_v[0]] = self.indices[i]
                
                self.F_extended_singular[i] = len(self.F_f) + len(self.F_e) + in_F_v[0]
                    
            # If the singularity is in an edge, it gives the thetas for the incident faces
            elif len(in_F_e) == 1:
                self.F_extended_singular[i] = len(self.F_f) + in_F_e[0]
                
                self.I_F[self.F_extended_singular[i]] = -1 * self.indices[i]
                
                self.E_singular[i] = np.where(
                    np.sum(np.isin(self.E_twin, self.F_e[in_F_e]), axis=1) == 2
                )[0]
                
                fs = np.where(
                    np.sum(np.isin(self.F_f, self.F_e[in_F_e]), axis=1) == 2
                )[0]
                
                self.F_f_singular += fs.tolist()
                
            # If the singularity is in a face, it gives thetas for the face
            elif in_F_f is not False:
                self.F_extended_singular[i] = in_F_f
                
                self.I_F[self.F_extended_singular[i]] = self.indices[i]
                
                self.mask_removed_v[self.F_f[in_F_f]] = False
                
                self.F_f_singular += [in_F_f]
                    
            else:
                raise ValueError(f'The singularity {singularity} is not in any face, edge or vertex.')
            
    def compute_thetas(self):
        '''
            Compute the set of thetas for each face singularity 
            by constrained optimisation using the KKT conditions.
        '''
        if self.metric_pre != self.metric or np.any(self.I_F != self.I_F_pre) or np.all(self.I_F == 0):
            # Solve the quadratic programming problem
            if self.metric == 'G':
                # print(self.A_Theta_KKT.shape, len(self.E_extended) + len(self.F_e))
                solution = self.M_Theta_KKT.solve(np.concatenate([
                    np.zeros(len(self.E_extended) * 2),
                    2 * np.pi * self.I_F - self.G_F, 
                    np.zeros(len(self.E_extended))
                ]))
                # solution = self.M_Theta_KKT.solve(np.concatenate([
                #     np.zeros(len(self.E_extended)),
                #     2 * np.pi * self.I_F - self.G_F
                # ]))
            elif self.metric == 'identity':
                solution = self.M_Theta_KKT_identity_metric.solve(np.concatenate([
                    np.zeros(len(self.E_extended)), 
                    2 * np.pi * self.I_F - self.G_F
                ]))
            elif self.metric == 'punishing_comb':
                solution = self.M_Theta_KKT_punishing_comb.solve(np.concatenate([
                    np.zeros(len(self.E_extended)), 
                    2 * np.pi * self.I_F - self.G_F
                ]))
            else:
                raise ValueError('The metric is not recognised.')
            
            Theta = solution[:len(self.E_extended)]
            
            self.Theta = Theta.copy()
            
            Theta[len(self.E_twin):] += self.pair_rotations
            
            self.A_Corner_arg = lil_matrix((len(self.E_extended) + 1, len(self.V_extended)), dtype=complex)
            self.A_Corner_arg[np.arange(len(self.E_extended)), self.E_extended[:, 0]] = np.exp(1j * Theta)
            self.A_Corner_arg[np.arange(len(self.E_extended)), self.E_extended[:, 1]] = -1
            self.A_Corner_arg[-1, np.where(self.mask_removed_v)[0][0]] = 1
            
    def reconstruct_Corner_arg(self, z_init=1000):
        if self.metric_pre != self.metric or np.any(self.I_F != self.I_F_pre) or np.all(self.I_F == 0):
            b_Corner_arg = np.zeros(len(self.E_extended) + 1, dtype=complex)
            b_Corner_arg[-1] = z_init
            
            self.Corner_arg = np.angle(lsqr(self.A_Corner_arg, b_Corner_arg, x0=self.Corner_arg_pre)[0])
            # self.Corner_arg = np.angle(self.lu_Corner_scale_augmented.solve(
            #     np.concatenate([b_Corner_arg, np.zeros(len(self.V_extended), dtype=complex)])
            # )[len(self.E_extended) + 1:])
            
            self.Corner_arg_pre = np.exp(1j * self.Corner_arg)
            
            self.metric_pre = self.metric
            
            self.I_F_pre = self.I_F.copy()
    
    def reconstruct_Corner_scale(self):
        A_Corner_scale = self.A_Corner_scale.copy()
        rank_U = 0
        
        self.V_singular_e = []
        self.Vec_singular_e = []
        
        x0 = np.ones(len(self.V_extended))
        
        def angle_difference(z1, z2):
            return np.arccos(
                np.real(z1 * np.conj(z2)) / (np.abs(z1) * np.abs(z2))
            )
                
        def scale_func(a, z):
            return np.abs(
                a * z + (1 - a) * np.conj(z)
            )
            
        def compute_uv(delta_A):
            """
            Compute U and V such that delta_A = U @ V.T
            """
            # Extract nonzero entries of delta_A
            row, col = delta_A.nonzero()
            data = delta_A.data

            # Create U and V based on the structure of delta_A
            U = np.zeros((delta_A.shape[0], len(data)))
            V = np.zeros((delta_A.shape[1], len(data)))

            for idx, (r, c, d) in enumerate(zip(row, col, data)):
                U[r, idx] = d
                V[c, idx] = 1  # Selector for the corresponding column

            return U, V
        
        # Face singularity
        for singularity, f in tqdm(zip(self.singularities[self.F_extended_singular < len(self.F_f)], 
                                       self.F_extended_singular[self.F_extended_singular < len(self.F_f)]),
                                desc='Processing face singularities', 
                                leave=False):
            # # Vertex singularity  
            v0, v1, v2 = self.F_f[f]
            edge01 = np.where(
                np.all(np.isin(self.E_twin, [v0,v1]), axis=1)
            )[0][0]
            edge12 = np.where(
                np.all(np.isin(self.E_twin, [v1,v2]), axis=1)
            )[0][0]
            edge20 = np.where(
                np.all(np.isin(self.E_twin, [v2,v0]), axis=1)
            )[0][0]
            
            z0 = complex_projection(
                self.B1[f][None, :],
                self.B2[f][None, :],
                self.normals[f][None, :],
                (self.V_extended[v0] - singularity)[None, :]
            )[0]
            
            z0, z1, z2 = complex_projection(
                self.B1[f][None, :],
                self.B2[f][None, :],
                self.normals[f][None, :],
                self.V_extended[self.F_f[f]] - singularity
            )[0]
            
            z1_relative = np.abs(z1) * np.exp(1j * (np.angle(z1) - np.angle(z0)))
            z2_relative = np.abs(z2) * np.exp(1j * (np.angle(z2) - np.angle(z0)))

            theta_01 = self.Corner_arg[v1] - self.Corner_arg[v0]
            theta_02 = self.Corner_arg[v2] - self.Corner_arg[v0]
            
            A = np.array([
                [1, np.tan(theta_01)],
                [1, np.tan(theta_02)]
            ])
            b = np.array([
                z1_relative.real * np.tan(theta_01) / (2 * z1_relative.imag) + 1/2, 
                z2_relative.real * np.tan(theta_02) / (2 * z2_relative.imag) + 1/2
            ])
            
            a_re, a_im = np.linalg.solve(A, b)
            a = a_re + 1j * a_im
            
            x0[v0] = np.linalg.norm(self.V_extended[v0] - singularity)
            x0[v1] = scale_func(a, z1_relative)
            x0[v2] = scale_func(a, z2_relative)
            
            for edge in [edge01, edge12, edge20]:
                v_start, v_end = self.E_twin[edge]
                if x0[v_start] > x0[v_end]:
                    A_Corner_scale[edge, v_end] = -x0[v_start]/x0[v_end]
                else:
                    A_Corner_scale[edge, v_start] = x0[v_end]/x0[v_start]
                    
            rank_U += 3

        # Edge singularity
        for singularity, f in tqdm(zip(self.singularities[self.F_extended_singular >= len(self.F_f)], 
                                       self.F_extended_singular[self.F_extended_singular >= len(self.F_f)]),
                                desc='Processing edge singularities', 
                                leave=False):  
            if f >= len(self.F_f) + len(self.F_e):
                continue
            
            f_e = self.F_e[f - len(self.F_f)]
            edge0 = np.where(
                np.sum(np.isin(self.E_twin, f_e[0:2]), axis=1) == 2
            )[0]
            edge1 = np.where(
                np.sum(np.isin(self.E_twin, f_e[2:4]), axis=1) == 2
            )[0]
            face0 = np.where(
                np.sum(np.isin(self.F_f, f_e[0:2]), axis=1) == 2
            )[0]
            face1 = np.where(
                np.sum(np.isin(self.F_f, f_e[2:4]), axis=1) == 2
            )[0]
            pair_rotation01 = self.pair_rotations[
                np.all(np.isin(self.E_comb, f_e[1:3]), axis=1)
            ]
            
            if np.any(np.isin(self.F_extended_singular[self.F_extended_singular < len(self.F_f)], [face0, face1])):
                pass
                
            else:
                # barycentric of singularity on the edge
                vec0 = self.V_extended[self.E_twin[edge0, 0]] - singularity
                vec1 = self.V_extended[self.E_twin[edge0, 1]] - singularity
                alpha = np.linalg.norm(vec1) / (np.linalg.norm(vec0) + np.linalg.norm(vec1))
                
                arg_l = [self.Corner_arg[self.E_twin[edge0, 0]], self.Corner_arg[self.E_twin[edge1, 0]] - pair_rotation01 + np.pi]
                arg_r = [self.Corner_arg[self.E_twin[edge0, 1]], self.Corner_arg[self.E_twin[edge1, 1]] - pair_rotation01 + np.pi]
                
                arg_l_used = arg_l[np.argmin([angle_difference(np.exp(1j * arg), np.exp(1j * arg_r[0])) for arg in arg_l])] % (2 * np.pi)
                arg_r_used = arg_r[np.argmin([angle_difference(np.exp(1j * arg), np.exp(1j * arg_l[0])) for arg in arg_r])] % (2 * np.pi)
                
                if abs(arg_l_used - arg_r_used) > np.pi:
                    if arg_l_used > arg_r_used:
                        arg_l_used -= 2 * np.pi
                    else:
                        arg_l_used += 2 * np.pi
                
                arg_singular0 = (arg_r_used + arg_l_used)/2
                arg_singular1 = arg_singular0 + pair_rotation01 + np.pi
                
                if alpha > 1/2:
                    A_Corner_scale[edge0, self.E_twin[edge0, 0]] = (alpha/(1-alpha)) * \
                        (np.tan(arg_singular0) * np.cos(self.Corner_arg[self.E_twin[edge0, 0]]) - np.sin(self.Corner_arg[self.E_twin[edge0, 0]])) /\
                            (np.sin(self.Corner_arg[self.E_twin[edge0, 1]]) - np.tan(arg_singular0) * np.cos(self.Corner_arg[self.E_twin[edge0, 1]]))
                    A_Corner_scale[edge1, self.E_twin[edge1, 1]] = -(alpha/(1-alpha)) * \
                        (np.tan(arg_singular1) * np.cos(self.Corner_arg[self.E_twin[edge1, 1]]) - np.sin(self.Corner_arg[self.E_twin[edge1, 1]])) /\
                            (np.sin(self.Corner_arg[self.E_twin[edge1, 0]]) - np.tan(arg_singular1) * np.cos(self.Corner_arg[self.E_twin[edge1, 0]]))
                            
                else:
                    A_Corner_scale[edge0, self.E_twin[edge0, 1]] = -((1-alpha)/alpha) * \
                        (np.tan(arg_singular0) * np.cos(self.Corner_arg[self.E_twin[edge0, 1]]) - np.sin(self.Corner_arg[self.E_twin[edge0, 1]])) /\
                            (np.sin(self.Corner_arg[self.E_twin[edge0, 0]]) - np.tan(arg_singular0) * np.cos(self.Corner_arg[self.E_twin[edge0, 0]]))
                    A_Corner_scale[edge1, self.E_twin[edge1, 0]] = ((1-alpha)/alpha) * \
                        (np.tan(arg_singular1) * np.cos(self.Corner_arg[self.E_twin[edge1, 0]]) - np.sin(self.Corner_arg[self.E_twin[edge1, 0]])) /\
                            (np.sin(self.Corner_arg[self.E_twin[edge1, 1]]) - np.tan(arg_singular1) * np.cos(self.Corner_arg[self.E_twin[edge1, 1]]))
                                                           
                self.V_singular_e += [singularity, singularity]
                z0 = np.exp(1j * arg_singular0)
                z1 = np.exp(1j * arg_singular1)
                vec0 = np.real(z0) * self.B1[face0] + np.imag(z0) * self.B2[face0]
                vec1 = np.real(z1) * self.B1[face1] + np.imag(z1) * self.B2[face1]
                self.Vec_singular_e += [vec0, vec1]
                
                rank_U += 2
                
        if len(self.F_f_singular) > 0:
            A_Corner_scale[-1, np.where(self.mask_removed_v)[0][0]] = 1
            if np.where(self.mask_removed_v)[0][0] != 0:
                rank_U += 1
            
            delta_A = A_Corner_scale - self.A_Corner_scale
            U, V = compute_uv(delta_A)
            
            # Compute A_inv_b and U_inv
            U_inv = np.column_stack(
                [self.lu_Corner_scale_augmented.solve(
                    np.concatenate([U[:, i], np.zeros(len(self.V_extended))])
                )[len(self.E_extended) + 1:] for i in range(U.shape[1])]
            )
            
            # Compute correction term
            correction = np.linalg.inv(np.eye(rank_U) + V.T @ U_inv) @ (V.T @ self.Corner_scale_harmonic)
            
            # Compute updated solution
            self.Corner_scale = self.Corner_scale_harmonic - U_inv @ correction
        
            # b_Corner_scale = np.zeros(len(self.E_extended) + 1)
            # b_Corner_scale[-1] = 1
            
            # self.Corner_scale = lsqr(A_Corner_scale, b_Corner_scale, x0=x0)[0]
        
        else:
            self.Corner_scale = self.Corner_scale_harmonic.copy()
        self.Corner_scale_pre = self.Corner_scale.copy()
        
    def reconstruct_linear_from_Corners(self):
        '''
            Reconstruct the coefficients of the linear field from the Corner values.
            Input:
                self.Us: (num_F) complex array of Corner values
                singularities_f: (num_singularities, 3) array of singularities in the faces
        '''
        self.U = self.Corner_scale * np.exp(1j * self.Corner_arg)
        b_Coeff = np.zeros(len(self.F_f) * 6, dtype=float)

        for i, f in tqdm(enumerate(self.F_f),
                         desc=f'Reconstructing linear field coefficients',
                         total=len(self.F_f),
                         leave=False):
            b_Coeff[6*i:6*(i+1)] = np.array([
                self.U[f[0]].real, self.U[f[0]].imag, self.U[f[1]].real, self.U[f[1]].imag, self.U[f[2]].real, self.U[f[2]].imag
            ])

        solution = self.M_Coeff.solve(b_Coeff)
        
        self.Coeff = np.zeros((len(self.F_f), 3), dtype=complex)
        self.Coeff[:, 0] = solution[0::6] + 1j * solution[1::6]
        self.Coeff[:, 1] = solution[2::6] + 1j * solution[3::6]
        self.Coeff[:, 2] = solution[4::6] + 1j * solution[5::6]

    def sample_prep(self, num_samples=7, margin = 0.01):
        
        self.points_sample = []
        self.F_sample = []
        
        for i in tqdm(range(num_samples), desc='Sampling points for field processing', 
                      total=num_samples, leave=False):
            if num_samples == 1:
                self.points_sample += np.mean(self.V[self.F], axis=1).tolist()
                self.F_sample += np.arange(len(self.F_f)).tolist()
                
            else:
                for j in range(num_samples - i):
                    # Barycentric coordinates
                    u = margin + (i / (num_samples-1)) * (1 - 3 * margin)
                    v = margin + (j / (num_samples-1)) * (1 - 3 * margin)
                    w = 1 - u - v
                    
                    # Interpolate to get the 3D point in the face
                    self.points_sample += (
                        u * self.V[self.F[:, 0]] + \
                            v * self.V[self.F[:, 1]] + \
                                w * self.V[self.F[:, 2]]
                    ).tolist()
                    self.F_sample += np.arange(len(self.F_f)).tolist()
                        
        self.points_sample = np.array(self.points_sample)
        self.F_sample = np.array(self.F_sample)
        self.Z_sample = complex_projection(
            self.B1[self.F_sample], self.B2[self.F_sample], self.normals[self.F_sample],
            self.points_sample - self.V_extended[self.F_f[self.F_sample, 0]], 
            diagonal=True
        )
        
    def sample_field(self):
        vectors_complex = self.Coeff[self.F_sample, 0] * self.Z_sample +\
            self.Coeff[self.F_sample, 1] * np.conjugate(self.Z_sample) +\
                self.Coeff[self.F_sample, 2]

        self.vectors = vectors_complex.real[:, None] * self.B1[self.F_sample] +\
            vectors_complex.imag[:, None] * self.B2[self.F_sample]
            
    def static_field(self, singularities, indices):
        self.construct_extended_mesh()
        self.sample_prep()
        self.process_singularities(singularities, indices)
        self.compute_thetas()
        self.reconstruct_Corner_arg()
        # self.reconstruct_Corner_scale()
        self.Corner_scale = 1
        coeffs = self.reconstruct_linear_from_Corners()
        vectors = self.sample_field(coeffs)
        
        ps.init()
        ps_mesh = ps.register_surface_mesh("Input Mesh", self.V, self.F, color=(0.95, 0.98, 1))

        ps_field = ps.register_point_cloud("Field_sample", self.points_sample, enabled=True, radius=0)
        ps_field.add_vector_quantity('Field', vectors, enabled=True, color=(0.03, 0.33, 0.77))
        
        if np.any(np.array(indices) != 0):
            singularity_marker = ps.register_point_cloud("singularity marker", 
                                    np.array([singularities[i] for i in range(len(singularities)) if indices[i] != 0]), 
                                    enabled=True, 
                                    radius=0.0015)
        
        ps.show()
    
    def dynamic_field(self, F_singular, indices):
        self.construct_extended_mesh()
        self.sample_prep()
        singularities = np.mean(self.V[self.F[F_singular]], axis=1)
        
        self.process_singularities(singularities, indices)
        
        list_reconstruction = [
            self.compute_thetas, 
            self.reconstruct_Corner_arg,
            self.reconstruct_Corner_scale,
            self.reconstruct_linear_from_Corners,
            self.sample_field
        ]
        
        for step in tqdm(list_reconstruction, 
                        desc='Reconstructing the field', 
                        leave=False):
            step()
        
        ps.init()
        ps_mesh = ps.register_surface_mesh("Input Mesh", self.V, self.F, color=(0.95, 0.98, 1))

        ps_field = ps.register_point_cloud("Field_sample", self.points_sample, enabled=True, radius=0)
        ps_field.add_vector_quantity('Field', self.vectors/np.linalg.norm(self.vectors, axis=1)[:, None],
                                     enabled=True, color=(0.03, 0.33, 0.77))
        
        if np.any(np.array(indices) != 0):
            singularity_marker = ps.register_point_cloud("singularity marker", 
                                    np.array([singularities[i] for i in range(len(singularities)) if indices[i] != 0]), 
                                    enabled=True, 
                                    radius=0.0015)
            
            reconstruct_singularities = []
            corners = []
            corner_vectors = []
            
            for f in self.F_f_singular:
                coeff = self.Coeff[f]
                singularity = (coeff[1] * np.conjugate(coeff[2]) - coeff[2] * np.conjugate(coeff[0])) /\
                    (np.abs(coeff[0])**2 - np.abs(coeff[1])**2)
                    
                corners += self.V_extended[self.F_f[f]].tolist()
                corner_vector = []
                for i in range(3):
                    corner_vector.append(
                        self.U[self.F_f[f, i]].real * self.B1[f] + self.U[self.F_f[f, i]].imag * self.B2[f]
                    )
                corner_vectors += corner_vector
                
                if f in self.F_extended_singular:
                    reconstruct_singularities.append(
                        singularity.real * self.B1[f] + singularity.imag * self.B2[f] + self.V_extended[self.F_f[f, 0]]
                    )
                
            reconstruct_singularities = np.array(reconstruct_singularities)
            singularity_marker_reconstruct = ps.register_point_cloud("singularity marker reconstruct", 
                                    reconstruct_singularities, 
                                    enabled=False, 
                                    radius=0.0030)
            singular_corner_reconstruct = ps.register_point_cloud("singular corner reconstruct",
                                    np.array(corners),
                                    enabled=False, 
                                    radius=0)
            singular_corner_reconstruct.add_vector_quantity('Corner', 
                                                            np.array(corner_vectors),
                                                            enabled=False)
            
        self.metric = 'G'
        self.bary_coors = np.ones((len(F_singular), 2)) * (1/3)
        
        def callback():
            
            changed = False
            
            if psim.RadioButton("Discrete integral metric", self.metric == 'G'):
                self.metric = 'G'
                changed = True
            if psim.RadioButton("Identity metric", self.metric == 'identity'):
                self.metric = 'identity'
                changed = True
            if psim.RadioButton("Punishing combinatorial rotation", self.metric == 'punishing_comb'):
                self.metric = 'punishing_comb'
                changed = True
            
            for i, f in enumerate(F_singular):
                changed_u, self.bary_coors[i][0] = psim.SliderFloat(f'Bary u for f {f}', self.bary_coors[i][0], v_min=0, v_max=1)
                changed_v, self.bary_coors[i][1] = psim.SliderFloat(f'Bary v for f {f}', self.bary_coors[i][1], v_min=0, v_max=1-self.bary_coors[i][0])
                self.bary_coors[i][1] = min(self.bary_coors[i][1], 1-self.bary_coors[i][0])
                
                if changed_u or changed_v:
                    changed = True
                    
            if changed:
                singularities = self.V[self.F[F_singular, 0]] * self.bary_coors[:, 0][:, None] +\
                    self.V[self.F[F_singular, 1]] * self.bary_coors[:, 1][:, None] +\
                        self.V[self.F[F_singular, 2]] * (1 - self.bary_coors[:, 0] - self.bary_coors[:, 1])[:, None]
                
                t0 = time.time()
                self.process_singularities(singularities, indices)
                print(f'Time taken for processing singularities: {round(time.time() - t0, 3)}')
                    
                for step in list_reconstruction:
                    t0 = time.time()
                    step()
                    print(f'Time taken for step {step.__name__}: {round(time.time() - t0, 3)}')
                
                print('')
                
                # self.vectors[np.any(self.vectors != 0, axis=1)] /= np.linalg.norm(self.vectors[np.any(self.vectors != 0, axis=1)], axis=1)[:, None]
                
                ps_field.remove_all_quantities()
                ps_field.add_vector_quantity('Field', self.vectors,
                                             enabled=True, color=(0.03, 0.33, 0.77))
                
                if np.any(np.array(indices) != 0):
                    singularity_marker.update_point_positions(singularities)
                #     reconstruct_singularities = []
                #     corners = []
                #     corner_vectors = []
                    
                #     for i, f in enumerate(self.F_f_singular):
                #         coeff = self.Coeff[f]
                #         singularity = (coeff[1] * np.conjugate(coeff[2]) - coeff[2] * np.conjugate(coeff[0])) /\
                #             (np.abs(coeff[0])**2 - np.abs(coeff[1])**2)
                    
                #         corners += self.V_extended[self.F_f[f]].tolist()
                #         corner_vector = []
                #         for j in range(3):
                #             corner_vector.append(
                #                 self.U[self.F_f[f, j]].real * self.B1[f] + self.U[self.F_f[f, j]].imag * self.B2[f]
                #             )
                #         corner_vectors += corner_vector
                        
                #         if f in self.F_extended_singular:
                #             reconstruct_singularities.append(singularity.real * self.B1[f] + singularity.imag * self.B2[f] + self.V_extended[self.F_f[f, 0]])
                    
                    # singularity_marker_reconstruct = ps.register_point_cloud("singularity marker reconstruct", 
                    #                         np.array(reconstruct_singularities),
                    #                         enabled=True, 
                    #                         radius=0.0030)

                    # singular_corner_reconstruct = ps.register_point_cloud("singular corner reconstruct",
                    #                         np.array(corners),
                    #                         enabled=True, 
                    #                         radius=0)
                    # singular_corner_reconstruct.add_vector_quantity('Corner', 
                    #                                                 np.array(corner_vectors),
                    #                                                 enabled=True)
                    # singular_corner_reconstruct.add_vector_quantity('Corner', np.array(corner_vectors), enabled=True)
                    
                    # if len(self.V_singular_e) > 0:
                    #     Vec_singular_edge = ps.register_point_cloud("Edge singularities", 
                    #                                 np.array(self.V_singular_e).squeeze(),
                    #                                 enabled=True, 
                    #                                 radius=0)
                    #     Vec_singular_edge.add_vector_quantity('Edge singularities', 
                    #                                         np.array(self.Vec_singular_e).squeeze(),
                    #                                         enabled=True)

        ps.set_user_callback(callback)
        ps.show()    
            
    def dynamic_field_along_curve(self, F_singular, indices, curve_len=3):
            self.construct_extended_mesh()
            self.sample_prep()
            
            curve_points = np.zeros((len(F_singular), curve_len * 2 + 1, 3))
            curve_points[:, 0] = self.V[self.F[F_singular, 0]]
            V_last_used = self.F[F_singular, 0]
            V_second_last_used = self.F[F_singular, 0]
            
            for i in range(curve_len):
                for j in range(len(F_singular)):
                    v_last_used = V_last_used[j]
                    v_second_last_used = V_second_last_used[j]
                    
                    F_neighbor = self.F[np.any(np.isin(self.F, v_last_used), axis=1)]
                    mid_points = np.mean(self.V[F_neighbor], axis=1)
                    f_next = F_neighbor[np.argmax(np.linalg.norm(mid_points - self.V[v_second_last_used], axis=1))]
                    
                    e_next = np.setdiff1d(f_next, v_last_used)
                    bary_coor = np.random.rand(1)
                    
                    curve_points[j, 2*i + 1] = self.V[e_next[0]] * bary_coor + self.V[e_next[1]] * (1 - bary_coor)
                    
                    v_next = np.setdiff1d(
                        self.F[np.sum(np.isin(self.F, e_next), axis=1) == 2].flatten(),
                        [v_last_used, e_next[0], e_next[1]]
                    )
                        
                    curve_points[j, 2*i + 2] = self.V[v_next]
                    
                    V_second_last_used[j] = v_last_used
                    V_last_used[j] = v_next
            
            def get_singularities(curve_progresses, curve_points):
                singularities = np.zeros((len(F_singular), 3))
                for i in range(len(F_singular)):
                    bar = curve_progresses[i] * (len(curve_points[i]) - 1)
                    
                    singularities[i] = curve_points[i, int(np.floor(bar))] * (1 - bar % 1) +\
                        curve_points[i, int(np.ceil(bar))] * (bar % 1)
                
                return singularities
            
            global curve_progresses
            
            curve_progresses = np.zeros(len(F_singular))
            singularities = get_singularities(curve_progresses, curve_points)
            
            self.process_singularities(singularities, indices)
            
            list_reconstruction = [
                self.compute_thetas, 
                self.reconstruct_Corner_arg,
                self.reconstruct_Corner_scale,
                self.reconstruct_linear_from_Corners,
                self.sample_field
            ]
            
            for step in tqdm(list_reconstruction, 
                            desc='Reconstructing the field', 
                            leave=False):
                step()
            
            ps.init()
            ps_mesh = ps.register_surface_mesh("Input Mesh", self.V, self.F, color=(0.95, 0.98, 1))

            ps_field = ps.register_point_cloud("Field_sample", self.points_sample, enabled=True, radius=0)
            ps_field.add_vector_quantity('Field', self.vectors/np.linalg.norm(self.vectors, axis=1)[:, None],
                                         enabled=True, color=(0.03, 0.33, 0.77))
            
            if np.any(np.array(indices) != 0):
                singularity_marker = ps.register_point_cloud("singularity marker", 
                                        np.array([singularities[i] for i in range(len(singularities)) if indices[i] != 0]), 
                                        enabled=True, 
                                        radius=0.0015)
            
            def callback():
                
                # bary_coors_new = bary_coors.copy()
                changed = False
                
                for i, f in enumerate(F_singular):
                    changed_any, curve_progresses[i] = psim.SliderFloat(f'Curve progress for singularirty {i}', curve_progresses[i], v_min=0, v_max=1)
                    
                    if changed_any:
                        changed = True
                
                if changed:
                    singularities = get_singularities(curve_progresses, curve_points)
                    
                    t0 = time.time()
                    self.process_singularities(singularities, indices)
                    print(f'Time taken for processing singularities: {round(time.time() - t0, 3)}')
                        
                    for step in list_reconstruction:
                        t0 = time.time()
                        step()
                        print(f'Time taken for step {step.__name__}: {round(time.time() - t0, 3)}')
                    
                    print('')
                    
                    ps_field.remove_all_quantities()
                    ps_field.add_vector_quantity('Field', self.vectors/np.linalg.norm(self.vectors, axis=1)[:, None],
                                                 enabled=True, color=(0.03, 0.33, 0.77))
                    
                    if np.any(np.array(indices) != 0):
                        singularity_marker.update_point_positions(singularities)

            ps.set_user_callback(callback)
            ps.show() 
    