import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix, eye, bmat, diags, vstack, hstack, lil_matrix, coo_matrix
from Functions.Auxiliary import (accumarray, find_indices, is_in_face, compute_planes, 
                                 complex_projection, obtain_E, compute_V_boundary, 
                                 compute_unfolded_vertex, compute_barycentric_coordinates, 
                                 compute_angle_defect)
from scipy.sparse.linalg import lsqr, spsolve, spilu, LinearOperator, eigsh, gmres, splu, cg
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
            
        self.V_extended = np.array(self.V_extended)
    
    def get_F_e(self):
        # Construct edge-faces
        for i, e in tqdm(enumerate(self.E), 
                        desc='Constructing edge-faces',
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
                
                # Corresponding comb properties
                self.E_comb += [
                    [edges[0][1], edges[1][0]],
                    [edges[1][1], edges[0][0]]
                ]
                
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

    def get_F_v_E_comb(self):
        # Mapping for efficient construction of d1
        self.F_v_map_E_comb = {}
        
        # Construct vertex-faces and combinatorial edges
        for v, indices_extended in tqdm(self.V_map.items(), 
                                        desc='Constructing vertex-faces and combinatorial edges',
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
                self.F_v_map_E_comb[len(self.F_v)] = np.arange(len(indices_sorted)) + len(self.E_comb)
                    
                # Add the vertex-face
                self.F_v.append(indices_sorted)

            else:
                raise ValueError(f'Wrong number of extended vertices found for {v}: {indices_extended}. \n Boundary vertices are not supported.')
        
    def get_preconds(self):
        # Theta computation
        self.pair_rotations = np.array(self.pair_rotations).squeeze()
        self.G_F = np.concatenate([
            np.zeros(len(self.F_f) + len(self.F_e)),
            self.G_V
        ])
        
        Q = eye(len(self.E_extended), format='lil') * 10
        Q[np.arange(len(self.E_twin)), np.arange(len(self.E_twin))] = 1
        
        self.A_Theta_KKT = bmat([
            [Q, self.d1.T],
            [self.d1, np.zeros((self.d1.shape[0], self.d1.shape[0]))]
        ], format='csr')
        
        self.lu_Theta_KKT = splu((self.A_Theta_KKT + 1e-8 * eye(self.A_Theta_KKT.shape[0])).tocsc())
        
        # Corner scale computation
        self.A_Corner_scale = lil_matrix((len(self.E_extended), len(self.V_extended)))
        self.A_Corner_scale[np.arange(len(self.E_extended)), self.E_extended[:, 0]] = -1
        self.A_Corner_scale[np.arange(len(self.E_extended)), self.E_extended[:, 1]] = 1
        self.A_Corner_scale.tocsr()
        
        ATA_Corner_scale = (self.A_Corner_scale.T @ self.A_Corner_scale).tocsc()
        
        self.MTM_Corner_scale = splu(ATA_Corner_scale + 1e-8 * eye(ATA_Corner_scale.shape[0], format = 'csc'))
        
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
        
        self.d1 = lil_matrix((len(self.F) + len(self.E) + len(self.V), 4 * len(self.E)), dtype=int)
        self.pair_rotations = []
        self.V_map = {v:[] for v in range(self.V.shape[0])}
        
        for step in tqdm([self.get_V_F_f_E_twin, 
                          self.get_F_e, 
                          self.get_F_v_E_comb, 
                          self.get_preconds], 
                         desc='Constructing the extended mesh',
                         leave=True):
            step()
        
        self.d1.tocsr()
        
        self.mask_removed_e_pre = None
        self.Theta_pre = np.zeros(len(self.E_extended))
        self.A_Corner_arg_pre = None
        self.Corner_arg_soln_pre = None
        self.Corner_scale_soln_pre = np.zeros(len(self.V_extended))
        
    def process_singularities(self, singularities=None, indices=None):
        if len(singularities) != len(indices):
            raise ValueError('The number of singularities and the number of indices do not match.')
        if np.sum(indices) != 2 - 2*self.genus:
            raise ValueError('The sum of indices is not equal to 2 - 2g.')
        
        self.singularities = singularities[np.array(indices) != 0]
        self.indices = indices[np.array(indices) != 0]
        self.I_F = np.zeros(len(self.F_f) + len(self.F_e) + len(self.F_v))
        self.F_extended_singular = np.zeros(len(self.singularities), dtype=int)
        self.V_singular = []
        self.any_singular_e = False
        self.e_phase_change = False
        self.mask_removed_v = np.ones(len(self.V_extended), dtype=bool)
        indices_dir = np.ones(len(self.singularities), dtype=int)
        self.Theta = np.zeros(len(self.E_extended))
        self.mask_removed_e = np.ones(len(self.E_extended), dtype=bool)
        self.mask_removed_f = np.ones(len(self.F_f) + len(self.F_e) + len(self.F_v), dtype=bool)
        
        def angle_difference(v1, v2):
            return np.abs(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
        
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
                self.F_extended_singular[i] = len(self.F_f) + len(self.F_e) + in_F_v[0]
                self.V_singular.append(in_F_v[0])
                
                for v in self.V_map[in_F_v[0]]:
                    f = self.F_f[np.where(
                        np.any(np.isin(self.F_f, v), axis=1)
                    )[0][0]]
                    
                    self.mask_removed_v[f] = False
                    
            # If the singularity is in an edge, it gives the thetas for the incident faces
            elif len(in_F_e) == 1:
                self.F_extended_singular[i] = len(self.F_f) + in_F_e[0]
                indices_dir[i] = 0
                
                e = self.F_e[in_F_e[0]]
                
                e_twin = np.where(
                    np.sum(np.isin(self.E_twin, e), axis=1) == 2
                )[0]
                e_comb = np.where(
                    np.sum(np.isin(self.E_comb, e), axis=1) == 2
                )[0] + len(self.E_twin)
                
                self.Theta[e_twin] = -np.pi * self.indices[i]
                
                f = self.F_f[
                    np.sum(np.isin(self.F_f, self.F_e[in_F_e[0]]), axis=1) == 2
                ].squeeze()
                    
                self.mask_removed_e[e_twin] = False
                self.mask_removed_e[e_comb] = False
                self.mask_removed_f[in_F_e[0]] = False
                self.mask_removed_v[f] = False
                
                self.any_singular_e = True
                
            # If the singularity is in a face, it gives thetas for the face
            elif in_F_f is not False:
                self.F_extended_singular[i] = in_F_f
                indices_dir[i] = 0
                
                e_twin = np.where(
                    np.sum(np.isin(self.E_twin, self.F_f[in_F_f]), axis=1) == 2
                )[0]
                
                for e in e_twin:
                    self.Theta[e] = angle_difference(self.V_extended[self.E_twin[e][0]] - singularity,
                                                     self.V_extended[self.E_twin[e][1]] - singularity) *\
                                    self.indices[i]
                
                self.mask_removed_e[e_twin] = False
                self.mask_removed_f[in_F_f] = False
                self.mask_removed_v[self.F_f[in_F_f]] = False
                    
            else:
                raise ValueError(f'The singularity {singularity} is not in any face, edge or vertex.')
        
        self.I_F[self.F_extended_singular] = self.indices * indices_dir
        self.mask_KKT = np.concatenate([
            self.mask_removed_e, 
            self.mask_removed_f
        ])
        
        t0 = time.time()
        if self.mask_removed_e_pre is None:
            self.A_Corner_scale_trunc = self.A_Corner_scale[:, self.mask_removed_v]
            self.A_Theta_KKT_trunc = self.A_Theta_KKT[:, self.mask_KKT][self.mask_KKT]
            print('Time taken for constructing the truncated matrices: ', time.time() - t0)
        elif np.any(self.mask_removed_e != self.mask_removed_e_pre):
            self.A_Corner_scale_trunc = self.A_Corner_scale[:, self.mask_removed_v]
            self.A_Theta_KKT_trunc = self.A_Theta_KKT[:, self.mask_KKT][self.mask_KKT]
            self.e_phase_change = True
            print('Time taken for constructing the truncated matrices: ', time.time() - t0)
        
        self.mask_removed_e_pre = self.mask_removed_e.copy()
        
    def solve_truncated(self, A_trunc, b, M, mask_col, x0=None):
        def preconditioner_trunc(x):
            # Expand x to the full size of A with zeros in non-selected indices
            full_x = np.zeros(len(mask_col))
            full_x[mask_col] = x  # Use only selected rows or columns

            # Apply the full preconditioner
            full_result = M.solve(full_x)

            # Return only the selected rows of the result
            return full_result[mask_col]
            
        M_trunc = LinearOperator(A_trunc.shape, matvec=preconditioner_trunc)

        # Return only the selected rows of the result
        solution, _ = cg(A_trunc, b, x0=x0, M=M_trunc, maxiter=100)
        
        return solution
        
    def compute_thetas(self):
        '''
            Compute the set of thetas for each face singularity 
            by constrained optimisation using the KKT conditions.
        '''
        
        with tqdm(total=1, desc='Computing Theta', leave=False):
            
            b_Theta_KKT = np.concatenate([
                np.zeros(np.sum(self.mask_removed_e)), 
                (2 * np.pi * self.I_F - self.G_F - self.d1 @ self.Theta)[self.mask_removed_f]
            ])
            x0 = np.concatenate([
                self.Theta_pre[self.mask_removed_e],
                np.ones(np.sum(self.mask_removed_f))
            ])
            
            if self.any_singular_e and self.e_phase_change:
                solution = lsqr(self.A_Theta_KKT_trunc, b_Theta_KKT, x0=x0)[0]
            elif self.any_singular_e and not self.e_phase_change:
                solution = self.Theta_pre[self.mask_removed_e].copy()
            else:
                solution = self.solve_truncated(self.A_Theta_KKT_trunc, b_Theta_KKT, self.lu_Theta_KKT, self.mask_KKT, x0=x0)
        
        Theta = self.Theta.copy()
        Theta[self.mask_removed_e] = solution[:np.sum(self.mask_removed_e)]
        self.Theta = Theta.copy()
        self.Theta_pre = Theta.copy()
            
    def reconstruct_Corner_arg(self, z_init=1):
        with tqdm(total=1, desc='Reconstructing corner arguments', leave=False):
            Theta = self.Theta.copy()
            Theta[len(self.E_twin):] += self.pair_rotations
            
            A_Corner_arg = lil_matrix((len(self.E_extended) + 1, len(self.V_extended)), dtype=complex)
            A_Corner_arg[np.arange(len(self.E_extended)), self.E_extended[:, 0]] = np.exp(1j * Theta)
            A_Corner_arg[np.arange(len(self.E_extended)), self.E_extended[:, 1]] = -1
            A_Corner_arg[-1, 0] = 1
            
            b_Corner_arg = np.zeros(len(self.E_extended) + 1, dtype=complex)
            b_Corner_arg[-1] = z_init/np.abs(z_init)
            
            # First time computing the corner arguments
            if self.A_Corner_arg_pre is None:
                solution = lsqr(A_Corner_arg, b_Corner_arg)[0]
                
                self.Corner_arg = np.angle(solution)
                
            # Reuse the previous matrix/solution
            else:
                # Perturbation matrix U
                # U = A_Corner_arg - self.A_Corner_arg_pre
                
                # If the change (U) is small, apply Sherman-Morrison-Woodbury or iterative refinement
                solution = lsqr(A_Corner_arg, b_Corner_arg, x0=self.Corner_arg_soln_pre)[0]

                # Update the solution
                self.Corner_arg = np.angle(solution)
            
            self.A_Corner_arg_pre = A_Corner_arg.copy()
            self.Corner_arg_soln_pre = solution.copy()
    
    def reconstruct_Corner_scale(self):
        self.Corner_scale = np.zeros(len(self.V_extended), dtype=float)
        v_mask = np.ones(len(self.V_extended), dtype=bool)
        difference_Corner_scale = np.zeros(len(self.E_extended))
        
        def cos_angle_difference(z1, z2):
            return np.dot(z1, z2) / (np.linalg.norm(z1) * np.linalg.norm(z2))
            
        def deal_scale(f, singularity):
            v_mask[self.F_f[f]] = False
            self.Corner_scale[self.F_f[f]] =  np.linalg.norm(self.V_extended[self.F_f[f]] - singularity, axis=1)
        
        for singularity, f_extended in tqdm(zip(self.singularities, self.F_extended_singular), 
                                desc='Processing singularities', 
                                total=len(self.singularities), 
                                leave=False):
            
            # Vertex singularity
            if f_extended >= len(self.F_f) + len(self.F_e):
                indices_extended = self.V_map[f_extended - len(self.F_f) - len(self.F_e)]
                    
                for v in indices_extended:
                    f = np.where(
                        np.any(np.isin(self.F_f, v), axis=1)
                    )[0][0]
                    
                    deal_scale(f, singularity)
                    
                    # other_v = np.setdiff1d(self.F_f[f], v)
                    
                    # other_e = np.where(
                    #     np.all(np.isin(self.E_twin, other_v), axis=1)
                    # )[0][0]
                    # other_edge = self.E_twin[other_e]
                    
                    # v_mask[other_edge] = False
                        
                    # self.Corner_scale[other_edge[0]] = np.linalg.norm(self.V_extended[other_v[0]] - singularity)
                    # self.Corner_scale[other_edge[1]] = np.linalg.norm(self.V_extended[other_v[1]] - singularity) *\
                    #                                    np.abs(cos_angle_difference(
                    #                                        self.V_extended[other_edge[1]] - singularity,
                    #                                        self.V_extended[other_edge[0]] - singularity
                    #                                    ) /\
                    #                                    np.cos(self.Theta[other_e]))
                    
                    # difference_Corner_scale[np.where(
                    #     np.all(np.isin(self.E_twin, [other_edge[0], v]), axis=1)
                    # )[0]] = self.Corner_scale[other_edge[0]]
                    # difference_Corner_scale[np.where(
                    #     np.all(np.isin(self.E_twin, [other_edge[1], v]), axis=1)
                    # )[0]] = -self.Corner_scale[other_edge[1]]
                    
            # Edge singularity
            elif f_extended >= len(self.F_f):
                for i in range(2):
                    f = np.where(
                        np.sum(np.isin(self.F_f, self.F_e[f_extended - len(self.F_f)][2*i:2*(i+1)]), axis=1) == 2
                    )[0][0]
                    
                    deal_scale(f, singularity)
            
            # Face singularity
            else:
                deal_scale(f_extended, singularity)
        
        # Directly solve the preconditioned system
        if len(self.singularities) > 0:
            ATb_Corner_scale = - self.A_Corner_scale.T @ (self.A_Corner_scale @ self.Corner_scale)
            
            # print(f'Nonzero entries in Corner_scale: {np.sort(np.abs(self.Corner_scale))[::-1][:10]}')
            # print(f'Nonzero difference in Corner_scale: {np.sum(b_Corner_scale != 0)}')
            # t0 = time.time()
            # self.Corner_scale[self.mask_removed_v] = lsqr(self.A_Corner_scale_trunc, b_Corner_scale, x0=self.Corner_scale[self.mask_removed_v])[0]
            # print(f'Time taken for solving Scale: {time.time() - t0}')
            
            self.Corner_scale[self.mask_removed_v] = self.solve_truncated(
                self.A_Corner_scale_trunc.T @ self.A_Corner_scale_trunc, 
                ATb_Corner_scale[self.mask_removed_v],
                self.MTM_Corner_scale, 
                self.mask_removed_v, 
                x0=self.Corner_scale_soln_pre[self.mask_removed_v])
            
            # t0 = time.time()
            # self.Corner_scale[self.mask_removed_v] = self.MTM_Corner_scale_trunc.solve(
            #     self.A_Corner_scale[:, self.mask_removed_v].T @ b_Corner_scale
            # )
            # print(f'Time taken for solving the preconditioned system: {time.time() - t0}')
        
        # Harmonic vertex scales
        else:
            self.Corner_scale = np.ones(len(self.V_extended))
        
        self.Corner_scale_soln_pre = self.Corner_scale.copy()

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
    
    def sample_prep(self, num_samples=3, margin = 0.15):
        
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
        ps_field.add_vector_quantity('Field', self.vectors, enabled=True, color=(0.03, 0.33, 0.77))
        
        if np.any(np.array(indices) != 0):
            singularity_marker = ps.register_point_cloud("singularity marker", 
                                    np.array([singularities[i] for i in range(len(singularities)) if indices[i] != 0]), 
                                    enabled=True, 
                                    radius=0.0015)
            
            reconstruct_singularities = []
            corners = []
            corner_vectors = []
            
            for f in F_singular:
                coeff = self.Coeff
                singularity = (coeff[1] * np.conjugate(coeff[2]) - coeff[2] * np.conjugate(coeff[0])) /\
                    (np.abs(coeff[0])**2 - np.abs(coeff[1])**2)
                    
                corners += self.V_extended[self.F_f[f]].tolist()
                corner_vector = []
                for i in range(3):
                    corner_vector.append(
                        self.U[self.F_f[f, i]].real * self.B1[f] + self.U[self.F_f[f, i]].imag * self.B2[f]
                    )
                corner_vectors += corner_vector
                
                reconstruct_singularities.append(
                    singularity.real * self.B1[f] + singularity.imag * self.B2[f] + self.V_extended[self.F_f[f, 0]]
                )
                
            reconstruct_singularities = np.array(reconstruct_singularities)
            # singularity_marker_reconstruct = ps.register_point_cloud("singularity marker reconstruct", 
            #                         reconstruct_singularities, 
            #                         enabled=True, 
            #                         radius=0.0030)
            singular_corner_reconstruct = ps.register_point_cloud("singular corner reconstruct",
                                    np.array(corners),
                                    enabled=True, 
                                    radius=0)
            singular_corner_reconstruct.add_vector_quantity('Corner', 
                                                            np.array(corner_vectors),
                                                            enabled=True)
            
        global bary_coors

        bary_coors = np.ones((len(F_singular), 2)) * (1/3)
        
        def callback():
            
            original_mask_removed_e = self.mask_removed_e.copy()
            
            # bary_coors_new = bary_coors.copy()
            changed = False
            
            for i, f in enumerate(F_singular):
                changed_u, bary_coors[i][0] = psim.SliderFloat(f'Bary u for f {f}', bary_coors[i][0], v_min=0, v_max=1)
                changed_v, bary_coors[i][1] = psim.SliderFloat(f'Bary v for f {f}', bary_coors[i][1], v_min=0, v_max=1-bary_coors[i][0])
                bary_coors[i][1] = min(bary_coors[i][1], 1-bary_coors[i][0])
                
                if changed_u or changed_v:
                    changed = True
                    
            if changed:
                singularities = self.V[self.F[F_singular, 0]] * bary_coors[:, 0][:, None] +\
                    self.V[self.F[F_singular, 1]] * bary_coors[:, 1][:, None] +\
                        self.V[self.F[F_singular, 2]] * (1 - bary_coors[:, 0] - bary_coors[:, 1])[:, None]
                
                t0 = time.time()
                self.process_singularities(singularities, indices)
                print(f'Time taken for processing singularities: {round(time.time() - t0, 3)}')
                    
                for step in list_reconstruction:
                    t0 = time.time()
                    step()
                    print(f'Time taken for step {step.__name__}: {round(time.time() - t0, 3)}')
                
                print('')
                
                ps_field.remove_all_quantities()
                ps_field.add_vector_quantity('Field', self.vectors, enabled=True, color=(0.03, 0.33, 0.77))
                
                if np.any(np.array(indices) != 0):
                    singularity_marker.update_point_positions(singularities)
                    corners = []
                    corner_vectors = []
                    
                    for i, f in enumerate(F_singular):
                        coeff = self.Coeff
                        singularity = (coeff[1] * np.conjugate(coeff[2]) - coeff[2] * np.conjugate(coeff[0])) /\
                            (np.abs(coeff[0])**2 - np.abs(coeff[1])**2)
                    
                        corners += self.V_extended[self.F_f[f]].tolist()
                        corner_vector = []
                        for j in range(3):
                            corner_vector.append(
                                self.U[self.F_f[f, j]].real * self.B1[f] + self.U[self.F_f[f, j]].imag * self.B2[f]
                            )
                        corner_vectors += corner_vector
                            
                        reconstruct_singularities[i] = singularity.real * self.B1[f] + singularity.imag * self.B2[f] + self.V_extended[self.F_f[f, 0]]
                    
                    # singularity_marker_reconstruct.update_point_positions(reconstruct_singularities)
                    
                    singular_corner_reconstruct.remove_all_quantities()
                    singular_corner_reconstruct.add_vector_quantity('Corner', 
                                                                    np.array(corner_vectors), 
                                                                    enabled=True)
                    # singular_corner_reconstruct.add_vector_quantity('Corner', np.array(corner_vectors), enabled=True)

        ps.set_user_callback(callback)
        ps.show()
            
    def dynamic_field_along_curve(self, F_singular, indices, curve_len=3):
            self.construct_extended_mesh()
            self.sample_prep()
            
            curve_points = np.zeros((len(F_singular), curve_len * 2 + 1, 3))
            curve_points[:, 0] = self.V[self.F[F_singular, 0]]
            V_last_used = self.F[F_singular, 0]
            V_used = [self.F[f][0] for f in F_singular]
            
            for i in range(curve_len):
                for j in range(len(F_singular)):
                    v_last_used = V_last_used[j]
                    V_neighbor = self.E[np.any(np.isin(self.E, v_last_used), axis=1)].flatten()
                    V_neighbor = V_neighbor[~np.isin(V_neighbor, V_used)]
                    
                    v_next = random.choice(V_neighbor)
                    bary_coor = np.random.rand(1)
                    
                    curve_points[j, 2*i + 1] = self.V[v_last_used] * (1 - bary_coor) + self.V[v_next] * bary_coor
                    V_used.append(v_next)
                    
                    V_neighbor = self.F[np.sum(np.isin(self.F, [v_last_used, v_next]), axis=1) == 2].flatten()
                    V_neighbor = V_neighbor[~np.isin(V_neighbor, V_used)]
                    
                    v_next = random.choice(V_neighbor)
                        
                    curve_points[j, 2*i + 2] = self.V[v_next]
                    
                    V_used.append(v_next)
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
            ps_field.add_vector_quantity('Field', self.vectors, enabled=True, color=(0.03, 0.33, 0.77))
            
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
                    ps_field.add_vector_quantity('Field', self.vectors, enabled=True, color=(0.03, 0.33, 0.77))
                    
                    if np.any(np.array(indices) != 0):
                        singularity_marker.update_point_positions(singularities)

            ps.set_user_callback(callback)
            ps.show()
                
        