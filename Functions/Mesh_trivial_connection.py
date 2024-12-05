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
import math

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
        
        self.Area = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]], axis=1), axis=1)
        
        print('Genus of the mesh:', self.genus)
        
        self.construct_mesh()
    
    def construct_d0(self):
        '''
            Construct the d0 matrix.
        '''
        for i in tqdm(range(len(self.V)),
                        desc='Constructing d0 matrix',
                        total=len(self.V),
                        leave=False):
            posi_e = np.where(self.E[:, 0] == i)[0]
            nega_e = np.where(self.E[:, 1] == i)[0]
            
            self.d0[posi_e, i] = 1
            self.d0[nega_e, i] = -1
            
    def get_edge_quantities(self):
        '''
            Compute the edge quantities.
        '''
        for i, e in tqdm(enumerate(self.E), 
                        desc='Computing edge quantities',
                        total=len(self.E),
                        leave=False):
            
            fs = np.where(np.sum(np.isin(self.F, e), axis=1) == 2)[0]
            
            self.cot_weights[i] = 3/(self.Area[fs[0]] + self.Area[fs[1]])
            
            B1, B2, normals = (self.B1[fs].squeeze(), 
                            self.B2[fs].squeeze(), 
                            self.normals[fs].squeeze())

            vec_e = self.V[e[1]] - self.V[e[0]]
            
            Z_e = complex_projection(B1, B2, normals, vec_e[None, :])
            
            edge_t = Z_e[0] * np.exp(1j * np.pi / 2)
            edge_t = edge_t.real * self.B1[fs[0]] + edge_t.imag * self.B2[fs[0]]
            
            endpoint_edge_t = np.mean(self.V[e], axis=0) + edge_t
            endpoint_edge_t_oppo = np.mean(self.V[e], axis=0) - edge_t
            
            if np.linalg.norm(endpoint_edge_t - np.mean(self.V[self.F[fs[0]]], axis=0)) >\
                np.linalg.norm(endpoint_edge_t_oppo - np.mean(self.V[self.F[fs[0]]], axis=0)):
                f_r = fs[0]
                f_l = fs[1]
                z_e_r = Z_e[0]
                z_e_l = Z_e[1]
            else:
                f_r = fs[1]
                f_l = fs[0]
                z_e_r = Z_e[1]
                z_e_l = Z_e[0]
                
            self.E_dual[i] = [f_r, f_l]
            
            self.E_dual_dict[tuple(self.E_dual[i])] = i
            
            rotation = np.angle(z_e_l) - np.angle(z_e_r)
            
            self.rotations_across_e[i] = rotation
            
    def get_homology_basis(self):
        
        # Create a graph from the mesh edges
        G = nx.Graph()
        G.add_edges_from(self.E)
        
        tree = nx.minimum_spanning_tree(G)
        E_tree = np.array(tree.edges())
        
        E_included = np.any((self.E[:, None] == E_tree).all(-1) | 
                            (self.E[:, None] == E_tree[:, ::-1]).all(-1), axis=1)
        
        E_dual_remained = self.E_dual[~E_included]
        
        # Construct the dual graph, where the edges 
        # of the previous spanning tree are removed
        G_dual = nx.Graph()
        G_dual.add_edges_from(E_dual_remained)
        tree_dual = nx.minimum_spanning_tree(G_dual)
        E_tree_dual = np.array(tree_dual.edges())
        
        E_dual_included = np.any((self.E_dual[:, None] == E_tree_dual).all(-1) | 
                                 (self.E_dual[:, None] == E_tree_dual[:, ::-1]).all(-1), axis=1)
        
        E_either_included = E_included | E_dual_included
        
        E_co_dual = self.E_dual[~E_either_included]
        
        if len(E_co_dual) != 2*self.genus:
            raise ValueError(f"Expected {2*self.genus} non-contractible edges, but found {len(E_co)}")
        
        # List to store non-contractible cycles
        cycles = []
        self.H = lil_matrix((len(self.E), len(E_co_dual)), dtype=int)
        self.G_H = np.zeros(len(E_co_dual))

        for i, cotree_edge in tqdm(enumerate(E_co_dual),
                                desc="Finding non-contractible cycles", 
                                total=len(E_co_dual),
                                leave=False):
            # Add the cotree edge back to form a cycle
            tree_dual.add_edge(*cotree_edge)
            
            cycle = nx.find_cycle(tree_dual, source=cotree_edge[0])
            
            # Find the cycle created by adding this edge
            cycles.append(cycle)
            
            # Remove the edge again to restore the tree
            tree_dual.remove_edge(*cotree_edge)
            
            for e in cycle:
                if e in self.E_dual_dict:
                    self.H[self.E_dual_dict[e], i] = 1
                    self.G_H[i] += self.rotations_across_e[self.E_dual_dict[e]]
                elif e[::-1] in self.E_dual_dict:
                    self.H[self.E_dual_dict[e[::-1]], i] = -1
                    self.G_H[i] -= self.rotations_across_e[self.E_dual_dict[e[::-1]]]
                else:
                    raise ValueError("Edge not found in the dictionary")
            
            self.G_H[i] = np.mod(self.G_H[i] + np.pi, 2 * np.pi) - np.pi
        
    def get_precond(self):
        self.A_Theta_KKT = bmat([
            [diags(self.cot_weights), self.d0],
            [self.d0.T, None]
        ], format='csc')
        self.lu_Theta_KKT = splu(self.A_Theta_KKT + 1e-8 * eye(self.A_Theta_KKT.shape[0], format='csc'))
        
        self.A_Theta = vstack([
            self.d0.T, self.H.T
        ])
        self.A_Theta_KKT_complete = bmat([
            [diags(self.cot_weights), self.A_Theta.T], 
            [self.A_Theta, None]
        ], format='csc')
        self.lu_Theta_KKT_complete = splu(self.A_Theta_KKT_complete + 1e-8 * eye(self.A_Theta_KKT_complete.shape[0], format='csc'))
        
        # Corner phase computation
        G = nx.Graph()
        G.add_edges_from(self.E_dual)
        tree_dual = nx.minimum_spanning_tree(G)
        E_tree = np.array(tree_dual.edges())
        self.mask_E_tree = np.zeros(len(E_tree), dtype=int)
        
        for i, e_tree in enumerate(E_tree):
            if tuple(e_tree) in self.E_dual_dict:
                self.mask_E_tree[i] = self.E_dual_dict[tuple(e_tree)]
            elif tuple(e_tree[::-1]) in self.E_dual_dict:
                self.mask_E_tree[i] = self.E_dual_dict[tuple(e_tree[::-1])]
            
            E_tree[i] = self.E_dual[self.mask_E_tree[i]]
        
        self.A_phase = lil_matrix((len(E_tree) + 1, len(self.F)))
        self.A_phase[np.arange(len(E_tree)), E_tree[:, 0]] = -1
        self.A_phase[np.arange(len(E_tree)), E_tree[:, 1]] = 1
        self.A_phase[-1, 0] = 1
        
        self.ATA_phase = self.A_phase.T @ self.A_phase
        
        self.lu_phase = splu(self.ATA_phase.tocsc() + 1e-8 * eye(self.ATA_phase.shape[0], format='csc'))

    def construct_mesh(self):
        
        self.d0 = lil_matrix((len(self.E), len(self.V)), dtype=float)
        
        self.E_dual = np.zeros((len(self.E), 2))
        self.E_dual_dict = {}
        self.rotations_across_e = np.zeros(len(self.E))
        self.cot_weights = np.zeros(len(self.E))
        
        for step in tqdm([self.construct_d0, 
                          self.get_edge_quantities,
                          self.get_homology_basis, 
                          self.get_precond], 
                         desc='Constructing the mesh',
                         leave=True):
            step()
        
        self.field_pre = np.ones(len(self.F), dtype=complex)
        
    def process_singularities(self, singularities=None, indices=None):
        assert len(singularities) == len(indices)
        assert np.sum(indices) == 2 - 2*self.genus
        
        self.singularities = singularities[np.array(indices) != 0]
        self.indices = indices[np.array(indices) != 0]
        
        self.I_V = np.zeros(len(self.V))
        self.V_singular = []
        
        for singularity, index in zip(self.singularities, self.indices):
            self.I_V[np.all(np.isclose(self.V, singularity[None, :]), axis=1)] = index
            
            self.V_singular.append(np.where(np.all(np.isclose(self.V, singularity[None, :]), axis=1))[0][0])
               
    def compute_field(self):
        '''
            Compute the set of thetas for each face singularity 
            by constrained optimisation using the KKT conditions.
        '''
        # solution = self.lu_Theta_KKT.solve(np.concatenate([
        #     np.zeros(len(self.E)), 
        #     2 * np.pi * self.I_V - self.G_V
        # ]))
        solution = self.lu_Theta_KKT_complete.solve(np.concatenate([
            np.zeros(len(self.E)), 
            2 * np.pi * self.I_V - self.G_V, 
            - self.G_H
        ]))
            
        Theta = solution[:len(self.E)]
        
        Theta += self.rotations_across_e
        
        self.phase = self.lu_phase.solve(
            self.A_phase.T @ np.append(Theta[self.mask_E_tree], 0)
        )
        
        self.field = np.exp(1j * self.phase)

    def sample_prep_adaptive(self, interval_r=1):
        """
        Fill the surface of a given mesh with a specified density of points.
        
        Parameters:
            V (np.ndarray): Vertices of the mesh (N, 3).
            F (np.ndarray): Faces of the mesh (M, 3), with each face represented by indices into V.
            density (float): The number of points per unit area of the mesh.

        Returns:
            np.ndarray: An (N, 3) array of filled points on the surface.
        """
        self.points_sample = []
        self.F_sample = []
        
        interval = interval_r * np.mean(np.linalg.norm(self.V[self.E[:, 0]] - self.V[self.E[:, 1]], axis=1))

        small_f = (np.linalg.norm(self.V[self.F[:, 0]] - self.V[self.F[:, 1]], axis=1) < (interval * 2)) & \
            (np.linalg.norm(self.V[self.F[:, 1]] - self.V[self.F[:, 2]], axis=1) < (interval * 2)) & \
                (np.linalg.norm(self.V[self.F[:, 2]] - self.V[self.F[:, 0]], axis=1) < (interval * 2))

        self.points_sample += np.mean(self.V[self.F[small_f]], axis=1).tolist() 
        self.F_sample += np.where(small_f)[0].tolist()

        for f in tqdm(np.where(~small_f)[0], desc='Sampling points for field processing', 
                      total=np.sum(~small_f), leave=False):
            
            # Get the vertices of the face
            v0, v1, v2 = self.F[f]
            
            lens = np.linalg.norm(
                self.V[self.F[f]] - np.roll(self.V[self.F[f]], -1, axis=0), axis=1
            )
            
            if np.sum(lens >= interval * 2) == 1:
                if np.argmax(lens) == 0:
                    vs = [v0, v1]
                elif np.argmax(lens) == 1:
                    vs = [v1, v2]
                else:
                    vs = [v2, v0]
                    
                self.points_sample.append(np.mean(self.V[self.F[f]], axis=0))
                
                self.points_sample.append((1/3) * self.points_sample[-1] + (2/3) * self.V[vs[0]])
                self.points_sample.append((1/3) * self.points_sample[-2] + (2/3) * self.V[vs[1]])
                
                self.F_sample += [f, f, f]
                
            elif np.sum(lens >= interval * 2) == 2:
                if np.argmin(lens) == 0:
                    v_far = v2
                    vs = [v0, v1]
                elif np.argmin(lens) == 1:
                    v_far = v0
                    vs = [v1, v2]
                else:
                    v_far = v1
                    vs = [v2, v0]
                
                num_points = math.ceil(((np.linalg.norm(self.V[v_far] - self.V[vs[0]]) + np.linalg.norm(self.V[v_far] - self.V[vs[1]])) / 2) // interval)
                us = np.linspace(1/(2 * num_points), 1-1/(2 * num_points), num_points)
                
                for u in us:
                    self.points_sample.append(u * self.V[v_far] + (1-u)/2 * self.V[vs[0]] + (1-u)/2 * self.V[vs[1]])
                    self.F_sample.append(f)
                    
            else:
                num_points = math.ceil(np.max(lens) // interval)
                margin = 1/(2 * num_points)
        
                for i in range(num_points):
                    for j in range(num_points - i):
                        # Barycentric coordinates
                        u = margin + (i / (num_points-1)) * (1 - 3 * margin)
                        v = margin + (j / (num_points-1)) * (1 - 3 * margin)
                        w = 1 - u - v
                        
                        # Interpolate to get the 3D point in the face
                        self.points_sample.append(
                            u * self.V[v0] + \
                                v * self.V[v1] + \
                                    w * self.V[v2]
                        )
                        self.F_sample.append(f)

        self.points_sample = np.array(self.points_sample)
        self.F_sample = np.array(self.F_sample, dtype=int)
        
        self.Z_sample = complex_projection(
            self.B1[self.F_sample], self.B2[self.F_sample], self.normals[self.F_sample],
            self.points_sample - self.V[self.F[self.F_sample, 0]], 
            diagonal=True
        )

    def sample_prep(self, num_samples=2, margin = 0.2):
        
        self.points_sample = []
        self.F_sample = []
        
        for i in tqdm(range(num_samples), desc='Sampling points for field processing', 
                      total=num_samples, leave=False):
            if num_samples == 1:
                self.points_sample += np.mean(self.V[self.F], axis=1).tolist()
                self.F_sample += np.arange(len(self.F)).tolist()
                
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
                    self.F_sample += np.arange(len(self)).tolist()
                        
        self.points_sample = np.array(self.points_sample)
        self.F_sample = np.array(self.F_sample)
        self.Z_sample = complex_projection(
            self.B1[self.F_sample], self.B2[self.F_sample], self.normals[self.F_sample],
            self.points_sample - self.V[self.F[self.F_sample, 0]], 
            diagonal=True
        )
        
    def sample_field(self):
        vectors_complex = self.field[self.F_sample]

        self.vectors = vectors_complex.real[:, None] * self.B1[self.F_sample] +\
            vectors_complex.imag[:, None] * self.B2[self.F_sample]
            
    def dynamic_field_vertex(self, V_singular, indices, interval_sample=0.5):
            self.sample_prep_adaptive(interval_r=interval_sample)
            
            singularities = self.V[V_singular]
            
            self.process_singularities(singularities, indices)
            
            self.compute_field()
            
            self.sample_field()
            
            ps.init()
            ps_mesh = ps.register_surface_mesh("Input Mesh", self.V, self.F, color=(0.95, 0.98, 1))

            ps_field = ps.register_point_cloud("Field_sample", self.points_sample, enabled=True, radius=0)
            ps_field.add_vector_quantity('Field', self.vectors, #/np.linalg.norm(self.vectors, axis=1)[:, None],
                                         enabled=True, color=(0.03, 0.33, 0.77))
            
            if np.any(np.array(indices) != 0):
                singularity_marker = ps.register_point_cloud("singularity marker", 
                                        np.array([singularities[i] for i in range(len(singularities)) if indices[i] != 0]), 
                                        enabled=True, 
                                        radius=0.0015)
            
            def callback():
                
                # bary_coors_new = bary_coors.copy()
                changed = False
                
                for i, v in enumerate(V_singular):
                    changed_any, V_singular[i] = psim.InputInt(f'{i}-th singular vertex', V_singular[i])
                    
                    if changed_any:
                        changed = True
                
                if changed:
                    singularities = self.V[V_singular]
                    
                    self.process_singularities(singularities, indices)
                    
                    self.compute_field()
            
                    self.sample_field()
                    
                    ps_field.remove_all_quantities()
                    ps_field.add_vector_quantity('Field', self.vectors, #/np.linalg.norm(self.vectors, axis=1)[:, None],
                                                 enabled=True, color=(0.03, 0.33, 0.77))
                    
                    if np.any(np.array(indices) != 0):
                        singularity_marker.update_point_positions(singularities)

            ps.set_user_callback(callback)
            ps.show() 
    