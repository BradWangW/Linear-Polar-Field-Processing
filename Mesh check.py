import polyscope as ps
from Functions.Auxiliary import *
from Functions.Mesh_noenforceangle import Triangle_mesh
import os

if __name__ == '__main__':
    
    V, F = load_off_file(os.path.join('..', 'data_patho', 'dragon.off'))
    print(len(F))
    
    mesh = Triangle_mesh(V, F)

    ps.init()
    ps_mesh = ps.register_surface_mesh("Input Mesh", V, F, color=(0.95, 0.98, 1))
    
    normal_field = ps.register_point_cloud("Normal field", 
                                           (mesh.V[F[:, 0]] + mesh.V[F[:, 1]] + mesh.V[F[:, 2]])/3, 
                                           enabled=False, radius=0)
    normal_field.add_vector_quantity('Field', mesh.normals, enabled=True, length=0.01)

    ps.show()
            
