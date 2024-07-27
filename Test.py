import polyscope as ps
import polyscope.imgui as psim
import os
from Functions.Auxiliary import *

ui_int = 1

V, F = load_off_file(os.path.join('..', 'data', 'spherers.off'))

vectors = np.random.rand(V.shape[0], 3)

ps.init()

mesh = ps.register_surface_mesh("Input Mesh", V, F, enabled=True)

mesh.add_vector_quantity("Normals", vectors, enabled=True)

# Define our callback function, which Polyscope will repeatedly execute while running the UI.
# We can write any code we want here, but in particular it is an opportunity to create ImGui 
# interface elements and define a custom UI.
def callback():

    global ui_int
    # == Settings

    # Use settings like this to change the UI appearance.
    # Note that it is a push/pop pair, with the matching pop() below.
    psim.PushItemWidth(150)

    psim.TextUnformatted("Some sample text")
    psim.TextUnformatted("An important value: {}".format(42))
    psim.Separator()

    # Input ints
    changed, ui_int = psim.InputInt("ui_int", ui_int, step=1, step_fast=3) 
    if changed:
        mesh.remove_all_quantities()
        mesh.add_vector_quantity(f"Normals {ui_int}", vectors * ui_int, enabled=True)

    psim.PopItemWidth()


ps.init() 
ps.set_user_callback(callback)
ps.show()