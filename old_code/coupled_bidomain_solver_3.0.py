from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from dolfin import *
np.set_printoptions(threshold=np.inf)

# Turn on FFC/FEniCS optimizations
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
parameters["form_compiler"]["quadrature_degree"] = 3

# Turn off adjoint functionality
import cbcbeat
if cbcbeat.dolfin_adjoint:
    parameters["adjoint"]["stop_annotating"] = True


def fitzhugh_nagumo_reparameterized(v, t):
    a = 0.13; b = 0.013            # Constant
    c1 = 0.26; c2 = 0.1; c3 = 1.0  # Constant
    i_app = 0.                     # Constant

    v_rest = -85.            # [mV]
    v_peak = 40.             # [mV]
    v_amp = v_peak - v_rest  # [mV]
    v_th = v_rest + a*v_amp  # [mV]
    I_app = v_amp * i_app    # [mV]

    dVdt = c1*(v[0] - v_rest)*(v[0] - v_th)*(v_peak - v[0])/(v_amp**2) - (c2*(v[0] - v_rest)*v[1])/v_amp
    dWdt = b*(v[0] - v_rest - c3*v[1])


    if t >= 50 and t <= 60:
        dVdt += I_app

    return dVdt, dWdt


def bidomain_model(W, theta, v_n, dt, heart_cells):
    sigma_e = 0.62                      # [Sm^-1]
    sigma_i = 0.17                      # [Sm^-1]
    chi = 140.                          # [mm^-1]
    C_m = 0.01                          # [mu*F*mm−2]
    M_i = (sigma_i)/(C_m*chi)
    M_e = (sigma_e)/(C_m*chi)
    M_o = 0.25*M_e

    v, u_e = TrialFunctions(W)
    psi_v, psi_ue = TestFunctions(W)

    dV = Measure("dx", domain=W.sub_space(0).mesh())
    dU = Measure("dx", domain=W.sub_space(1).mesh(), subdomain_data=heart_cells)

    #print("parent id = ", W.sub_space(1).mesh().id())
    #print("heart id = ", W.sub_space(0).mesh().id())
    #print(dU(0))
    #print(dU(1))


    if theta == 1:
        """F = (
            v * psi_v * dV
            + dt * (inner(M_i * grad(v), grad(psi_v)) * dV)
            + dt * (inner(M_i * grad(u_e), grad(psi_v)) * dV)
            + dt * (inner(M_i * grad(v), grad(psi_ue)) * dU)
            + dt * (inner((M_i + M_e) * grad(u_e), grad(psi_ue)) * dU)
            + dt * (inner(M_o * grad(u_e), grad(psi_ue)) * dU)
            - (v_n * psi_v * dV)
        )"""

        # dt * (inner(M_i * grad(v), grad(psi_ue)) * dU) originally line

        a = v * psi_v * dV \
            + dt * (inner(M_i * grad(v), grad(psi_v)) * dV) \
            + dt * (inner(M_i * grad(u_e), grad(psi_v)) * dV) \
            + dt * (inner(M_i * grad(v), grad(psi_ue)) * dV) \
            + dt * (inner((M_i + M_e) * grad(u_e), grad(psi_ue)) * dU(1)) \
            + dt * (inner(M_o * grad(u_e), grad(psi_ue)) * dU(0))

        L = v_n * psi_v * dV

    else:
        F = (
            v * psi_v * dV
            + theta * dt * (inner(M_i * grad(v), grad(psi_v)) * dV)
            + dt * (inner(M_i * grad(u_e), grad(psi_v)) * dV)
            + dt * (inner(M_i * grad(v), grad(psi_ue)) * dU)
            + (dt/theta) * (inner((M_i + M_e) * grad(u_e), grad(psi_ue)) * dU)
            + (dt/theta) * (inner(M_o * grad(u_e), grad(psi_ue)) * dU)
            - (v_n * psi_v * dV)
            + (1 - theta) * dt * (inner(M_i * grad(v_n), grad(psi_v)) * dV)
            + ((1 - theta)/theta) * (inner(M_i * grad(v_n), grad(psi_ue)) * dU)
        )

    #a, L = lhs(F), rhs(F)

    vu = Function(W)
    solve(a == L, vu)
    v, u_e = vu.split(True)



    return v, u_e


def step(W, T, N, dt, tn, Nx, Ny, degree, v0, w0, theta, derivative, heart_cells):

    v0 = np.array(v0)
    w0 = np.array(w0)

    # Step one
    v_values = np.zeros(len(v0))
    w_values = np.zeros(len(v0))

    t = np.array([tn, tn + theta * dt])
    for i in range(len(v0)):
        v_values[i], w_values[i] = odeint(derivative, [v0[i], w0[i]], t)[-1]

    v_n = Function(W.sub_space(0))
    v_n.vector()[:] = v_values

    v = v_n.vector()[:]

    # Step two
    v, u_e = bidomain_model(W, theta, v_n, dt, heart_cells)


    # Step three
    if theta == 0.5:
        new_v_values= np.zeros(len(v0))
        new_w_values = np.zeros(len(v0))

        v_n = v.vector()[:]
        t = np.array([tn + theta * dt, tn + dt])
        for i in range(len(v0)):
            new_v_values[i], new_w_values[i] = odeint(derivative, [v_n[i], w_values[i]], t)[-1]

        v_new = Function(V)
        v_new.vector()[:] = new_v_values
        v = v_new


    return v, w_values, u_e


def save_for_line_plot(v, name):
    tol = 0.001  # avoid hitting points outside the domain
    x = np.linspace(tol, 20 - tol, 101)
    points = [(x_, 10) for x_ in x]  # 2D points
    v_line = np.array([v(point) for point in points])
    np.savetxt(name, v_line)


def run_solver(make_gif, dimension):

    theta = 1  # =0.5 Strang/CN and N must be large, =1 Godunov/BE
    degree = 1
    N = 200
    Nx = 50
    Ny = 50
    T = 750                         # [ms]
    dt = T / N                      # [ms]
    t = np.linspace(0, T, N+1)      # [ms]

    if dimension == "1D":
        mesh = IntervalMesh(Nx, 0, 20)
        v0 = Expression('x[0] <= 2.0 ? 0 : -85', degree=0)
        submesh = IntervalMesh(Nx, 0, 20)


    if dimension == "2D":
        #mesh = RectangleMesh(Point(0, 0), Point(20, 20), Nx, Ny)
        mesh = UnitSquareMesh(50, 50)
        v0 = Expression('(x[0] <= 2.0 && x[1] <= 2.0) ? 0 : -85', degree=0)

        def circle_heart(x,y):
            r = 0.25
            xshift = x - 0.5
            yshift = y - 0.5
            return xshift*xshift + yshift*yshift < r*r


        marker = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)

        for c in cells(mesh):
            marker[c] = circle_heart(c.midpoint().x(), c.midpoint().y()) ## Beutel heart

        submesh = MeshView.create(marker, 1) # Heart


    print("parent id = ", mesh.id())
    print("heart id  = ", submesh.id())
    #H_space = FiniteElement("CG", submesh.ufl_cell(), 1)
    #V_space = FiniteElement("CG", mesh.ufl_cell(), 1)
    #W = FunctionSpace(mesh, MixedElement((H_space,V_space)))

    V = FunctionSpace(submesh, "CG", 1)
    U = FunctionSpace(mesh, "CG", 1)
    W = MixedFunctionSpace(V, U)

    mapping = submesh.topology().mapping()[mesh.id()]
    cell_map = mapping.cell_map()
    assert mapping and mapping.mesh().id() == mesh.id(), \
        "The CardiacModel mesh should be built from TorsoModel mesh (MeshView)"

    heart_cells = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    for c in cells(submesh):
        idx = int(cell_map[c.index()])
        heart_cells[idx] = 1;


    #v0 = interpolate(v0, W.sub_space(0))   #Finds all the x and y points that fullfills the criteria given in Expression
    v0 = Function(W.sub_space(0), name="v0")


    #x0 = Expression('x[0]', degree=0)
    #x0 = interpolate(x0, W.sub_space(0))   #Finds all the x points that is used in functionspace V

    v0 = v0.vector()[:]

    w0 = np.zeros(len(v0))

    derivative = fitzhugh_nagumo_reparameterized

    tn = 0
    count = 0
    count_index = 0
    skip_frames = 10
    v_array = np.zeros((3,N+1))

    out_v = File("paraview/coupled_bidomain_v.pvd")
    out_u = File("paraview/coupled_bidomain_u.pvd")


    for i in range(N + 1):
        print("tn: %0.4f / %0.4f" % (tn, T))
        v, w, u_e = step(W, T, N, dt, tn, Nx, Ny, degree, v0, w0, theta, derivative, heart_cells)


        tn += dt
        v0 = v.vector()[:]
        w0 = w


        out_v << v
        out_u << u_e

        if make_gif:
            if i == count:
                # Create and save every skip_frames'th plots to file
                plt.clf()
                if dimension == "1D":
                    plt.plot(x0.vector()[:], v.vector()[:], label="v")
                    plt.plot(x0.vector()[:], w, label="w")
                    plt.axis([0, 20, -100, 100])
                    plt.legend()
                    plt.xlabel("[mm]")
                    plt.ylabel("[mV]")

                if dimension == "2D":
                    c = plot(v, mode='color', vmin=-85, vmax=40)
                    plt.colorbar(c, orientation='vertical')
                    plt.xlabel("x [mm]")
                    plt.ylabel("y [mm]")

                plt.title("i=%d" % i)
                plt.savefig(f"plots/u{i:04d}.png")
                if tn == T - 1*dt:
                    plt.savefig(f"v_final.png")

                count += skip_frames


    if make_gif:

        import glob; import os
        from PIL import Image

        filepath_in = "plots/u*.png"
        filepath_out = "bidomain_%s_hjørne.gif" % dimension

        # Collecting the plots and putting them in a ordered list
        img, *imgs = [(Image.open(f)) for f in sorted(glob.glob(filepath_in))]

        # Create GIF
        img.save(fp=filepath_out,
            format="GIF",
            append_images=imgs,
            save_all=True,
            duration=200,
            loop=0,
        )

        # Delete all plots in file after creating the GIF
        [os.remove(f) for f in sorted(glob.glob(filepath_in))]




if __name__ == "__main__":
    run_solver(make_gif=False, dimension="2D")
