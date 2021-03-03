from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

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


def set_initial_condition(V, mesh):
    coordinates = []
    element = V.element()
    for cell in cells(mesh):
        for i in range(int(element.tabulate_dof_coordinates(cell).size / 2)):
            coordinates.append((element.tabulate_dof_coordinates(cell)[i]).tolist())


    coord = []
    x0 = []
    y0 = []
    for i in coordinates:
        if i not in coord:
            coord.append(i)
            x0.append(i[0])
            y0.append(i[1])

    print(x0)
    print(y0)

    x0 = np.array(sorted(x0))  # Sorting the x coordinates
    y0 = np.array(sorted(y0))  # Sorting the y coordinates
    np.save("x0", x0); np.save("y0", y0)
    u0 = x0
    u0[u0 < 0.2] = -0.0  # if x < 0.2, set u0 = 0.
    u0[u0 >= 0.2] = -85.0  # if x >= 0.2, set u0 = -85.

    u_e = u0
    
    w0 = np.zeros(len(u0))

    return u0, w0


def bidomain_model(V, theta, u_1, u_2, v_1, v_2, u_n1, u_n2, dt):
    sigma_e = 0.62                             # [Sm^-1]
    sigma_i = 0.17                             # [Sm^-1]
    sigma = sigma_i*sigma_e/(sigma_e+sigma_i)  # [Sm^-1]
    chi = 140.                                 # [mm^-1]
    C_m = 0.01                                 # [mu*F*mm−2]
    M_i = (sigma)/(C_m*chi)

    lmda = 0.004
    M_e = lmda * M_i


    F = (
        u_1 * v_1 * dx
        + dt * (dot(M_i * grad(u_2), grad(v_2)) * dx)
        - dt * (dot(M_e * grad(u_2), grad(v_2)) * dx)
        - u_n1 * v_1 * dx
    )

    a, L = lhs(F), rhs(F)

    u = Function(V)  # u from step 2, inital value for step 3
    solve(a == L, u)

    return u



def step(V, T, N, dt, tn, Nx, Ny, degree, u0, w0, theta, derivative):

    v_1, v_2 = TestFunctions(V)

    u = Function(V)
    u_1, u_2 = u.split(deepcopy=True)

    # Step one
    v_values = np.zeros(Nx + 1)
    w_values = np.zeros(Nx + 1)

    t = np.array([tn, tn + theta * dt])
    for i in range(Nx + 1):
        v_values[i], w_values[i] = odeint(derivative, [u0[i], w0[i]], t)[-1]

    print(v_values.size)
    u_n = Function(V)
    print(u_n.vector()[:].size)
    u_n1, u_n2 = u_n.split(deepcopy=True)
    print(u_n1.vector()[:].size)


    # Step two
    #u = bidomain_model(V, theta, u_1, u_2, v_1, v_2, u_n1, u_n2, dt)


    # Step three
    if theta == 0.5:
        new_v_values= np.zeros(Nx + 1)
        new_w_values = np.zeros(Nx + 1)

        u_n = u.vector()[:]
        t = np.array([tn + theta * dt, tn + dt])
        for i in range(Nx + 1):
            new_v_values[i], new_w_values[i] = odeint(derivative, [u_n[i], w_values[i]], t)[-1]

        u_new = Function(V)
        u_new.vector()[:] = new_v_values
        u = u_new


    return u, w_values
    #return v_values, w_values


def run_solver(make_gif):

    theta = 1  # =0.5 Strang/CN and N must be large, =1 Godunov/BE
    degree = 1
    N = 10
    Nx = 2
    Ny = 2
    T = 500.0                       # [ms]
    dt = T / N                      # [ms]
    t = np.linspace(0, T, N+1)      # [ms]
    #mesh = RectangleMesh(Point(0, 0), Point(20, 7), Nx, Ny)
    mesh = UnitSquareMesh(Nx, Ny)
    #V = FunctionSpace(mesh, "P", degree)

    P1 = FiniteElement('P', triangle, degree)
    element = MixedElement([P1, P1])
    V = FunctionSpace(mesh, element)

    plot(mesh)
    plt.savefig("mesh.png")

    u0, w0 = set_initial_condition(V, mesh)
    derivative = fitzhugh_nagumo_reparameterized

    """
    tn = 0
    count = 0
    skip_frames = 10
    for i in range(N + 1):
        print("tn: %0.4f / %0.4f" % (tn, T))
        u, w = step(V, T, N, dt, tn, Nx, Ny, degree, u0, w0, theta, derivative)
        tn += dt
        u0 = u.vector()[:]
        w0 = w
        break

        if make_gif:
            if i == count:
                # Create and save every skip_frames'th plots to file
                plt.clf()
                plt.plot(t, u.vector()[:], label="v")
                #plt.plot(t, u, label="v")
                plt.plot(t, w, label="w")
                plt.axis([0, 400, -100, 100])
                plt.legend()
                plt.title("i=%d" % i)
                plt.savefig(f"plots/u{i:04d}.png")

                count += skip_frames


    if make_gif:

        import glob
        from PIL import Image
        import os

        filepath_in = "plots/u*.png"
        filepath_out = "bidomain_animation.gif"

        # Collecting the plots and putting them in a list
        img, *imgs = [(Image.open(f)) for f in sorted(glob.glob(filepath_in))]

        # Create GIF
        img.save(fp=filepath_out,
            format="GIF",
            append_images=imgs,
            save_all=True,
            duration=200,
            loop=0,
        )

        # Delete all plots in faile after creating the GIF
        [os.remove(f) for f in sorted(glob.glob(filepath_in))]
    """

if __name__ == "__main__":
    run_solver(make_gif=False)
