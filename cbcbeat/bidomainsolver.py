"""
These solvers solve the (pure) bidomain equations on the form: find
the transmembrane potential :math:`v = v(x, t)` and the extracellular
potential :math:`u = u(x, t)` such that

.. math::

   v_t - \mathrm{div} ( G_i v + G_i u) = I_s

   \mathrm{div} (G_i v + (G_i + G_e) u) = I_a

where the subscript :math:`t` denotes the time derivative; :math:`G_x`
denotes a weighted gradient: :math:`G_x = M_x \mathrm{grad}(v)` for
:math:`x \in \{i, e\}`, where :math:`M_i` and :math:`M_e` are the
intracellular and extracellular cardiac conductivity tensors,
respectively; :math:`I_s` and :math:`I_a` are prescribed input. In
addition, initial conditions are given for :math:`v`:

.. math::

   v(x, 0) = v_0

Finally, boundary conditions must be prescribed. For now, this solver
assumes pure homogeneous Neumann boundary conditions for :math:`v` and
:math:`u` and enforces the additional average value zero constraint
for u.

"""

# Copyright (C) 2013 Marie E. Rognes (meg@simula.no)
# Use and modify at will
# Last changed: 2013-04-18

__all__ = ["BasicBidomainSolver", "BidomainSolver"]

from cbcbeat.dolfinimport import *
from cbcbeat.markerwisefield import *
from cbcbeat.utils import end_of_time, annotate_kwargs
from cbcbeat import debug

class BasicBidomainSolver(object):
    """This solver is based on a theta-scheme discretization in time
    and CG_1 x CG_1 (x R) elements in space.

    .. note::

       For the sake of simplicity and consistency with other solver
       objects, this solver operates on its solution fields (as state
       variables) directly internally. More precisely, solve (and
       step) calls will act by updating the internal solution
       fields. It implies that initial conditions can be set (and are
       intended to be set) by modifying the solution fields prior to
       simulation.

    *Arguments*
      mesh (:py:class:`dolfin.Mesh`)
        The spatial domain (mesh)

      time (:py:class:`dolfin.Constant` or None)
        A constant holding the current time. If None is given, time is
        created for you, initialized to zero.

      M_i (:py:class:`ufl.Expr`)
        The intracellular conductivity tensor (as an UFL expression)

      M_e (:py:class:`ufl.Expr`)
        The extracellular conductivity tensor (as an UFL expression)

      I_s (:py:class:`dict`, optional)
        A typically time-dependent external stimulus given as a dict,
        with domain markers as the key and a
        :py:class:`dolfin.Expression` as values. NB: it is assumed
        that the time dependence of I_s is encoded via the 'time'
        Constant.

      I_a (:py:class:`dolfin.Expression`, optional)
        A (typically time-dependent) external applied current

      v\_ (:py:class:`ufl.Expr`, optional)
        Initial condition for v. A new :py:class:`dolfin.Function`
        will be created if none is given.

      params (:py:class:`dolfin.Parameters`, optional)
        Solver parameters

      """
    def __init__(self, mesh, heart_mesh, torso_mesh, time, M_i, M_e, M_T, I_s=None, I_a=None, v_=None,
                 params=None):

        # Check some input
        assert isinstance(mesh, Mesh), \
            "Expecting mesh to be a Mesh instance, not %r" % mesh
        assert isinstance(heart_mesh, Mesh), \
            "Expecting heart_mesh to be a Mesh instance, not %r" % heart_mesh
        assert isinstance(torso_mesh, Mesh), \
            "Expecting torso_mesh to be a Mesh instance, not %r" % torso_mesh
        assert isinstance(time, Constant) or time is None, \
            "Expecting time to be a Constant instance (or None)."
        assert isinstance(params, Parameters) or params is None, \
            "Expecting params to be a Parameters instance (or None)"

        self._nullspace_basis = None


        # Store input
        self._mesh = mesh
        self._heart_mesh = heart_mesh
        self._torso_mesh = torso_mesh
        self._time = time
        self._M_i = M_i
        self._M_e = M_e
        self._M_T = M_T
        self._I_s = I_s
        self._I_a = I_a

        # Initialize and update parameters if given
        self.parameters = self.default_parameters()
        if params is not None:
            self.parameters.update(params)

        # Set-up function spaces
        k = self.parameters["polynomial_degree"]

        V = FunctionSpace(self._heart_mesh, "CG", k)
        U = FunctionSpace(self._mesh, "CG", k)
        T = FunctionSpace(self._torso_mesh, "CG", k)

        use_R = self.parameters["use_avg_u_constraint"]
        if use_R:
            R = FunctionSpace(self._mesh, "R", 0)
            self.VUR = MixedFunctionSpace(V, U, T, R)
        else:
            self.VUR = MixedFunctionSpace(V, U, T)

        self.V = V

        # Set-up solution fields:
        if v_ is None:
            self.merger = FunctionAssigner(V, self.VUR.sub_space(0))
            self.v_ = Function(self.VUR.sub_space(0), name="v_")
        else:
            debug("Experimental: v_ shipped from elsewhere.")
            self.merger = None
            self.v_ = v_

        self.vur = Function(self.VUR)

        # Figure out whether we should annotate or not
        self._annotate_kwargs = annotate_kwargs(self.parameters)

    def function_space(self):
        "Return the MixedFunctionSpace"
        return self.VUR

    @property
    def time(self):
        "The internal time of the solver."
        return self._time

    def solution_fields(self):
        """
        Return tuple of previous and current solution objects.

        Modifying these will modify the solution objects of the solver
        and thus provides a way for setting initial conditions for
        instance.

        *Returns*
          (previous v, current vur) (:py:class:`tuple` of :py:class:`dolfin.Function`)
        """

        return (self.v_, self.vur)

    def solve(self, interval, dt=None):
        """
        Solve the discretization on a given time interval (t0, t1)
        with a given timestep dt and return generator for a tuple of
        the interval and the current solution.

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval for the solve given by (t0, t1)
          dt (int, optional)
            The timestep for the solve. Defaults to length of interval

        *Returns*
          (timestep, solution_fields) via (:py:class:`genexpr`)

        *Example of usage*::

          # Create generator
          solutions = solver.solve((0.0, 1.0), 0.1)

          # Iterate over generator (computes solutions as you go)
          for (interval, solution_fields) in solutions:
            (t0, t1) = interval
            v_, vur = solution_fields
            # do something with the solutions
        """
        timer = Timer("PDE step")

        # Initial set-up
        # Solve on entire interval if no interval is given.
        (T0, T) = interval
        if dt is None:
            dt = (T - T0)
        t0 = T0
        t1 = T0 + dt

       # Step through time steps until at end time
        while (True) :
            info("Solving on t = (%g, %g)" % (t0, t1))
            self.step((t0, t1))

            # Yield solutions
            yield (t0, t1), self.solution_fields()

            # Break if this is the last step
            if end_of_time(T, t0, t1, dt):
                break

            # If not: update members and move to next time
            # Subfunction assignment would be good here.
            if isinstance(self.v_, Function):
                self.merger.assign(self.v_, self.vur.sub(0))
            else:
                debug("Assuming that v_ is updated elsewhere. Experimental.")
            t0 = t1
            t1 = t0 + dt

    def step(self, interval):
        """
        Solve on the given time interval (t0, t1).

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval (t0, t1) for the step

        *Invariants*
          Assuming that v\_ is in the correct state for t0, gives
          self.vur in correct state at t1.
        """

        timer = Timer("PDE step")

        # Extract interval and thus time-step
        (t0, t1) = interval
        k_n = Constant(t1 - t0)
        theta = self.parameters["theta"]

        # Extract conductivities
        M_i, M_e = self._M_i, self._M_e

        # Define variational formulation
        use_R = self.parameters["use_avg_u_constraint"]
        if use_R:
             (v, u, l) = TrialFunctions(self.VUR)
             (w, q, lamda) = TestFunctions(self.VUR)
        else:
             (v, u, l) = TrialFunctions(self.VUR)
             (w, q, lamda) = TestFunctions(self.VUR)

        # Set time
        t = t0 + theta*(t1 - t0)
        self.time.assign(t)

        # Set-up measure and rhs from stimulus
        (dz, rhs) = rhs_with_markerwise_field(self._I_s, self._heart_mesh, w)

        # Set up integration domain
        dV = Measure("dx", domain=self.VUR.sub_space(0).mesh())
        dU = Measure("dx", domain=self.VUR.sub_space(1).mesh())
        dT = Measure("dx", domain=self.VUR.sub_space(2).mesh())

        # Set-up variational problem
        a = v * w * dV \
            + theta * k_n * (dot(M_i * grad(v), grad(w)) * dV) \
            + k_n * (dot(M_i * grad(u), grad(w)) * dV) \
            + k_n * (dot(M_i * grad(v), grad(q)) * dV) \
            + (k_n/theta) * (dot((M_i + M_e) * grad(u), grad(q)) * dU) \
            + (k_n/theta) * (dot((M_T) * grad(u), grad(q)) * dT)

        L = (self.v_ * w * dV) \
            - (1. - theta) * k_n * (dot(M_i * grad(self.v_), grad(w)) * dV) \
            - ((1. - theta)/theta) * (dot(M_i * grad(self.v_), grad(q)) * dV) \
            + rhs


        if use_R:
            a += k_n*(lamda*u + l*q)*dV

        if self._I_a:
            L += k_n*self._I_a*q*dV


        solve(a == L, self.vur)


    @staticmethod
    def default_parameters():
        """Initialize and return a set of default parameters

        *Returns*
          A set of parameters (:py:class:`dolfin.Parameters`)

        To inspect all the default parameters, do::

          info(BasicBidomainSolver.default_parameters(), True)
        """

        params = Parameters("BasicBidomainSolver")
        params.add("enable_adjoint", True)
        params.add("theta", 0.5)
        params.add("polynomial_degree", 1)
        params.add("use_avg_u_constraint", True)

        return params

class BidomainSolver(BasicBidomainSolver):
    __doc__ = BasicBidomainSolver.__doc__

    def __init__(self, mesh, heart_mesh, torso_mesh, time, M_i, M_e, M_T, I_s=None, I_a=None, v_=None,
                 params=None):

        # Call super-class
        BasicBidomainSolver.__init__(self, mesh, heart_mesh, torso_mesh, time, M_i, M_e, M_T,
                                     I_s=I_s, I_a=I_a, v_=v_,
                                     params=params)

        # Check consistency of parameters first
        if self.parameters["enable_adjoint"] and not dolfin_adjoint:
            warning("'enable_adjoint' is set to True, but no "\
                    "dolfin_adjoint installed.")

        # Mark the timestep as unset
        self._timestep = None


    @staticmethod
    def default_parameters():
        """Initialize and return a set of default parameters

        *Returns*
          A set of parameters (:py:class:`dolfin.Parameters`)

        To inspect all the default parameters, do::

          info(BidomainSolver.default_parameters(), True)
        """
        params = Parameters("BidomainSolver")
        params.add("enable_adjoint", True)
        params.add("theta", 0.5)
        params.add("polynomial_degree", 1)

        # Set default solver type to be iterative
        params.add("linear_solver_type", "iterative")
        params.add("use_avg_u_constraint", False)

        # Set default iterative solver choices (used if iterative
        # solver is invoked)
        params.add("algorithm", "cg")
        params.add("preconditioner", "petsc_amg")
        #params.add("preconditioner", "fieldsplit") # This seg faults

        # Add default parameters from both LU and Krylov solvers
        params.add(LUSolver.default_parameters())
        petsc_params = PETScKrylovSolver.default_parameters()
        # FIXME: work around DOLFIN bug #583. Just deleted this when fixed.
        petsc_params.update({"convergence_norm_type": "preconditioned"})
        params.add(petsc_params)

        # Customize default parameters for PETScKrylovSolver
        #params["petsc_krylov_solver"]["preconditioner"]["structure"] = "same"

        return params

    def variational_forms(self, k_n):
        """Create the variational forms corresponding to the given
        discretization of the given system of equations.

        *Arguments*
          k_n (:py:class:`ufl.Expr` or float)
            The time step

        *Returns*
          (lhs, rhs) (:py:class:`tuple` of :py:class:`ufl.Form`)

        """

        # Extract theta parameter and conductivities
        theta = self.parameters["theta"]
        M_i = self._M_i
        M_e = self._M_e
        M_T = self._M_T

        # Define variational formulation
        use_R = self.parameters["use_avg_u_constraint"]
        if use_R:
             (v, u, a, l) = TrialFunctions(self.VUR)
             (w, q, b, lamda) = TestFunctions(self.VUR)
        else:
            (v, u, a) = TrialFunctions(self.VUR)
            (w, q, b) = TestFunctions(self.VUR)

        # Set-up measure and rhs from stimulus
        (dz, rhs) = rhs_with_markerwise_field(self._I_s, self._heart_mesh, w)

        # Set up integration domain
        dV = Measure("dx", domain=self.VUR.sub_space(0).mesh())
        dU = Measure("dx", domain=self.VUR.sub_space(1).mesh())
        dT = Measure("dx", domain=self.VUR.sub_space(2).mesh())

        # Set-up variational problem
        a = v * w * dV \
            + theta * k_n * (dot(M_i * grad(v), grad(w)) * dV) \
            + k_n * (dot(M_i * grad(u), grad(w)) * dV) \
            + k_n * (dot(M_i * grad(v), grad(q)) * dV) \
            + (k_n/theta) * (dot((M_i + M_e) * grad(u), grad(q)) * dV) \
            + (k_n/theta) * (dot((M_T) * grad(u), grad(q)) * dT)

        L = (self.v_ * w * dV) \
            - (1. - theta) * k_n * (dot(M_i * grad(self.v_), grad(w)) * dV) \
            - ((1. - theta)/theta) * k_n * (dot(M_i * grad(self.v_), grad(q)) * dV) \
            + k_n*rhs


        if use_R:
            a += k_n*(lamda*u + l*q)*dV

        if self._I_a:
            L += k_n*self._I_a*q*dV

        return (a, L)


    def step(self, interval):
        """
        Solve on the given time step (t0, t1).

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval (t0, t1) for the step

        *Invariants*
          Assuming that v\_ is in the correct state for t0, gives
          self.vur in correct state at t1.
        """

        timer = Timer("PDE step")
        #solver_type = self.parameters["linear_solver_type"]

        # Extract interval and thus time-step
        (t0, t1) = interval
        dt = t1 - t0
        theta = self.parameters["theta"]
        t = t0 + theta*dt
        self.time.assign(t)

        # Update matrix and linear solvers etc as needed
        if self._timestep is None:
            self._timestep = Constant(dt)

        (self._lhs, self._rhs) = self.variational_forms(self._timestep)
        solve(self._lhs == self._rhs, self.vur)
