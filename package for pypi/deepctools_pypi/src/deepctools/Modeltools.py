"""
Name: Modeltools.py
Author: Xuewen Zhang
Date:at 13/04/2024
version: 1.0.0
Description: Toolbox to formulate the casadi Function
"""
import casadi as cs
import numpy as np
import warnings

# packages from deepctools
from . import util


# Using a numpy array of casadi MX symbols is almost always a bad idea, so we
# warn the user if they request it. Users who know what they are doing can
# disable the warning via this constant (see __getCasadiFunc).
WARN_NUMPY_MX = True

def getCasadiFunc(f, varsizes=None, varnames=None, funcname=None, rk4=False,
                  Delta=1, M=1, scalar=None, casaditype=None, wraps=None,
                  numpy=None):
    """
    Takes a function handle and turns it into a Casadi function.

    f should be defined to take a specified number of arguments and return a
    scalar, list, or numpy array. varnames, if specified, gives names to each
    of the inputs, but this is not required.

    sizes should be a list of how many elements are in each one of the inputs.

    Alternatively, instead of specifying varsizes, varnames, and funcname,
    you can pass a casadi.Function as wraps to copy these values from the other
    function.

    The numpy argument determines whether arguments are passed with numpy
    array semantics or not. By default, numpy=True, which means symbolic
    variables are passed as numpy arrays of Casadi scalar symbolics. This means
    your function should be written to accept (and should also return) numpy
    arrays. If numpy=False, the arguments are passed as Casadi symbolic
    vectors, which have slightly different semantics. Note that 'scalar'
    is a deprecated synonym for numpy.

    To choose what type of Casadi symbolic variables to use, pass
    casaditype="SX" or casaditype="MX". The default value is "SX" if
    numpy=True, and "MX" if numpy=True.
    """
    # Decide if user specified wraps.
    if wraps is not None:
        if not isinstance(wraps, cs.Function):
            raise TypeError("wraps must be a casadi.Function!")
        if varsizes is None:
            varsizes = [wraps.size_in(i) for i in range(wraps.n_in())]
        if varnames is None:
            varnames = [wraps.name_in(i) for i in range(wraps.n_in())]
        if funcname is None:
            funcname = wraps.name()

    # Pass the buck to the sub function.
    if varsizes is None:
        raise ValueError("Must specify either varsizes or wraps!")
    if funcname is None:
        funcname = "f"
    if numpy is None and scalar is not None:
        numpy = scalar
        warnings.warn("Passing 'scalar' is deprecated. Replace with 'numpy'.")
    symbols = __getCasadiFunc(f, varsizes, varnames, funcname,
                              numpy=numpy, casaditype=casaditype,
                              allowmatrix=True)
    args = symbols["casadiargs"]
    fexpr = symbols["fexpr"]

    # Evaluate function and make a Casadi object.
    fcasadi = cs.Function(funcname, args, [fexpr], symbols["names"],
                              [funcname])

    # Wrap with rk4 if requested.
    if rk4:
        frk4 = util.rk4(fcasadi, args[0], args[1:], Delta, M)
        fcasadi = cs.Function(funcname, args, [frk4], symbols["names"],
                                  [funcname])

    return fcasadi


def getCasadiIntegrator(f, Delta, argsizes, argnames=None, funcname="int_f",
                        abstol=1e-8, reltol=1e-8, wrap=True, verbosity=1,
                        scalar=None, casaditype=None, numpy=None):
    """
    Gets a Casadi integrator for function f from 0 to Delta.

    Argsizes should be a list with the number of elements for each input. Note
    that the first argument is assumed to be the differential variables, and
    all others are kept constant.

    The scalar, casaditype, and numpy arguments all have the same behavior as
    in getCasadiFunc. See getCasadiFunc documentation for more details.

    wrap can be set to False to return the raw casadi Integrator object, i.e.,
    with inputs x and p instead of the arguments specified by the user.
    """
    # First get symbolic expressions.
    if numpy is None and scalar is not None:
        numpy = scalar
        warnings.warn("Keyword argument 'scalar=...' is deprecated. Replace "
                      "with 'numpy=...'.")
    symbols = __getCasadiFunc(f, argsizes, argnames, funcname,
                              numpy=numpy, casaditype=casaditype,
                              allowmatrix=False)
    x0 = symbols["casadiargs"][0]
    par = symbols["casadiargs"][1:]
    fexpr = symbols["fexpr"]

    # Build ODE and integrator.
    ode = dict(x=x0, p=cs.vertcat(*par), ode=fexpr)
    options = {
        "abstol": abstol,
        "reltol": reltol,
        "tf": Delta,
        "disable_internal_warnings": verbosity <= 0,
        "verbose": verbosity >= 2,
    }
    integrator = cs.integrator(funcname, "cvodes", ode, options)

    # Now do the subtle bit. Integrator has arguments x0 and p, but we need
    # arguments as given by the user. First we need MX arguments.
    if wrap:
        names = symbols["names"]
        sizes = symbols["sizes"]
        wrappedx0 = cs.MX.sym(names[0], *sizes[0])
        wrappedpar = [cs.MX.sym(names[i], *sizes[i]) for i
                      in range(1, len(sizes))]
        wrappedIntegrator = integrator(x0=wrappedx0,
                                       p=cs.vertcat(*wrappedpar))["xf"]
        integrator = cs.Function(funcname, [wrappedx0] + wrappedpar,
                                     [wrappedIntegrator], symbols["names"],
                                     [funcname])
    return integrator


def __getCasadiFunc(f, varsizes, varnames=None, funcname="f", numpy=None,
                    casaditype=None, allowmatrix=True):
    """
    Core logic for getCasadiFunc and its relatives.

    casaditype chooses what type of casadi variable to use, while numpy chooses
    to wrap the casadi symbols in a NumPy array before calling f. Both
    numpy and casaditype are None by default; the table below shows what values
    are used in the various cases.

                  +----------------------+-----------------------+
                  |       numpy is       |       numpy is        |
                  |         None         |       not None        |
    +-------------+----------------------+-----------------------+
    | casaditype  | casaditype = "SX"    | casaditype = ("SX" if |
    |  is None    | numpy = True         |   numpy else "MX")    |
    +------------------------------------+-----------------------+
    | casaditype  | numpy = (False if    | warning issued if     |
    | is not None |   casaditype == "MX" |   numpy == True and   |
    |             |   else True)         |   casaditype == "MX"  |
    +------------------------------------+-----------------------+

    Returns a dictionary with the following entries:

    - casadiargs: a list of the original casadi symbolic primitives

    - numpyargs: a list of the numpy analogs of the casadi symbols. Note that
                 this is None if numpy=False.

    - fargs: the list of arguments passed to f. This is numpyargs if numpyargs
             is not None; otherwise, it is casadiargs.

    - fexpr: the casadi expression resulting from evaluating f(*fargs).

    - XX: is either casadi.SX or casadi.MX depending on what type was used
          to create casadiargs.

    - names: a list of string names for each argument.

    - sizes: a list of one- or two-element lists giving the sizes.
    """
    # Check names.
    if varnames is None:
        varnames = ["x%d" % (i,) for i in range(len(varsizes))]
    else:
        varnames = [str(n) for n in varnames]
    if len(varsizes) != len(varnames):
        raise ValueError("varnames must be the same length as varsizes!")

    # Loop through varsizes in case some may be matrices.
    realvarsizes = []
    for s in varsizes:
        goodInput = True
        try:
            s = [int(s)]
        except TypeError:
            if allowmatrix:
                try:
                    s = list(s)
                    goodInput = len(s) <= 2
                except TypeError:
                    goodInput = False
            else:
                raise TypeError("Entries of varsizes must be integers!")
        if not goodInput:
            raise TypeError("Entries of varsizes must be integers or "
                            "two-element lists!")
        realvarsizes.append(s)

    # Decide which Casadi type to use and whether to wrap as a numpy array.
    # XX is either casadi.SX or casadi.MX.
    if numpy is None and casaditype is None:
        numpy = True
        casaditype = "SX"
    elif numpy is None:
        numpy = False if casaditype == "MX" else True
    elif casaditype is None:
        casaditype = "SX" if numpy else "MX"
    else:
        if numpy and (casaditype == "MX") and WARN_NUMPY_MX:
            warnings.warn("Using a numpy array of casadi MX is almost always "
                          "a bad idea. Consider refactoring to avoid.")
    XX = dict(SX=cs.SX, MX=cs.MX).get(casaditype, None)
    if XX is None:
        raise ValueError("casaditype must be either 'SX' or 'MX'!")

    # Now make the symbolic variables. Make numpy versions if requested.
    casadiargs = [XX.sym(name, *size)
                  for (name, size) in zip(varnames, realvarsizes)]
    if numpy:
        numpyargs = [__casadi_to_numpy(x) for x in casadiargs]
        fargs = numpyargs
    else:
        numpyargs = None
        fargs = casadiargs

    # Evaluate the function and return everything.
    fexpr = util.safevertcat(f(*fargs))
    return dict(fexpr=fexpr, casadiargs=casadiargs, numpyargs=numpyargs, XX=XX,
                names=varnames, sizes=realvarsizes)


def __casadi_to_numpy(x, matrix=False, scalar=False):
    """
    Converts casadi symbolic variable x to a numpy array of scalars.

    If matrix=False, the function will guess whether x is a vector and return
    the appropriate numpy type. To force a matrix, set matrix=True. To use
    a numpy scalar when x is scalar, use scalar=True.
    """
    shape = None
    if not matrix:
        if scalar and x.is_scalar():
            shape = ()
        elif x.is_vector():
            shape = (x.numel(),)
    if shape is None:
        shape = x.shape
    y = np.empty(shape, dtype=object)
    if y.ndim == 0:
        y[()] = x  # Casadi uses different behavior for x[()].
    else:
        for i in np.ndindex(shape):
            y[i] = x[i]
    return y


def __getargnames(func):
    """
    Returns a list of named input arguments of a Casadi Function.

    For convenience, if func is None, an empty list is returned.
    """
    argnames = []
    if func is not None:
        try:
            for i in range(func.n_in()):
                argnames.append(func.name_in(i))
        except AttributeError:
            if not isinstance(func, cs.Function):
                raise TypeError("func must be a casadi.Function!")
            else:
                raise
    return argnames


class DummySimulator(object):
    """
    Wrapper class to simulate a generic discrete-time function.
    """

    @property
    def Nargs(self):
        return self.__Nargs

    @property
    def args(self):
        return self.__argnames

    def __init__(self, model, argsizes, argnames=None):
        """Initilize the simulator using a model function."""
        # Decide argument names.
        if argnames is None:
            argnames = ["x"] + ["p_%d" % (i,) for i in range(1, self.Nargs)]

        # Store names and model.
        self.__argnames = argnames
        self.__integrator = model
        self.__argsizes = argsizes
        self.__Nargs = len(argsizes)

    def call(self, *args):
        """
        Simulates one timestep and returns a vector.
        """
        self._checkargs(args)
        return self.__integrator(*args)

    def _checkargs(self, args):
        """Checks that the right number of arguments have been given."""
        if len(args) != self.Nargs:
            raise ValueError("Wrong number of arguments: %d given; %d "
                             "expected." % (len(args), self.Nargs))

    def __call__(self, *args):
        """
        Interface to self.call.
        """
        return self.call(*args)

    def sim(self, *args):
        """
        Simulate one timestep and returns a Numpy array.
        """
        return np.array(self.call(*args)).flatten()


class DiscreteSimulator(DummySimulator):
    """
    Simulates one timestep of a continuous-time system.
    """

    @property
    def Delta(self):
        return self.__Delta

    def __init__(self, ode, Delta, argsizes, argnames=None, verbosity=1,
                 casaditype=None, numpy=None, scalar=None):
        """
        Initialize by specifying model and sizes of everything.

        See getCasadiIntegrator for description of arguments.
        """
        # Call subclass constructor.
        super(DiscreteSimulator, self).__init__(None, argsizes, argnames)

        # Store names and Casadi Integrator object.
        self.__Delta = Delta
        self.verbosity = verbosity
        self.__integrator = getCasadiIntegrator(ode, Delta, argsizes, argnames,
                                                wrap=False, scalar=scalar,
                                                casaditype=casaditype,
                                                verbosity=verbosity)

    def call(self, *args):
        """
        Simulates one timestep and returns a Casadi vector (DM, SX, or MX).

        Useful if you are using this object to construct a new symbolic
        function. If you are just simulating with numeric values, see self.sim.
        """
        # Check arguments.
        self._checkargs(args)
        integratorargs = dict(x0=args[0])
        if len(args) > 1:
            integratorargs["p"] = util.safevertcat(args[1:])

        # Call integrator.
        nextstep = self.__integrator(**integratorargs)
        xf = nextstep["xf"]
        return xf

