'''
method_name = ['gcn', 'sgc', 'gat', 'jknet', 'appnp', 'gprgnn',
'prognn', 'idgl', 'grcn', 'gaug', 'slaps', 'gen', 'gt', 'nodeformer', 'cogsl', 'sublime', 'stable', 'segsl']
'''
from opengsl.method.solver import Solver
from opengsl.method.gnnsolver import GCNSolver
from opengsl.method.gnnsolver import SGCSolver
from opengsl.method.gnnsolver import GATSolver
from opengsl.method.gnnsolver import JKNetSolver
from opengsl.method.gnnsolver import APPNPSolver
from opengsl.method.gnnsolver import GPRGNNSolver
from opengsl.method.gnnsolver import LINKSolver
from opengsl.method.gnnsolver import LPASolver
from opengsl.method.gnnsolver import GINSolver
from opengsl.method.solvers.GLCNSolver import GLCNSolver
from opengsl.method.solvers.WSGNNSolver import WSGNNSolver
from opengsl.method.solvers.PROGNNSolver import PROGNNSolver
from opengsl.method.solvers.IDGLSolver import IDGLSolver
from opengsl.method.solvers.GRCNSolver import GRCNSolver
from opengsl.method.solvers.GAUGSolver import GAUGSolver
from opengsl.method.solvers.SLAPSSolver import SLAPSSolver
from opengsl.method.solvers.GENSolver import GENSolver
from opengsl.method.solvers.GTSolver import GTSolver
from opengsl.method.solvers.NODEFORMERSolver import NODEFORMERSolver
from opengsl.method.solvers.COGSLSolver import COGSLSolver
from opengsl.method.solvers.SUBLIMESolver import SUBLIMESolver
from opengsl.method.solvers.STABLESolver import STABLESolver
from opengsl.method.solvers.SEGSLSolver import SEGSLSolver