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
from opengsl.method.gslsolver import WSGNNSolver
from opengsl.method.gslsolver import PROGNNSolver
from opengsl.method.gslsolver import IDGLSolver
from opengsl.method.gslsolver import GRCNSolver
from opengsl.method.gslsolver import GAUGSolver
from opengsl.method.gslsolver import SLAPSSolver
from opengsl.method.gslsolver import GENSolver
from opengsl.method.gslsolver import GTSolver
from opengsl.method.gslsolver import NODEFORMERSolver
from opengsl.method.gslsolver import COGSLSolver
from opengsl.method.gslsolver import SUBLIMESolver
from opengsl.method.gslsolver import STABLESolver
from opengsl.method.gslsolver import SEGSLSolver