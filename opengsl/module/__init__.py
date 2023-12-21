'''
method_name = ['gcn', 'sgc', 'gat', 'jknet', 'appnp', 'gprgnn',
'prognn', 'idgl', 'grcn', 'gaug', 'slaps', 'gen', 'gt', 'nodeformer', 'cogsl', 'sublime', 'stable', 'segsl']
'''
from opengsl.module.solver.solver import Solver
from opengsl.module.solver.gnnsolver import GCNSolver
from opengsl.module.solver.gnnsolver import SGCSolver
from opengsl.module.solver.gnnsolver import GATSolver
from opengsl.module.solver.gnnsolver import JKNetSolver
from opengsl.module.solver.gnnsolver import APPNPSolver
from opengsl.module.solver.gnnsolver import GPRGNNSolver
from opengsl.module.solver.gnnsolver import LINKSolver
from opengsl.module.solver.gnnsolver import LPASolver
from opengsl.module.solver.gnnsolver import GINSolver
from opengsl.module.solver.GLCNSolver import GLCNSolver
from opengsl.module.solver.WSGNNSolver import WSGNNSolver
from opengsl.module.solver.PROGNNSolver import PROGNNSolver
from opengsl.module.solver.IDGLSolver import IDGLSolver
from opengsl.module.solver.GRCNSolver import GRCNSolver
from opengsl.module.solver.GAUGSolver import GAUGSolver
from opengsl.module.solver.SLAPSSolver import SLAPSSolver
from opengsl.module.solver.GENSolver import GENSolver
from opengsl.module.solver.GTSolver import GTSolver
from opengsl.module.solver.NODEFORMERSolver import NODEFORMERSolver
from opengsl.module.solver.COGSLSolver import COGSLSolver
from opengsl.module.solver.SUBLIMESolver import SUBLIMESolver
from opengsl.module.solver.STABLESolver import STABLESolver
from opengsl.module.solver.SEGSLSolver import SEGSLSolver
from opengsl.module.solver.BMGCNSolver import BMGCNSolver
from opengsl.module.solver.PROSESolver import PROSESolver
from opengsl.module.solver.PASTELSolver import PASTELSolver
from opengsl.module.graphlearner import GraphLearner