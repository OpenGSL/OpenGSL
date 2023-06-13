'''
method_name = ['gcn', 'sgc', 'gat', 'jknet', 'appnp', 'gprgnn',
'prognn', 'idgl', 'grcn', 'gaug', 'slaps', 'gen', 'gt', 'nodeformer', 'cogsl', 'sublime', 'stable', 'segsl']
'''
from .gnnsolver import GCNSolver as gcn
from .gnnsolver import SGCSolver as sgc
from .gnnsolver import GATSolver as gat
from .gnnsolver import JKNetSolver as jknet
from .gnnsolver import APPNPSolver as appnp
from .gnnsolver import GPRGNNSolver as gprgnn
from .gslsolver import PROGNNSolver as prognn
from .gslsolver import IDGLSolver as idgl
from .gslsolver import GRCNSolver as grcn
from .gslsolver import GAUGSolver as gaug
from .gslsolver import SLAPSSolver as slaps
from .gslsolver import GENSolver as gen
from .gslsolver import GTSolver as gt
from .gslsolver import NODEFORMERSolver as nodeformer
from .gslsolver import CoGSLSolver as cogsl
from .gslsolver import SUBLIMESolver as sublime
from .gslsolver import STABLESolver as stable
from .gslsolver import SEGSLSolver as segsl