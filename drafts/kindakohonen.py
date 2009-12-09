""" A little script exploring the possibility of building
Kohonen-like networks in a modular fashion, from PyBrain components. 
"""

from pybrain.structure.networks.recurrent import RecurrentNetwork
from pybrain.structure.modules import LinearLayer, SigmoidLayer, SoftmaxLayer, MultiplicationLayer
from pybrain.structure.connections import IdentityConnection, FullConnection
from pybrain.structure.connections.prototype import PrototypeConnection


__author__ = 'Tom Schaul, tom@idsia.ch'


def buildKN1(indim, hiddendim):
    N = RecurrentNetwork()
    inmod = LinearLayer(indim, 'in')
    hmod = SigmoidLayer(hiddendim, 'h')
    multmod = MultiplicationLayer(hiddendim, 'mult')
    outmod = SoftmaxLayer(hiddendim, 'out')
    N.addInputModule(inmod)
    N.addOutputModule(outmod)
    N.addModule(hmod)
    N.addModule(multmod)
    # note: multmod and hmod could be collapse into one by using a GaetLayer instead.
    N.addConnection(PrototypeConnection(inmod, multmod, outSliceTo=hiddendim))
    N.addConnection(IdentityConnection(hmod, outmod))
    N.addConnection(IdentityConnection(multmod, hmod))
    N.addRecurrentConnection(FullConnection(outmod, multmod, outSliceFrom=hiddendim))
    N.sortModules()
    return N
    
n = buildKN1(2, 3)
print n
print n.activate([2,3])
print n.activate([2,3])
print n.activate([2,3])
