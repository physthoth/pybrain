__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy.linalg import norm
from scipy import reshape, tile

from full import FullConnection

        
class PrototypeConnection(FullConnection):
    """ Assuming that each output unit corresponds to one prototype,
    defined by a vector within the connection's weight matrix,
    a PrototypeConnection computes the distance of the input to 
    each of the prototypes, as measured by the Euclidean norm.
    """
    
    
    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = map(norm, (reshape(self.params, (self.outdim, self.indim)) - tile(inbuf, (self.outdim, 1))))
    
    def _backwardImplementation(self, outerr, inerr, inbuf):
        raise NotImplementedError()