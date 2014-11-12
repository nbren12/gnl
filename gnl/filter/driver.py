from joblib import Parallel, delayed

class FilterDriver(object):

    def __init__(self, ensemble, evolver, analyzer):
        """
        Args:
            ensemble (2d array): (num state, num ensemble members)

            evolver: a callable that propagates each ensemble member with
                prototype `prior_member = evolver(ensemble_member, tcur, tout)`

            analyzer: a callable that analyzes each ensemble member given
                observations. It should have prototype:

                    analysis = analyzer(ensemble, obs)


        """
        self._ensemble = ensemble
        self._evolver= evolver
        self._analyzer = analyzer

        self._num_ensemble = ensemble.shape[1]

    def predict(self, tcur, tout, n_jobs=2):
        """Evolve ensemble members from tcur to tout."""

        ne = self._num_ensemble
        vecs = Parallel(n_jobs=n_jobs)(delayed(self._evolver)(self._ensemble[:,i],tcur, tout) 
                                  for i in range(ne))
        
        for i, vec in enumerate(vecs):
            self._ensemble[:,i] = vec

    def analyze(self, obs):
        """Analysis step"""
        self._ensemble = self._analyzer(self._ensemble, obs)

    
    def iter(self, tout, obs):
        """Iterator for predict analyze loop"""

        nt = len(tout)
        ob = next(obs)

        # Assimilate Observations

        self.analyze(ob)
        yield self._ensemble

        for i in range(1,nt):
            print('Iteration {i} of {n}'.format(i=i, n=nt), end='\r')

            # Prediction
            self.predict(tout[i-1], tout[i])

            ob = next(obs)

            # Analysis
            self.analyze(ob)

            yield self._ensemble
