class FilterDriver(object):

    def __init__(self, ensemble, evolver, analyzer):
        """
        Args:
            ensemble (2d array): shape is (num state, num ensemble members)

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

    def predict(tcur, tout):
        """Evolve ensemble members from tcur to tout."""

        for i in range(self._num_ensemble):
            cur_member = self._ensemble[:,i]
            cur_member[:] = self._evolver(cur_member, tcur, tout)

    def analyze(obs):
        """Analysis step"""
        self._ensemble = self._analyzer(self._ensemble, obs)