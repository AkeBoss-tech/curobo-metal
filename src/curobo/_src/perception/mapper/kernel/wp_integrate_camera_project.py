class CameraProjectIntegrator:
    def __init__(self, integrator=None, *args, **kwargs):
        self.integrator = integrator

    def integrate(self, observation, *args, **kwargs):
        if self.integrator is None or not hasattr(self.integrator, "integrate"):
            raise NotImplementedError(
                "construct with a portable BlockSparseTSDFIntegrator"
            )
        return self.integrator.integrate(observation, *args, **kwargs)

    __call__ = integrate
