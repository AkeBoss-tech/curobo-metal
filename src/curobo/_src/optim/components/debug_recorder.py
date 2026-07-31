class DebugRecorder:
    def __init__(self, *args, **kwargs): self.clear()
    def record(self, iteration_state, action_horizon, action_dim):
        del action_horizon, action_dim
        self.trace.append(iteration_state.clone())
    def clear(self): self.trace=[]
    def get_trace(self): return self.trace
