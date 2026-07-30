"""Tensor equivalents of small scalar helpers used by upstream Warp kernels."""
def shrink_bounds_with_activation_distance(lower, upper, activation_distance):
    return lower + activation_distance, upper - activation_distance
def aggregate_bound_cost(position, position_limit_lower, position_limit_upper, weight,
                         cost=None, gradient=None):
    below=max(position_limit_lower-position,0); above=max(position-position_limit_upper,0)
    value=0.5*weight*(below*below+above*above)
    grad=weight*(-below+above)
    return value,grad
def aggregate_bound_cost_l1(position, position_limit_lower, position_limit_upper, weight,
                            cost=None, gradient=None):
    below=max(position_limit_lower-position,0); above=max(position-position_limit_upper,0)
    return weight*(below+above), weight*((above>0)-(below>0))
def aggregate_energy_regularization(torque,velocity,dt,weight,cost=None,
                                    gradient_torque=None,gradient_velocity=None):
    return weight*torque*velocity*dt, weight*velocity*dt, weight*torque*dt
def aggregate_squared_l2_regularization(value,weight,cost=None,gradient=None):
    return 0.5*weight*value*value,weight*value
__all__=["aggregate_bound_cost","aggregate_bound_cost_l1","aggregate_energy_regularization",
         "aggregate_squared_l2_regularization","shrink_bounds_with_activation_distance"]
