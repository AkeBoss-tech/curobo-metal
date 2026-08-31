"""Tensor equivalents of small scalar helpers used by upstream Warp kernels."""

from curobo._src.util.warp import wp as _raw_wp


class _WarpCompat:
    """Minimal declaration-only Warp façade for CPU/MPS helper calls."""

    float32 = float

    @staticmethod
    def func(function):
        return function


# CUDA Warp modules are deliberately unavailable on the portable backend.  A
# small local decorator/type façade preserves the callable declaration without
# claiming a Warp kernel ABI.
wp = _raw_wp if _raw_wp is not None else _WarpCompat()


def _shrink_bounds_with_activation_distance_portable(lower, upper, activation_distance):
    return lower + activation_distance, upper - activation_distance


def _aggregate_bound_cost_portable(position, position_limit_lower, position_limit_upper, weight,
                         cost=None, gradient=None):
    below=max(position_limit_lower-position,0); above=max(position-position_limit_upper,0)
    value=0.5*weight*(below*below+above*above)
    grad=weight*(-below+above)
    return value,grad


def _aggregate_bound_cost_l1_portable(position, position_limit_lower, position_limit_upper, weight,
                            cost=None, gradient=None):
    below=max(position_limit_lower-position,0); above=max(position-position_limit_upper,0)
    return weight*(below+above), weight*((above>0)-(below>0))


def _aggregate_energy_regularization_portable(torque,velocity,dt,weight,cost=None,
                                    gradient_torque=None,gradient_velocity=None):
    return weight*torque*velocity*dt, weight*velocity*dt, weight*torque*dt


def _aggregate_squared_l2_regularization_portable(value,weight,cost=None,gradient=None):
    return 0.5*weight*value*value,weight*value


@wp.func
def shrink_bounds_with_activation_distance(
    lower: wp.float32,
    upper: wp.float32,
    activation_distance: wp.float32,
):
    return _shrink_bounds_with_activation_distance_portable(lower, upper, activation_distance)


@wp.func
def aggregate_bound_cost(
    position: wp.float32,
    position_limit_lower: wp.float32,
    position_limit_upper: wp.float32,
    weight: wp.float32,
    cost: wp.float32,
    gradient: wp.float32,
):
    return _aggregate_bound_cost_portable(
        position, position_limit_lower, position_limit_upper, weight, cost, gradient
    )


@wp.func
def aggregate_bound_cost_l1(
    position: wp.float32,
    position_limit_lower: wp.float32,
    position_limit_upper: wp.float32,
    weight: wp.float32,
    cost: wp.float32,
    gradient: wp.float32,
):
    return _aggregate_bound_cost_l1_portable(
        position, position_limit_lower, position_limit_upper, weight, cost, gradient
    )


@wp.func
def aggregate_squared_l2_regularization(
    value: wp.float32,
    weight: wp.float32,
    cost: wp.float32,
    gradient: wp.float32,
):
    return _aggregate_squared_l2_regularization_portable(value, weight, cost, gradient)


@wp.func
def aggregate_energy_regularization(
    torque: wp.float32,
    velocity: wp.float32,
    dt: wp.float32,
    weight: wp.float32,
    cost: wp.float32,
    gradient_torque: wp.float32,
    gradient_velocity: wp.float32,
):
    return _aggregate_energy_regularization_portable(
        torque, velocity, dt, weight, cost, gradient_torque, gradient_velocity
    )
__all__=["aggregate_bound_cost","aggregate_bound_cost_l1","aggregate_energy_regularization",
         "aggregate_squared_l2_regularization","shrink_bounds_with_activation_distance"]
