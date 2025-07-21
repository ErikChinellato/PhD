import numpy as np

def lorenz63_batch(states, t, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x = states[:, 0]
    y = states[:, 1]
    z = states[:, 2]

    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z

    return np.stack((dxdt, dydt, dzdt), axis=1)

def predict_rk4(particles, t, dt, deriv_func):
    k1 = deriv_func(particles, t)
    k2 = deriv_func(particles + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = deriv_func(particles + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = deriv_func(particles + dt * k3, t + dt)

    return t+dt, particles + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


particles = np.array([
    [1.0, 1.0, 1.0],
    [0.5, 0.5, 0.5],
    [2.0, 1.0, 0.0],
    [1.0, 2.0, 3.0],
    [5.0, 5.0, 5.0]
])

t = 0.0
dt = 0.01

next_particles = predict_rk4(particles, t, dt, lorenz63_batch)
print(next_particles)
