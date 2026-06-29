import jax.numpy as jnp

def angle_difference_wrap(x):
	"""map to [-pi, pi] range"""
	return (x + jnp.pi) % (2 * jnp.pi) - jnp.pi

lerp = lambda a, b, t: a + (b - a) * t

softplus_1 = jnp.log(jnp.exp(1) - 1)
softplus = lambda x: jnp.log(jnp.exp(x) + 1) #/ softplus_1
softplus_norm = lambda x: softplus(x) / softplus(0)
assert jnp.allclose(softplus_norm(0), 1)


class Rotor():
	def __init__(self, angle):
		self.sc = jnp.sin(angle), jnp.cos(angle)
	def forward(self, x, y):
		s, c = self.sc
		return c * x - s * y, s * x + c * y
	def backward(self, x, y):
		s, c = self.sc
		return c * x + s * y, -s * x + c * y