import flax.linen as nn
from jax import random,vmap
from jax import numpy as jnp
import pprint

class net(nn.Module):
    dx: int
    @nn.compact
    def __call__(self,s):
        f = nn.vmap(
            nn.Dense,
            in_axes=1, out_axes=1,
            variable_axes={'params': 0},
            split_rngs={'params': True}
        )
        x = f(self.dx,use_bias=False)(s)
        return x
if __name__ == '__main__':
    seed = 123
    key = random.PRNGKey( seed )
    key,subkey = random.split( key )

    # build input
    outer_batches = 4
    s_observations = 5 # AKA the inner batch
    x_features = 2
    s_features = 3
    s_shape = (outer_batches,s_observations, s_features)
    s = random.uniform( subkey, s_shape )

    # instantiate the model and apply
    model = net(x_features)
    p = model.init( subkey, s )
    x = model.apply( p, s )

    # validate results
    pkernel = p['params']['VmapDense_0']['kernel']
    x_=jnp.zeros((outer_batches,s_observations,x_features))
    g = vmap(vmap(lambda a,b: a@b),in_axes=(0,None))
    x_=g(s,pkernel)
    print('s shape:',s.shape)
    print('p shape:',pkernel.shape)
    print('x shape:',x.shape)
    print('x_ shape:',x_.shape)
    print('sum of difference:',jnp.sum(x-x_)) # should be 0.0
