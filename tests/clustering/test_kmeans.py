import pytest
import jax
import jax.numpy as jnp
import jax_healpy as jhp
from numpy.testing import assert_array_equal
import numpy as np
from jax.errors import TracerBoolConversionError
import chex
import healpy as hp
import matplotlib.pyplot as plt 

@pytest.fixture(scope="module", params=["FULL_MAP" , "GAL020" , "GAL040" , "GAL060"])
def mask(request , data_path , nside):
    if request.param == "FULL_MAP":
        return request.param , jnp.ones(jhp.nside2npix(nside))
    else:
        return request.param , np.load(f"{data_path}/GAL_PlanckMasks_64.npz")[request.param]

@pytest.fixture(scope="module", params=[64])
def nside(request):
    return request.param


def test_kmeans(mask):

    name , mask = mask

    indices , = jnp.where(mask == 1)

    n_regions = 10
    key = jax.random.key(0)

    clustered = jhp.get_clusters(mask, indices, n_regions, key)
    print(f"Got {n_regions} regions for mask {name}")

    cutout = jhp.get_cutout_from_mask(clustered, indices)
    # Shape must be the same as the mask
    assert cutout.shape == indices.shape

    labels , counts = jnp.unique(cutout, return_counts=True)

    # Check that all the regions are present
    assert_array_equal(labels , jnp.arange(n_regions))

    # Check that number of pixels in each region is close
    assert (counts.std() / counts.mean()) < 0.5

    print(f"all good for mask {name}")



def test_kmeans_jit(mask):

    name , mask = mask

    indices , = jnp.where(mask == 1)

    n_regions = 10
    key = jax.random.key(0)

    # number of regions cannot be a tracer if max_centroids is None
    jitted_clusters = jax.jit(jhp.get_clusters , static_argnums=(4 , 5))
    with pytest.raises(TracerBoolConversionError):
        jitted_clusters(mask, indices, n_regions, key, max_centroids=None)
#
    #If max_centroids is not None, n_regions can be a tracer and it is jitted once
    @jax.jit
    @chex.assert_max_traces(n=1)
    def jit_clusters(mask, indices, n_regions):
        return jhp.get_clusters(mask, indices, n_regions, key, max_centroids=10)
    
    clustered = jit_clusters(mask, indices, 5)
    clustered = jit_clusters(mask, indices, 10)

    chex.clear_trace_counter()

    # If requsted number of regions is greater than max_centroids, raise a runtime error

    with pytest.raises(RuntimeError):
        jit_clusters(mask, indices, 20)
    

def test_cutout_and_reconstruct(mask , nside):

    name , mask = mask

    indices , = jnp.where(mask == 1)
    inv_indices , = jnp.where(mask != 1)

    gaussian_map = jax.random.normal(jax.random.key(0) , jhp.nside2npix(nside))
    # set to unseen everything outside the mask
    gaussian_map = gaussian_map.at[inv_indices].set(jhp.UNSEEN)

    cutout = jhp.get_cutout_from_mask(gaussian_map, indices)

    assert cutout.shape == indices.shape

    reconstruct = jhp.from_cutout_to_fullmap(cutout, indices, nside)

    assert_array_equal(reconstruct, gaussian_map)


def test_frequency_map_cutout(mask ,nside):

    # This is usually done to get a cutout out of d the Frequency landscape object from furax

    name , mask = mask

    indices , = jnp.where(mask == 1)

    frequency_maps = jax.random.normal(jax.random.key(0) ,(10 ,  jhp.nside2npix(nside)))
    # set to unseen everything outside the mask
    frequency_maps = frequency_maps.at[...,~indices].set(jhp.UNSEEN)

    cutout = jhp.get_cutout_from_mask(frequency_maps, indices , axis=1)

    assert cutout.shape == (10 , *indices.shape)
