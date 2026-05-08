

# What positions do the muons map to for the simulated diamond?
import os
import jax
import itertools
import numpy as np
import matplotlib.pyplot as plt

from utils.min_distance import min_image_distance_triclinic, Lattice


base_path = r"C:\Users\Parv\Doc\RA\Projects\ferminet_versions\ferminet_af"

added_path = "positions\silicon\positions"

a = 10.26 # 10.26 - Si, 6.74 - C
n_particles = 65 # 65 - Si, 97 - C
lat = np.array([
        [a, a, 0],
        [0, a, a],
        [a, 0, a]
    ])


def load_position_data(n):

    data_path = os.path.join(base_path, added_path)

    n_files = len(os.listdir(data_path)[:n])

    positions_arr = []
    for i, file in enumerate(os.listdir(data_path)):
        k = np.load(os.path.join(data_path, file))
        positions_arr.append(k)

        if i >= (n_files - 1):
            break

    position_data = np.stack(positions_arr)
    position_data = position_data.reshape(512 * n_files, n_particles, 3)
    return position_data

def map_to_lattice(pos, lat):
    lat_inv = np.linalg.inv(lat)
    return np.einsum('...ij,kj->...ik', pos, lat_inv)

def position_histogram(muon_pos):
    hist, bins = np.histogram(muon_pos, bins=50)
    plt.plot(bins[:-1], hist)
    plt.show()

def get_tetrahedral_sites(supercell=True):

    base_sites = np.array([
        [1/4, 1/4, 3/4],
        [1/4, 3/4, 1/4],
        [3/4, 1/4, 1/4],
        [3/4, 3/4, 3/4],
    ])

    if supercell:
        translations = np.array(list(itertools.product([0,1], repeat=3)))

        sites = []
        for t in translations:
            for s in base_sites:
                sites.append((s + t) / 2)

        sites = np.array(sites)
        return sites
    return base_sites

def get_t_site_distance(t_site, muon_pos):
    # t_site is (1,3), muon_pos is (k,3)
    r_ij = muon_pos - t_site
    _, rabs = min_image_distance_triclinic(r_ij, Lattice(lat))
    return rabs


n = 100
pos = load_position_data(n=n)
muon_pos = pos[:, -1, :].reshape(n*512, 3)

t_sites_supercell = get_tetrahedral_sites(supercell=True) @ lat
t_sites = get_tetrahedral_sites(supercell=False) @ lat


t_site_distance = jax.vmap(get_t_site_distance, in_axes=(0, None))

all_dists_small = t_site_distance(t_sites, muon_pos)

all_dists_super = t_site_distance(t_sites_supercell, muon_pos)

t_distances = np.min(all_dists_small, axis=1) # n_t_sites, batch

t_distances_supercell = np.min(all_dists_super, axis=1)


# nearest T site for each sample
closest_small = np.min(all_dists_small, axis=0)

closest_super = np.min(all_dists_super, axis=0)

#plt.hist(closest_small, bins=50, alpha=0.6, label='primitive cell')
plt.hist(closest_super, bins=50, alpha=0.6, label='2x2x2 supercell')

plt.xlabel("Distance to nearest T site (bohr)")
plt.ylabel("Count")
plt.legend()
plt.show()

print("\nSupercell:")
print("mean =", np.mean(closest_super))
print("std  =", np.std(closest_super))

nearest_site_idx = np.argmin(all_dists_super, axis=0)

unique, counts = np.unique(nearest_site_idx, return_counts=True)

fractions = counts / counts.sum()

for u, c, f in zip(unique, counts, fractions):
    print(f"Site {u:2d}: {c:6d} samples  ({100*f:.2f}%)")
    plt.bar(unique, fractions)

plt.xlabel("Tetrahedral site index")
plt.ylabel("Fraction of muon samples")
plt.title("Nearest T-site occupancy")

plt.show()

localized = closest_super < 3.0
print(np.mean(localized))