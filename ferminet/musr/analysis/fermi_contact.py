import os
import jax
import itertools
from tqdm import tqdm
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from fitting import fit_correlation_function, get_coupling_constant, plot_fitted_correlation_function
from utils.min_distance import min_image_distance_triclinic, Lattice



def spin_resolved_pair_density(
                pos, lat, a, n_bins, n_up_electrons=None, target_species=-1, coord_centre=None):
        # In PBC - D, B, N, 3 -> R, R - if per device then don't divide by device in hist

        r_max = 10  # Distance works up to half the size of the primitive vector length
        grids = jnp.linspace(0, r_max, n_bins + 1)
        dr = grids[1] - grids[0]
        bin_volume = 4 * jnp.pi / 3.0 * (grids[1:]**3 - grids[:-1]**3)
        r_search = 1
        D, B, N, d = pos.shape  # Device, Batch, Particles, dimension    
        
        if not n_up_electrons:
                n_up_electrons = N // 2 # Assuming n_up is directly divisible
        
        def rabs(x):
                """
                Np, 3 -> Np (vmap, pmap)
                """
                if target_species == 0:
                    rvec = x - coord_centre  # If we don't have a target species, we are doing classical around one Coord center position
                else:
                    rvec = x[:-1, :] - x[-1, :]
                _, rabs = min_image_distance_triclinic(rvec, Lattice(lat), r_search)

                return rabs

        rabs_p = jax.vmap(jax.vmap(rabs))  # Will be constants.pmap when over devices
        r = rabs_p(pos)
        
        def hist_per_device(r):
                """
                D, B, Np -> R
                """
                hist, _ = jnp.histogram(r, bins= n_bins, range=(0, r_max))
                hist = hist / bin_volume
                hist /= (B)
                #hist = constants.pmean(hist) - in actual observable
                return hist
        
        spin_up_hist = hist_per_device(r[:, :, :n_up_electrons])
        spin_down_hist = hist_per_device(r[:, :, n_up_electrons:])

        return spin_up_hist, spin_down_hist


def plot_correlation_function(popt):
        # Plotting correlation function
        def fit_func(r, a, c):
                b = -0.995187 * 2
                return a + b * r + c * r**2

        new_max_range = 0.5
        r_fine = jnp.linspace(0, new_max_range, 1000)
        y_fitted = fit_func(r_fine, *popt)
        plt.plot(r_fine, -jnp.exp(y_fitted), "-", label="Cusp-constrained fit", zorder=10)
        plt.xlabel("Separation (bohr)")
        plt.ylabel("g(r)")
        plt.title("Muon-Electron Spin Correlation Function g(r)")
        plt.ylim(-0.1, 0.1)
        plt.legend()
        plt.show()




a = 6.74  # 10.26 - Si, 6.74 - C
lat = np.array([
        [a, a, 0],
        [0, a, a],
        [a, 0, a]
    ])

D, B, N, dim = 6, 128, 65, 3
n_bins = 200
#pos = np.random.rand(D, B, N, dim)
#spin_up_hist, spin_down_hist = pair_density(pos, lat, a, n_bins)

run_inference = False
save_to_file = True
run_stats = False
t_site_stats = True

base_path_w = r'/mnt/c/Users/Parv/Doc/RA/Projects/ferminet_versions/ferminet_af/positions/diamond/positions'
base_path = r'C:\\Users\\Parv\\Doc\\RA\\Projects\\ferminet_versions\\ferminet_af\\positions\\diamond\\positions'

if run_inference:
        spin_up_hist = np.zeros(n_bins)
        spin_down_hist = np.zeros(n_bins)
        #pair_density = jax.jit(pair_density)
        n_files = len(os.listdir(base_path))
        #n_files = 1000

        for file in tqdm(os.listdir(base_path)[:n_files]):
                pos = np.load(os.path.join(base_path, file))
                D, B, _ = pos.shape
                pos = pos.reshape(D, B, -1, 3)
                up, down = spin_resolved_pair_density(pos, lat, a, n_bins)
                spin_up_hist += up
                spin_down_hist += down
                
        
        if save_to_file:
                with open('up_hist_.npy', 'wb') as f:
                        np.save(f, spin_up_hist)        
                
                with open('down_hist_.npy', 'wb') as f:
                        np.save(f, spin_down_hist)

        grids = np.linspace(0, 0.5 * a, n_bins)

        spin_diff_hist = (spin_up_hist - spin_down_hist) / n_files
        #plt.plot(np.linspace(0, 0.5 * a, n_bins), spin_up_hist)
        #plt.plot(np.linspace(0, 0.5 * a, n_bins), spin_down_hist)
        plt.plot(grids, spin_diff_hist)
        plt.show()
        y_at_r0, popt, errs = fit_correlation_function(spin_up_hist/n_files, spin_down_hist/n_files, grids)
        A_mhz = get_coupling_constant(y_at_r0)
        plot_correlation_function(popt=popt)


if run_stats:
        
        spin_up_hist = np.load('up_hist_si.npy')
        spin_down_hist = np.load('down_hist_si.npy')
        spin_diff_hist = (spin_up_hist - spin_down_hist) / 6217

        grids = np.linspace(0, 0.5 * a, n_bins)

        plt.plot(np.linspace(0, 0.5 * a, n_bins), spin_diff_hist)
        plt.show()

        y_at_r0, popt, errs = fit_correlation_function(spin_up_hist/6217, spin_down_hist/6217, grids)
        A_mhz = get_coupling_constant(y_at_r0)
        plot_correlation_function(popt=popt)


if t_site_stats == True:
        lat = lat / 2
        base_site = jnp.array([3/8, 3/8, 3/8])

        translations = jnp.array(
        list(itertools.product([0, 1], [0, 1], [0, 1]))
        )
        tetrahedral_sites_frac = (base_site + translations) / 2
        tetrahedral_sites_cart = jnp.matmul(
                tetrahedral_sites_frac,
                lat
                )

        def get_muon_localizing_t_site(pos):
                # Get muon location in lattice - map to lattice
                muon_pos = pos[-1]     # shape (3,)

                r_ij = muon_pos[None, :] - tetrahedral_sites_cart

                _, dist = min_image_distance_triclinic(
                        r_ij,
                        Lattice(lat),
                        radius=1
                )
                nearest_site = jnp.argmin(dist)

                histogram = jax.nn.one_hot(
                        nearest_site,
                        tetrahedral_sites_cart.shape[0]
                )
                muon_frac = jnp.matmul(
                        muon_pos,
                        jnp.linalg.inv(lat).T
                ) % 1

                return histogram, muon_frac, nearest_site, dist

        t_site = jax.vmap(
        jax.vmap(get_muon_localizing_t_site)
        )

        n_files = len(os.listdir(base_path))

        total_histogram = jnp.zeros(len(tetrahedral_sites_cart))

        for file in tqdm(os.listdir(base_path)[:n_files]):

                pos = np.load(os.path.join(base_path, file))
                D, B, _ = pos.shape
                pos = pos.reshape(D, B, -1, 3)
                histograms, muon_frac, nearest_site, dist = t_site(pos)
                total_histogram += histograms.sum(axis=(0,1))

                # --------------------------------------------------------------
                # Debug output
                # --------------------------------------------------------------

                print("Muon fractional coordinates:")
                print(muon_frac[0,:5])

                print("\nNearest tetrahedral sites:")
                print(nearest_site[0,:5])

                print("\nDistances:")
                print(dist[0,:5])

                print("\nHistogram:")
                print(total_histogram)

                break
                                