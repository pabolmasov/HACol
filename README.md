# HACol

Code calculates 1-D time-dependent structure of the accretion flow onto a neutron star with dipole magnetic field. Equations apply to the regime when radiation shock is expected to form.

## Installation and Running

<details>
<summary><strong>Installation and dependencies</strong></summary>

To run the code, make sure you have:

* Python 3 (tested with Python 3.6 and 3.7)
* MPI
* `mpi4py`
* `numpy`
* `scipy`
* preferably IPython
* HDF5 support, if HDF5 output is required

The main configuration file is `globals.conf`. In most cases, you should not need to modify the code itself; physical and numerical parameters can be adjusted through the configuration file.

`globals.conf` contains different configuration sets, which may differ in physical and numerical parameters.

</details>

<details>
<summary><strong>Running the code</strong></summary>

The main working file is `tire_MPI.py`, and its main routine is `alltire()`.

Tune the required global parameters in `globals.conf` and run the code from the command line:

```bash
mpirun -np <N> python tire_MPI.py <CONF>
```

where:

* `<N>` is the number of MPI processes to use. It should coincide with the value of `parallelfactor` in `globals.conf`.
* `<CONF>` is the name of the configuration set, specified in square brackets in `globals.conf`.

If `<CONF>` is omitted, the default configuration from `globals.conf` is used.

For example:

```bash
mpirun -np 8 python tire_MPI.py my_configuration
```

</details>

<details>
<summary><strong>Output during a simulation</strong></summary>

If everything is working correctly, you should get substantial output on `stdout` and nothing on `stderr`.

The output directory (by default `out/`, or the directory specified by `outdir` in the `globals` configuration) should immediately start filling with simulation output, unless graphical output has been disabled with:

```text
ifplot = False
```

Typical output files include:

* `vtie*.png` — graphical snapshots
* `tireout*.dat` — ASCII simulation snapshots
* `flux.dat` — total flux (luminosity in units of (L_{\rm Edd}/4\pi))
* `totals.dat` — total mass, momentum, and energy

The main output is either:

* `tireout.hdf5`, if `ifhdf` is `True`, or
* a series of `tireout*.dat` snapshot files.

**Important:** different simulations should use different output directories. Set a different `outdir` for each configuration. This allows several simulations to be run simultaneously.

</details>

<details>
<summary><strong>Running without IPython</strong></summary>

The code can also be run without IPython. In this case, make sure that all required Python libraries, in particular `scipy`, are installed.

</details>

<details>
<summary><strong>Machines without matplotlib</strong></summary>

If you are running the code on a machine that does not support `matplotlib`, disable graphical output in the configuration:

```text
ifplot = False
```

</details>

## Numerical Algorithms

<details>
<summary><strong>Overview</strong></summary>

The code solves the conservation equations for:

* mass,
* momentum,
* energy,

taking into account several physical effects, including:

* radiation losses,
* photon diffusion,
* mass loss when the force-free condition is violated.

The solution is obtained using either the **HLLE** or **HLLC** Riemann solver.

The Riemann solvers are implemented in `solvers.py`, while the signal velocities are computed in `sigvel.py`.

A detailed description of the physical model and numerical techniques is given in:

[2023 MNRAS paper](https://ui.adsabs.harvard.edu/abs/2023MNRAS.tmp.1899A/abstract)

</details>

## Additional Physics

<details>
<summary><strong>Additional physical effects</strong></summary>

**TBD**

</details>

## Outputs

<details>
<summary><strong>ASCII and HDF5 output</strong></summary>

If the `ifhdf` flag is `False`, the code produces multiple structure snapshots:

```text
tireout*.dat
```

These files contain quantities such as density, velocity, energy density, etc. as functions of distance along the field line.

If `ifhdf` is `True`, all this information is written to a single HDF5 file:

```text
tireout.hdf5
```

ASCII output is still produced, but less frequently. The frequency is controlled by `ascalias`, which is set to `10` by default.

The total luminosity is always written to:

```text
flux.dat
```

as a function of time.

The geometry of the flow, including radius, polar angle, and geometrical cross-section, is written to:

```text
geo.dat
```

in the output directory.

</details>

<details>
<summary><strong>HDF5 file structure</strong></summary>

In the HDF5 output file, individual snapshots are stored as separate datasets:

```text
entry000000
entry000001
...
```

All global parameters are stored in the dataset:

```text
globals
```

The contents of `geo.dat` are also stored in the HDF5 file as:

```text
geometry
```

Thus, `tireout.hdf5` is self-consistent and contains both the simulation results and the information required to interpret them.

</details>

<details>
<summary><strong>Combining and analysing HDF5 files</strong></summary>

HDF5 outputs with the same set of global parameters and the same geometry can be combined using the `liststitch` routine defined in:

```text
hdfoutput.py
```

The file:

```text
postpro.py
```

contains a number of routines useful for analysing and visualizing simulation results.

</details>

## Restart

<details>
<summary><strong>Restarting a simulation</strong></summary>

The code supports restarting a simulation with either identical or different spatial resolution.

If the number of points in the radial mesh is identical, no interpolation is performed.

If the number of points is different, all quantities are remapped onto the new grid.

If the outer radius of the new simulation is larger than the outer radius of the restart file, the code will report an error.

</details>

<details>
<summary><strong>HDF5 file locking</strong></summary>

Reading from and writing to an HDF5 file is possible when the global locking flag is set to a negative value.

Alternatively, in Bash you can disable HDF5 file locking with:

```bash
export HDF5_USE_FILE_LOCKING=FALSE
```

</details>

## Possible Problems

<details>
<summary><strong>Reading output during a simulation</strong></summary>

The code is designed to make simulation results accessible as soon as possible.

All ASCII files and HDF5 datasets are flushed as soon as a snapshot is written. Therefore, you do not need to wait until the simulation finishes before reading and reducing the output.

However, simultaneous access to an HDF5 file while it is being written may cause problems.

</details>

<details>
<summary><strong>Stopping a simulation</strong></summary>

If the simulation needs to be stopped before completion, the results written up to that point should be preserved.

Problems may nevertheless occur when an HDF5 file is read simultaneously with writing.

One possible solution is to disable HDF5 file locking before running the code:

```bash
export HDF5_USE_FILE_LOCKING=FALSE
```

</details>

## Plotting Figures

<details>
<summary><strong>Generating plots</strong></summary>

To generate plots from simulation results, use the plotting script:

```bash
ploth [DIRNAME]
```

where `[DIRNAME]` should match the `MODELNAME` specified in `globals.conf`.

For example:

```bash
ploth out/my_model
```

</details>

<details>
<summary><strong>Plot output</strong></summary>

The plotting script generates figures in both **PDF** and **PNG** formats. The figures are saved in the specified `DIRNAME` folder.

Three additional subdirectories are created to organize the plots:

```text
early/
late/
radial_times/
```

They contain:

* `early/` — plots from early simulation times
* `late/` — plots from late simulation times
* `radial_times/` — plots showing radial evolution at different times

These plots are intended to visualize the temporal and spatial evolution of the simulation results.

</details>
