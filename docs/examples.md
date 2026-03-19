# Examples

The `examples/` directory contains fourteen self-contained scripts that form a
progressive tutorial, from vacuum to a full universe simulation.  Run them in
order — each one builds on the concepts introduced by the previous.

## Running the examples

```bash
git clone https://github.com/gpartin/lfm-physics.git
cd lfm-physics
pip install -e ".[viz]"   # optional: matplotlib for plots
python examples/01_empty_space.py
```

---

## 01 — Empty Space

**File**: `examples/01_empty_space.py`

What does the vacuum look like?  In LFM, empty space is a lattice where every
point has stiffness $\chi = 19$.  This example creates a 32³ vacuum and
confirms it is completely stable.

**Key output:**

```
chi everywhere  = 19.0
After 500 steps: chi std = 0.000000
```

**Concept**: $\chi_0 = 19$ is derived from 3D lattice geometry:
$1 + 6 + 12 = 19$ (centre + face-neighbours + edge-neighbours).

---

## 02 — Your First Particle

**File**: `examples/02_first_particle.py`

Drop a Gaussian energy blob onto the lattice.  GOV-02 immediately pulls $\chi$
downward around it — a gravitational well emerges with no force law injected.

**Key output:**

```
Before equilibration: χ_min = 19.00
After equilibration:  χ_min = 14.xx  ← gravity well appeared
```

---

## 03 — Measuring Gravity

**File**: `examples/03_measuring_gravity.py`

Extract the radial $\chi(r)$ profile of a single soliton and verify that the
well depth falls off as $\Delta\chi \propto 1/r$ — exactly the Newtonian
potential shape, emerging from the lattice dynamics.

**Key output:**

```
Δχ(r=4) / Δχ(r=8) ≈ 2.0   (1/r behaviour verified)
```

---

## 04 — Two Bodies

**File**: `examples/04_two_bodies.py`

Place two solitons 14 cells apart and watch them attract.  Separation
decreases over time because each soliton curves toward the other's $\chi$-well.
No Newtonian force law, no equations of motion — just the two governing
equations.

---

## 05 — Electric Charge

**File**: `examples/05_electric_charge.py`

Switch to complex wave fields (`FieldLevel.COMPLEX`) and discover that the
**phase** of $\Psi$ acts as electric charge:

- $\theta = 0$ → "electron" (negative)
- $\theta = \pi$ → "positron" (positive)

Same-phase solitons undergo constructive interference → energy UP → **repel**.
Opposite-phase solitons undergo destructive interference → energy DOWN →
**attract**.  Coulomb's law emerges from wave interference.

---

## 06 — Dark Matter

**File**: `examples/06_dark_matter.py`

Create a deep $\chi$-well, then remove all matter ($\Psi = 0$).  The well
persists — $\chi$ is governed by its own wave equation (GOV-02) and does not
snap back instantly.  The empty well still attracts matter that comes near it.

**Concept**: Dark matter in LFM is substrate *memory* — the $\chi$ field
remembering where matter used to be.

---

## 07 — Matter Creation

**File**: `examples/07_matter_creation.py`

Drive $\chi$ at the parametric resonance frequency $\Omega = 2\chi_0 = 38$.
GOV-01 becomes a Mathieu equation with exponentially growing solutions.
Machine-epsilon noise in $\Psi$ amplifies by factors of $10^{20}$ or more.

**Concept**: In the early universe, $\chi$ oscillated violently after the Big
Bang, and this parametric resonance is what filled space with matter.

---

## 08 — Simulate a Universe

**File**: `examples/08_universe.py`

The capstone example.  Nine Poisson-equilibrated solitons on a 64³ grid under
GOV-01 + GOV-02 spontaneously form a cosmic web: $\chi$-wells (proto-galaxies)
separated by high-$\chi$ voids.  `CosmicScale` converts lattice ticks into
billions of years.

**Key output:**

```
Time         χ_min   Wells%   Voids%  Clusters  Energy
 0.0 Gyr     14.xx    xx.x%    xx.x%        9   x.xxe+03
 5.0 Gyr     12.xx    xx.x%    xx.x%       xx   ...
13.8 Gyr     11.xx    xx.x%    xx.x%       xx   ...
```

---

## 13 — Weak Force (Parity Asymmetry)

**File**: `examples/13_weak_force.py`

Isolate the `epsilon_w * j` term in GOV-02 by running the same setup with
`epsilon_w=0.1` and with `epsilon_w=0.0` (control).  Use
`lfm.momentum_density()` and `lfm.weak_parity_asymmetry()` to measure the
left/right chi-depression imbalance.

**Concept**: parity asymmetry emerges from a lattice current observable `j`,
not from an injected weak-force potential.

---

## 14 — Strong Force (Color Confinement Proxy)

**File**: `examples/14_strong_force.py`

Switch to `FieldLevel.COLOR` and measure a confinement proxy directly from chi:

```
I = integral (max(chi) - chi) ds
```

For tube-like color flux, this line integral grows approximately with source
separation. The script prints a linear fit and R^2 for the trend.

**Concept**: confinement can be quantified from the simulated substrate field
without injecting an external strong-potential formula.
