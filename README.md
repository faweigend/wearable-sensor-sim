# Wearable Sensor Simulation

This code is linked to our open dataset at [LINK] and our publication
```
BIBTEX REFERENCE
```

## Usage Notes

This code operates directly on the CSV files in Repository 1 of our dataset: it reads each `body_segments.csv` file and
writes a corresponding `simulated_imus.csv` and `simulated_insoles.csv` file alongside it. When utilizing the synthetic sensor outputs, please keep in mind the fundamental differences between these simulated streams and real-world hardware measurements. 

Virtual IMU signals are calculated directly from rigid-body segment trajectories, yielding ideal kinematic data that do not model soft-tissue artifacts or cumulative gyroscope drift. 

Similarly, physical insole hardware typically measures scalar normal force distributions rather than 3D forces, and the vertical center of pressure coordinate in the local calcaneus frame represents a planar transformation artifact included strictly for debugging. For downstream estimation tasks targeting physical insole hardware, we recommend to simplify these signals to normal or vertical GRF magnitudes and 2D planar center of pressure measurements without their vertical component. 


## Output reference frames

Simulated IMUs are calculated from local body segment coordinate frames and the global-to-local rotations. Simulated insoles report GRFs and CoPs in respective foot frames.

![coordinate frames](https://raw.githubusercontent.com/faweigend/wearable-sensor-sim/refs/heads/main/httpdocs/coordinate-frames.png?token=GHSAT0AAAAAAEEUJWTWKCRYOLC2HFVZ5QJ42UVZ5IQ)


