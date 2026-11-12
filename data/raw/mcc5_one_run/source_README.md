---
pretty_name: Multi-mode Fault Diagnosis Datasets of Three-phase Asynchronous Motor Under Variable Working Conditions
language:
- en
license: mit
task_categories:
- time-series
- classification
- anomaly-detection
tags:
- fault-diagnosis
- condition-monitoring
- rotating-machinery
- induction-motor
- vibration
- motor-current
- variable-operating-conditions
- compound-faults
- time-series
- domain-adaptation
- test-time-adaptation
size_categories:
- 100M<n<1B
---

# Multi-mode Fault Diagnosis Datasets of Three-phase Asynchronous Motor Under Variable Working Conditions

<img width="666" alt="截屏2026-01-06 12 36 22" src="https://github.com/user-attachments/assets/8901f731-e80a-411f-9734-550b543e5b60" />

## Dataset Summary
This dataset provides synchronized multi-modal time-series data collected from a **2.2 kW three-phase asynchronous (induction) motor** operating under **variable speed/load conditions** with **deliberately induced faults**. It is intended for developing and benchmarking robust fault diagnosis methods under realistic operating scenarios, especially for **time-varying (transitional) working conditions**.

Key characteristics:
- Covers **electrical faults**, **mechanical faults**, and **electromechanical compound faults**
- Includes **two severity levels** for representative fault categories
- Provides **8-channel synchronous measurements** combining vibration and current signals for multi-modal diagnosis

## Data Availability
- **Mendeley Data:** https://data.mendeley.com/datasets/6s3dggj9mw/1
- **IEEE DataPort:** https://ieee-dataport.org/documents/multi-mode-fault-diagnosis-datasets-three-phase-asynchronous-motor-under-variable-working
- **Hugging Face:** https://huggingface.co/datasets/Samlzy/MCC5-THU-Motor


## Supported Tasks and Leaderboard-Style Use Cases
This dataset can support (but is not limited to) the following research tasks:
- **Multi-modal Fault Diagnosis (Vibration + Current Fusion)**
- **Compound Fault Diagnosis (Electromechanical Coupling)**
- **Fault Diagnosis with Different Fault Severity Degrees**
- **Fault Diagnosis with Multiple Steady Working Conditions**
- **Fault Diagnosis with Unknown / Unseen Working Conditions**
- **Fault Diagnosis under Variable Working Conditions**
- **Fault Diagnosis under Transitional Working Conditions (Time-varying Speed/Load Profiles)**

Typical problem formulations:
- **Multiclass classification** over fault types (and optionally severity)
- **Open-set / unknown condition generalization** (unseen speed/load profiles)
- **Online / test-time adaptation (OTTA)** under transitional profiles
- **Multi-modal fusion learning** using vibration + current

## Languages
- English (metadata, documentation)

## Dataset Structure

### Data Format
- **File format:** CSV
- **Total recordings:** 282 runs
- **Duration per run:** 90 seconds
- **Sampling frequency:** 12.8 kHz
- **Timestamp column:** not included (samples are uniformly sampled)

Each CSV file contains **8 columns** (synchronized signals). The key-phase signal is dimensionless and can be used to derive rotational speed.

### Data Fields (8 Columns)
Each CSV file contains the following 8 columns:

1. **speed**: motor key-phase signal (dimensionless; rotational speed can be derived)
2. **torque**: torque on the gearbox input shaft (Nm)
3. **motor_vibration_X**: vibration acceleration at motor drive end (horizontal radial direction, 0.1g)
4. **motor_vibration_Y**: vibration acceleration at motor drive end (axial direction, 0.1g)
5. **motor_vibration_Z**: vibration acceleration at motor drive end (vertical radial direction, 0.1g)
6. **motor_current_A**: phase-A current (0.1A)
7. **motor_current_B**: phase-B current (0.1A)
8. **motor_current_C**: phase-C current (0.1A)

### Variable Working Conditions
Two operating scenarios are included:

1) **Constant-speed, variable-torque:** motor speed is held constant (e.g., 1000 / 2000 / 3000 rpm), while torque changes over time.  
2) **Constant-torque, variable-speed:** load torque is held constant (e.g., 20 Nm / 40 Nm), while speed changes over time.

**Note:** Due to magnetic hysteresis effects in the motor and torque generator, the measured speed–torque trajectories may show slight deviations from the preset profiles.

### Fault Types (24 types; electrical, mechanical, and compound)
The dataset covers a wide range of motor faults, including (representative list):

#### Electrical faults
- **Stator winding inter-turn short circuit** (two severity levels, e.g., ~5% vs. ~10% of rated phase current equivalence)
- **Voltage unbalance** (two severity levels, e.g., ~4% vs. ~8%)
- **Broken rotor bars** (e.g., removal of consecutive rotor bars with rebalancing)

#### Mechanical faults
- **Rotor unbalance** (added imbalance mass)
- **Bent shaft** (permanent shaft bend)
- **Eccentricity** (including static eccentricity with two radial-offset severities, e.g., 0.125 mm and 0.250 mm)
- **Bearing faults (SKF 6205 deep-groove ball bearing)**
  - inner raceway defect (light / high)
  - outer raceway defect (light / high)
  - rolling element damage (ball defect)

#### Electromechanical compound faults (examples)
To study coupled signatures and cross-modulation, compound-fault scenarios are included, such as:
- bearing defect + static eccentricity
- bearing defect + rotor unbalance
- bearing defect + broken rotor bars
- bearing defect + winding short circuit

### File Naming Convention (Example)
Filenames encode fault type, severity, operating mode, and key condition settings. For example:
- `Bearing_inner_L_speed_circulation_20Nm_1000rpm`  
  indicates an inner-race bearing defect (light severity) under a variable-speed profile, with 20 Nm torque and the corresponding speed-time profile tag.
- `Bearing_inner_H_torque_circulation_20Nm_1000rpm`  
  indicates an inner-race bearing defect (high severity) under a variable-torque profile, with 1000 rpm and the corresponding torque-time profile tag.

(Exact naming patterns may vary slightly across fault categories; see the dataset directory structure for full coverage.)

## Experimental Setup
The test rig includes:
- 2.2 kW three-phase asynchronous motor
- Torque sensor (e.g., S2001; ±0.5% F.S. accuracy)
- Two-stage parallel gearbox (used in the rig configuration)
- Magnetic powder brake (as the load generator)
- Multi-channel data acquisition system (synchronous sampling at 12.8 kHz)

Sensors and measurements:
- Triaxial vibration acceleration sensor mounted on the motor drive end
- Three-phase current clamps for phase currents
- Key-phase sensor for key-phase signal (dimensionless)
- Laboratory temperature controlled within a small range (e.g., ±2°C) to reduce experimental variance.

Faults were physically introduced (e.g., precision machining / laser etching with tight tolerance) to ensure controllable and repeatable fault conditions.

## Citation
If you use this dataset, please cite:
```
@article{Chen2026MotorDataset,
  title   = {Multi-mode Fault Diagnosis Datasets of Three-phase Asynchronous Motor Under Variable Working Conditions},
  author  = {Shijin Chen and Zeyi Liu and Chenyang Li and Dongliang Zou and Xiao He and Donghua Zhou},
  journal = {arXiv preprint arXiv:2601.02278},
  year    = {2026}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
We extend our sincere gratitude to the THUFDD Group, led by Prof. Xiao He and Prof. Donghua Zhou, for their invaluable support and contributions to the development of this scheme.

We express our gratitude to the MCC5 Group Shanghai Co. LTD and Zhengzhou University for their invaluable support.
