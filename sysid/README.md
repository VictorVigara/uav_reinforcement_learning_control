# Air65 system identification

How the numbers in `drones/air65/params.yaml` were obtained, and how to redo
them after a new flight.

## Pipeline

```bash
# 1. Export skybridge flight recordings (needs skybridge's interpreter, for rerun)
~/skybridge/.venv/bin/python sysid/extract_flight_logs.py

# 2. Decode the Betaflight blackbox logs   (pip install orangebox==0.4.0)
python sysid/decode_blackbox.py

# 3. Fit the model to the logs              -> sysid/air65_fit.json
python sysid/fit_air65.py

# 4. Replay the flights through the simulator; --tune searches the delays and
#    controller parameters that match best  -> sysid/plots/replay_*.png
python sysid/validate_replay.py [--tune]

# 5. Closed loop: skybridge's altitude-hold PID flown in the simulator
python sysid/validate_althold.py            # -> sysid/plots/althold.png
```

Steps 1 and 2 write small `.npz` files to `sysid/data/`, so steps 3 to 5 need
neither skybridge nor the raw logs. Fitted values are copied into
`drones/air65/params.yaml` by hand, each with a tag saying where it came from.

## Data

| Source | Content | Used for |
|---|---|---|
| `~/skybridge/data/live/*_flight.rrd`, 3 flights, 2026-11-13 | RC sent (50 Hz), camera position and velocity (30 Hz), attitude telemetry (17 Hz), altitude-hold state | throttle response, angle response, delays, drag, trim, closed-loop check |
| `~/skybridge/data/blackbox/*.bbl`, 14 short logs, 2026-09-20 | gyro, motor output, eRPM, battery at 1 kHz | motor rpm map, battery dependence, motor lag, inertia bounds |

## Results

| Quantity | Value | Source |
|---|---|---|
| Motor rpm | 75300 · (V / 3.7)^0.86 · (u + 0.047)^0.835, rmse 550 rpm, measured up to 43 % output | blackbox |
| Thrust vs throttle | ∝ (u + 0.047)^1.67 from rpm; 1.4 from the climb response. Model uses 1.5 | blackbox, flights |
| Thrust vs battery | ∝ V^1.7 | blackbox |
| Motor lag | 35 ms | blackbox |
| Hover throttle | 0.30 to 0.32 | flights |
| Vertical drag | 0.25 to 0.4 1/s | flights |
| Command to motion delay | 100 to 140 ms, of which 45 ms is the radio link | flights, replay |
| Angle response | level P 5 1/s plus 0.7 feedforward; 10 degree step reaches 63 % in 0.12 s | replay |
| Telemetry latency | 90 ms, one sample every 59 ms | replay, flights |
| Tracker velocity lag | 80 ms | replay |
| Level trim error | drifts as if tilted 0.9 to 2.8 degrees (median per flight), up to 3.6 | replay |

Validation, after fitting:

| Check | Result |
|---|---|
| Replay, attitude | 0.4 to 0.7 degrees rms on two flights, 1.5 on the third (signal spread 1.2 to 2.5) |
| Replay, horizontal velocity, trim removed | 0.03 to 0.06 m/s rms (spread 0.07 to 0.16) |
| Replay, vertical velocity | 0.04 to 0.06 m/s rms on the two piloted flights (spread 0.20 to 0.35) |
| Altitude hold, engage transient | segment 2: same overshoot and 10 s settling; height rms difference 36 mm, height spread 68 mm real vs 71 mm sim |

## What is not pinned down

- **Mass.** Assumed 28 g. Because thrust is tied to the measured hover throttle,
  accelerations hardly depend on it; weigh the drone and set `mass`.
- **Thrust above 45 % throttle.** No data: the curve is extrapolated and capped
  at 3.4 times hover thrust per motor. The policy's throttle stops at 0.56.
- **Rate loop.** The blackbox logs were flown with centred sticks, so the rate
  setpoint never moves. The rate loop time constants are estimates; the overall
  angle response is fitted.
- **Inertia.** Only bounded (roll 8e-6 to 3e-5, pitch 5e-6 to 1e-5 kg m^2).
- **Horizontal drag.** The flights stay under 1 m/s. 0.2 1/s, randomized 0.5x to 2x.
- **Heading.** These recordings predate skybridge's heading estimate, so the
  heading at each flight's start is fitted and gyro drift during the flight is
  not corrected. It limits how well horizontal velocity can match.
- **Altitude hold, segment 1.** The real drone rose at engage while the sim
  sank: the real hover throttle was below the stick at that moment and then
  drifted up with the battery. The sim uses one hover throttle per segment.

A blackbox log of a flight with stick activity, and a flight recorded with the
heading estimate, would close most of these.
