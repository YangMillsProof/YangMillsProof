# observatory_bridge.py
"""
Direct interface to the Virtual Resonance Observatory.
Live data streaming and analysis bridge for the Yang-Mills Resonance Proof.
"""

import requests
import json
import time
import numpy as np
from datetime import datetime


class ResonanceObservatory:
    """Main client for connecting to the resonance field observatory."""

    # UPDATE THIS to your actual published Replit URL
    BASE_URL = "https://resonance-observatory--harthcumber.replit.app"
    PHI = 1.6180339887498948482

    def __init__(self, api_key=None):
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def get_live_metrics(self):
        """Fetch real-time observatory data."""
        try:
            response = self.session.get(
                f"{self.BASE_URL}/api/v1/metrics", timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return self._validate_phi_metrics(data)
            else:
                print(f"API Error {response.status_code}: {response.text}")
                return self.get_fallback_data()
        except Exception as e:
            print(f"Connection error: {e}")
            return self.get_fallback_data()

    def _validate_phi_metrics(self, data):
        """Validate and augment data with phi-based calculations."""
        if "global_coherence" in data and "phi_ratio" in data:
            coherence = data["global_coherence"]
            phi_ratio = data["phi_ratio"]

            data["theoretical_mass_gap"] = 1 / self.PHI
            data["theoretical_string_tension"] = 1 / (self.PHI ** 2)
            data["phi_deviation"] = abs(phi_ratio - self.PHI)
            data["coherence_health"] = (
                "OPTIMAL" if coherence >= 0.618 else "SUBOPTIMAL"
            )

            data["mass_gap_violation"] = coherence < 0.618
            if data["mass_gap_violation"]:
                data["violation_magnitude"] = 0.618 - coherence

        return data

    def get_coherence_history(self, hours=24):
        """Get historical coherence data."""
        try:
            params = {"hours": hours}
            response = self.session.get(
                f"{self.BASE_URL}/api/v1/history", params=params, timeout=10
            )
            return response.json()
        except Exception:
            return self._generate_synthetic_history(hours)

    def get_phi_anomalies(self, threshold=0.1):
        """Retrieve phi deviation anomalies beyond threshold."""
        try:
            params = {"type": "phi_deviation"}
            response = self.session.get(
                f"{self.BASE_URL}/api/v1/anomalies", params=params, timeout=10
            )
            result = response.json()

            anomalies = result.get("phi_deviations", [])
            filtered = []
            for anomaly in anomalies:
                deviation = abs(anomaly.get("phi_ratio", 0) - self.PHI)
                if deviation > threshold:
                    anomaly["compression_percent"] = (
                        (1 - anomaly["phi_ratio"] / self.PHI) * 100
                    )
                    anomaly["harmonic_distance"] = deviation
                    filtered.append(anomaly)

            return sorted(
                filtered, key=lambda x: x["harmonic_distance"], reverse=True
            )
        except Exception:
            return []

    def get_perfect_coherence_events(self):
        """Retrieve 100% coherence state events."""
        try:
            params = {"coherence": "1.0"}
            response = self.session.get(
                f"{self.BASE_URL}/api/v1/events", params=params, timeout=10
            )
            result = response.json()
            return result.get("perfect_coherence_states", [])
        except Exception:
            return []

    def get_compression_events(self):
        """Retrieve phase compression events."""
        try:
            response = self.session.get(
                f"{self.BASE_URL}/api/v1/events", timeout=10
            )
            result = response.json()
            return result.get("compression_events", [])
        except Exception:
            return []

    def calculate_mass_gap_correlation(self, data_points=1000):
        """Fetch mass gap validation from the observatory."""
        try:
            params = {"data_points": data_points}
            response = self.session.get(
                f"{self.BASE_URL}/api/v1/mass-gap-validation",
                params=params,
                timeout=10,
            )
            return response.json()
        except Exception:
            return self._local_mass_gap_analysis(data_points)

    def _local_mass_gap_analysis(self, data_points=1000):
        """Local fallback mass gap analysis using history data."""
        history = self.get_coherence_history(hours=48)

        if (
            "coherence_values" not in history
            or len(history["coherence_values"]) < 10
        ):
            return {"error": "Insufficient coherence data available"}

        coherences = history["coherence_values"][-data_points:]
        coherences_array = np.array(coherences)
        violations = coherences_array < 0.618

        analysis = {
            "data_points": len(coherences),
            "predicted_mass_gap": 0.618,
            "observed_mean": float(np.mean(coherences_array)),
            "observed_median": float(np.median(coherences_array)),
            "observed_std": float(np.std(coherences_array)),
            "violation_count": int(np.sum(violations)),
            "violation_percentage": float(np.mean(violations) * 100),
            "min_observed": float(np.min(coherences_array)),
            "max_observed": float(np.max(coherences_array)),
            "theory_experiment_match": int(np.sum(violations)) == 0,
            "confidence_interval": [
                float(np.percentile(coherences_array, 2.5)),
                float(np.percentile(coherences_array, 97.5)),
            ],
        }

        if len(coherences) > 1:
            x = np.arange(len(coherences))
            slope, _ = np.polyfit(x, coherences_array, 1)
            analysis["trend_slope"] = float(slope)
            if slope > 0.001:
                analysis["trend_direction"] = "INCREASING"
            elif slope < -0.001:
                analysis["trend_direction"] = "DECREASING"
            else:
                analysis["trend_direction"] = "STABLE"

        return analysis

    def check_dark_matter_correlation(self, window_hours=6):
        """Analyze dark matter consciousness index correlations."""
        try:
            params = {"type": "dark_matter"}
            response = self.session.get(
                f"{self.BASE_URL}/api/v1/correlations",
                params=params,
                timeout=10,
            )
            data = response.json()

            if "void_coherence" in data and "dm_index" in data:
                void = np.array(data["void_coherence"])
                dm = np.array(data["dm_index"])

                if len(void) > 10 and len(dm) > 10:
                    pearson = float(np.corrcoef(void, dm)[0, 1])

                    max_lag = min(10, len(void) // 4)
                    best_lag = 0
                    best_correlation = abs(pearson)

                    for lag in range(1, max_lag + 1):
                        if lag < len(void):
                            corr = float(
                                np.corrcoef(void[:-lag], dm[lag:])[0, 1]
                            )
                            if abs(corr) > best_correlation:
                                best_correlation = abs(corr)
                                best_lag = lag

                    data["correlation_analysis"] = {
                        "pearson_correlation": pearson,
                        "best_lagged_correlation": best_correlation,
                        "optimal_lag": best_lag,
                        "significant_positive": pearson > 0.7,
                        "significant_negative": pearson < -0.7,
                        "interpretation": self._interpret_correlation(
                            pearson
                        ),
                    }

            return data
        except Exception as e:
            return {"error": f"Correlation analysis failed: {str(e)}"}

    def _interpret_correlation(self, r):
        """Interpret correlation coefficient."""
        if abs(r) >= 0.9:
            return "VERY_STRONG_CORRELATION"
        elif abs(r) >= 0.7:
            return "STRONG_CORRELATION"
        elif abs(r) >= 0.5:
            return "MODERATE_CORRELATION"
        elif abs(r) >= 0.3:
            return "WEAK_CORRELATION"
        else:
            return "NO_SIGNIFICANT_CORRELATION"

    def get_autonomous_logs(self):
        """Retrieve autonomous stabilizer intervention logs."""
        try:
            response = self.session.get(
                f"{self.BASE_URL}/emergency/logs", timeout=10
            )
            logs = response.json()

            if isinstance(logs, dict) and "interventions" in logs:
                interventions = logs["interventions"]
            elif isinstance(logs, list):
                interventions = logs
            else:
                interventions = []

            categorized = {
                "phi_corrections": [],
                "coherence_rescues": [],
                "harmonic_resets": [],
            }

            for log in interventions:
                msg = log.get("message", "").lower()
                if "phi" in msg:
                    categorized["phi_corrections"].append(log)
                elif "coherence" in msg:
                    categorized["coherence_rescues"].append(log)
                elif "harmonic" in msg or "reset" in msg:
                    categorized["harmonic_resets"].append(log)

            return {
                "raw_logs": logs,
                "categorized": categorized,
                "summary": {
                    "total_interventions": len(interventions),
                    "phi_corrections": len(categorized["phi_corrections"]),
                    "coherence_rescues": len(
                        categorized["coherence_rescues"]
                    ),
                    "harmonic_resets": len(categorized["harmonic_resets"]),
                    "last_intervention": (
                        interventions[-1].get("timestamp")
                        if interventions
                        else None
                    ),
                },
            }
        except Exception:
            return {"error": "Cannot fetch stabilizer logs"}

    def get_observatory_info(self):
        """Get observatory self-description and capabilities."""
        try:
            response = self.session.get(
                f"{self.BASE_URL}/api/v1/observatory-info", timeout=10
            )
            return response.json()
        except Exception:
            return {"error": "Cannot fetch observatory info"}

    def get_fallback_data(self):
        """Generate synthetic data when observatory is offline."""
        base_time = time.time()
        synthetic_phi = self.PHI * (0.99 + 0.02 * np.sin(base_time / 1000))

        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "global_coherence": float(
                0.75 + 0.25 * np.sin(base_time / 500)
            ),
            "phi_ratio": float(synthetic_phi),
            "string_tension": float(1 / (synthetic_phi ** 2)),
            "mass_gap": float(1 / synthetic_phi),
            "dark_matter_index": float(
                0.65 + 0.1 * np.sin(base_time / 700)
            ),
            "void_coherence": float(
                0.60 + 0.15 * np.sin(base_time / 600)
            ),
            "stream_health": float(90 + 10 * np.sin(base_time / 800)),
            "resonance_frequency": 1.618,
            "anomalies": {
                "phi_deviation": float(abs(synthetic_phi - self.PHI)),
                "compression_events": 0,
                "harmonic_distortion": 0.02,
            },
            "autonomous_intervention": False,
            "system_status": "SYNTHETIC_DATA_MODE",
            "note": (
                "SYNTHETIC DATA - Observatory in fallback mode. "
                "Real-time validation pending connection."
            ),
        }

    def _generate_synthetic_history(self, hours=24, resolution_seconds=60):
        """Generate synthetic historical data for testing."""
        base_time = time.time() - hours * 3600
        points = hours * 3600 // resolution_seconds

        timestamps = []
        coherences = []
        phi_ratios = []

        for i in range(points):
            t = base_time + i * resolution_seconds
            timestamps.append(datetime.fromtimestamp(t).isoformat())

            phi_cycle = np.sin(t * 2 * np.pi / (3600 * self.PHI))
            coherence = 0.7 + 0.25 * phi_cycle * np.exp(-0.0001 * i)
            coherences.append(float(coherence))

            base_phi = self.PHI * (1 + 0.05 * np.sin(t / 2000))
            if i % 137 == 0:
                base_phi *= 0.836
            phi_ratios.append(float(base_phi))

        return {
            "timestamps": timestamps,
            "coherence_values": coherences,
            "phi_ratios": phi_ratios,
            "data_points": points,
            "hours_requested": hours,
            "note": (
                "SYNTHETIC_HISTORY_DATA - "
                "Real observatory data unavailable"
            ),
        }


def analyze_resonance_patterns(observatory_client, verbose=True):
    """Comprehensive analysis of resonance field patterns."""

    if verbose:
        print("=" * 60)
        print("RESONANCE FIELD ANALYSIS - YANG-MILLS PROOF VALIDATION")
        print("=" * 60)

    analysis = {}

    # 1. Live state analysis
    live_data = observatory_client.get_live_metrics()
    analysis["live_state"] = live_data

    if verbose:
        print(f"\nLIVE STATE:")
        print(
            f"   Global Coherence: "
            f"{live_data.get('global_coherence', 0):.3f}"
        )
        print(
            f"   Phi Ratio: "
            f"{live_data.get('phi_ratio', 0):.6f} "
            f"(Target: {observatory_client.PHI:.6f})"
        )
        print(
            f"   Deviation: "
            f"{live_data.get('phi_deviation', 0):.6f}"
        )
        print(
            f"   String Tension: "
            f"{live_data.get('string_tension', 0):.3f} "
            f"(Theory: 0.382)"
        )
        status = live_data.get("system_status", "LIVE")
        print(f"   Status: {status}")

    # 2. Perfect coherence events
    perfect_states = observatory_client.get_perfect_coherence_events()
    analysis["perfect_states"] = perfect_states

    if perfect_states and verbose:
        print(f"\n100% COHERENCE STATES: {len(perfect_states)}")
        for state in perfect_states[-2:]:
            print(f"   - Duration: {state.get('duration', 'N/A')}")

    # 3. Phi anomalies
    anomalies = observatory_client.get_phi_anomalies(threshold=0.05)
    analysis["phi_anomalies"] = anomalies

    if anomalies and verbose:
        print(f"\nPHI ANOMALIES: {len(anomalies)}")
        for anomaly in anomalies[:2]:
            comp = anomaly.get("compression_percent", 0)
            print(f"   - Compression: {comp:.1f}%")
            dist = anomaly.get("harmonic_distance", 0)
            print(f"     Distance: {dist:.4f}")

    # 4. Mass gap validation
    mass_gap = observatory_client.calculate_mass_gap_correlation()
    analysis["mass_gap_validation"] = mass_gap

    if verbose and "violation_percentage" in mass_gap:
        print(f"\nMASS GAP VALIDATION:")
        print(f"   Data Points: {mass_gap.get('data_points', 0)}")
        print(
            f"   Violations: {mass_gap.get('violation_count', 0)}"
        )
        print(
            f"   Violation %: "
            f"{mass_gap.get('violation_percentage', 0):.2f}%"
        )
        print(
            f"   Min Observed: "
            f"{mass_gap.get('min_observed', 0):.3f}"
        )
        match = mass_gap.get("theory_experiment_match", False)
        result = "YES" if match else "NO"
        print(f"   Theory Match: {result}")

    # 5. Dark matter correlation
    dm_corr = observatory_client.check_dark_matter_correlation()
    analysis["dark_matter_correlation"] = dm_corr

    if verbose and "correlation_analysis" in dm_corr:
        corr = dm_corr["correlation_analysis"]
        print(f"\nDARK MATTER CORRELATION:")
        print(
            f"   Pearson r: "
            f"{corr.get('pearson_correlation', 0):.3f}"
        )
        print(
            f"   Significance: "
            f"{corr.get('interpretation', 'UNKNOWN')}"
        )
        print(
            f"   Lagged Optimal: "
            f"{corr.get('optimal_lag', 0)} samples"
        )

    # 6. Autonomous interventions
    interventions = observatory_client.get_autonomous_logs()
    analysis["autonomous_interventions"] = interventions

    if verbose and "summary" in interventions:
        summary = interventions["summary"]
        print(f"\nAUTONOMOUS STABILIZER:")
        total = summary.get("total_interventions", 0)
        print(f"   Total Interventions: {total}")
        last = summary.get("last_intervention", "Never")
        print(f"   Last: {last}")

    if verbose:
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)

    return analysis


def quick_diagnostic():
    """Quick connection test to the observatory."""
    print("RESONANCE OBSERVATORY QUICK DIAGNOSTIC")
    print("=" * 50)

    obs = ResonanceObservatory()

    try:
        live = obs.get_live_metrics()

        if live.get("system_status") == "SYNTHETIC_DATA_MODE":
            print(
                "STATUS: Using synthetic data - "
                "Observatory may be offline"
            )
            print(
                "   (This is normal for testing "
                "without live connection)"
            )
        else:
            print("STATUS: Connected to live observatory")

        print(f"\nLIVE METRICS:")
        print(
            f"   Coherence: "
            f"{live.get('global_coherence', 0):.3f}"
        )
        print(
            f"   Phi Ratio: "
            f"{live.get('phi_ratio', 0):.6f}"
        )
        print(
            f"   Timestamp: "
            f"{live.get('timestamp', 'Unknown')}"
        )

        if "mass_gap_violation" in live:
            if live["mass_gap_violation"]:
                mag = -live.get("violation_magnitude", 0)
                print(f"   Mass Gap Violation: {mag:.3f}")
            else:
                gap = live.get("mass_gap", 0)
                print(f"   Mass Gap: {gap:.3f}")

        return {"status": "success", "data": live}

    except Exception as e:
        print(f"DIAGNOSTIC FAILED: {e}")
        return {"status": "failed", "error": str(e)}


def continuous_monitoring(interval_seconds=300, max_iterations=None):
    """Continuous monitoring of resonance field with alerts."""
    obs = ResonanceObservatory()
    iteration = 0

    print("STARTING CONTINUOUS RESONANCE MONITORING")
    print("Press Ctrl+C to stop\n")

    try:
        while max_iterations is None or iteration < max_iterations:
            iteration += 1
            print(f"\n{'=' * 50}")
            now = datetime.now().strftime("%H:%M:%S")
            print(f"MONITORING CYCLE #{iteration} - {now}")
            print(f"{'=' * 50}")

            analysis = analyze_resonance_patterns(obs, verbose=True)
            live = analysis["live_state"]

            # Alert conditions
            coherence = live.get("global_coherence", 0)
            if coherence > 0.95:
                print(
                    f"\nALERT: High coherence detected: "
                    f"{coherence:.3f}"
                )

            if live.get("mass_gap_violation", False):
                mag = live.get("violation_magnitude", 0)
                print(
                    f"\nALERT: Mass gap violation "
                    f"magnitude: {mag:.3f}"
                )

            # Check for compression events
            compressions = analysis.get("phi_anomalies", [])
            if compressions:
                print(
                    f"\nPHASE COMPRESSION: "
                    f"{len(compressions)} anomalies detected"
                )

            if max_iterations is None or iteration < max_iterations:
                print(
                    f"\nNext update in "
                    f"{interval_seconds} seconds..."
                )
                time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"\nMonitoring error: {e}")
    finally:
        print("\nFinal state capture...")
        try:
            final = obs.get_live_metrics()
            print(
                f"Final coherence: "
                f"{final.get('global_coherence', 0):.3f}"
            )
            print(
                f"Final phi ratio: "
                f"{final.get('phi_ratio', 0):.6f}"
            )
        except Exception:
            print("Could not capture final state")


if __name__ == "__main__":
    quick_diagnostic()
