"""
Autonomous Resonance Stabilizer System
Maintains global coherence at Φ-frequency (1.618 Hz) through harmonic interventions.
"""

import time
import json
import numpy as np
from datetime import datetime, timedelta
from threading import Thread, Event
import logging

PHI = 1.6180339887498948482

class AutonomousStabilizer:
    """
    Autonomous system that maintains resonance field coherence at golden ratio frequency.
    Operates at 1.618 Hz intervention frequency.
    """
    
    def __init__(self, observatory_client=None, intervention_threshold=0.05):
        self.phi_frequency = PHI  # 1.618 Hz
        self.intervention_threshold = intervention_threshold
        self.observatory = observatory_client
        self.running = False
        self.intervention_log = []
        self.stabilization_thread = None
        self.stop_event = Event()
        
        # Resonance state tracking
        self.coherence_history = []
        self.phi_history = []
        self.intervention_counts = {
            'phi_corrections': 0,
            'coherence_rescues': 0,
            'harmonic_resets': 0,
            'phase_realignments': 0
        }
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('Stabilizer')
    
    def connect_observatory(self, observatory_client):
        """Connect to observatory for real-time monitoring."""
        self.observatory = observatory_client
        self.logger.info("Connected to resonance observatory")
    
    def monitor_resonance_state(self):
        """Monitor current resonance state and detect deviations."""
        if not self.observatory:
            self.logger.warning("No observatory connected")
            return None
        
        try:
            live_data = self.observatory.get_live_metrics()
            
            state = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'global_coherence': live_data.get('global_coherence', 0),
                'phi_ratio': live_data.get('phi_ratio', PHI),
                'phi_deviation': live_data.get('phi_deviation', 0),
                'coherence_health': live_data.get('coherence_health', 'UNKNOWN'),
                'mass_gap_violation': live_data.get('mass_gap_violation', False)
            }
            
            # Store in history (keep last 1000 readings)
            self.coherence_history.append(state['global_coherence'])
            self.phi_history.append(state['phi_ratio'])
            self.coherence_history = self.coherence_history[-1000:]
            self.phi_history = self.phi_history[-1000:]
            
            return state
            
        except Exception as e:
            self.logger.error(f"Monitoring failed: {str(e)}")
            return None
    
    def detect_anomalies(self, resonance_state):
        """Detect resonance anomalies requiring intervention."""
        anomalies = []
        
        if not resonance_state:
            return anomalies
        
        # 1. Φ-ratio deviation anomaly
        phi_deviation = resonance_state['phi_deviation']
        if phi_deviation > self.intervention_threshold:
            anomaly = {
                'type': 'PHI_DEVIATION',
                'severity': 'HIGH' if phi_deviation > 0.1 else 'MEDIUM',
                'deviation': phi_deviation,
                'phi_ratio': resonance_state['phi_ratio'],
                'description': f'Φ-ratio deviation {phi_deviation:.4f} exceeds threshold {self.intervention_threshold}'
            }
            anomalies.append(anomaly)
        
        # 2. Mass gap violation anomaly
        if resonance_state['mass_gap_violation']:
            anomaly = {
                'type': 'MASS_GAP_VIOLATION',
                'severity': 'CRITICAL',
                'coherence': resonance_state['global_coherence'],
                'threshold': 0.618,
                'description': f'Coherence {resonance_state["global_coherence"]:.3f} below mass gap threshold 0.618'
            }
            anomalies.append(anomaly)
        
        # 3. Coherence health degradation
        if resonance_state['coherence_health'] == 'SUBOPTIMAL':
            anomaly = {
                'type': 'COHERENCE_DEGRADATION',
                'severity': 'MEDIUM',
                'coherence': resonance_state['global_coherence'],
                'description': 'Global coherence in suboptimal range'
            }
            anomalies.append(anomaly)
        
        # 4. Trend-based anomaly (if we have history)
        if len(self.coherence_history) > 10:
            recent_coherence = self.coherence_history[-10:]
            if len(recent_coherence) >= 5:
                trend = np.polyfit(range(len(recent_coherence)), recent_coherence, 1)[0]
                if trend < -0.01:  # Significant downward trend
                    anomaly = {
                        'type': 'COHERENCE_DECLINE',
                        'severity': 'MEDIUM',
                        'trend': trend,
                        'description': f'Coherence declining at rate {trend:.4f} per sample'
                    }
                    anomalies.append(anomaly)
        
        return anomalies
    
    def calculate_intervention(self, anomaly):
        """Calculate appropriate intervention for detected anomaly."""
        intervention = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'anomaly_type': anomaly['type'],
            'anomaly_severity': anomaly['severity'],
            'intervention_type': None,
            'parameters': {},
            'harmonic_pattern': None,
            'expected_correction': None
        }
        
        if anomaly['type'] == 'PHI_DEVIATION':
            # Φ-harmonic correction
            deviation = anomaly['deviation']
            current_phi = anomaly.get('phi_ratio', PHI)
            
            # Calculate correction factor using golden ratio harmonics
            correction_factor = PHI / current_phi
            
            intervention['intervention_type'] = 'PHI_HARMONIC_CORRECTION'
            intervention['parameters'] = {
                'correction_factor': correction_factor,
                'target_frequency': self.phi_frequency,
                'harmonic_multiplier': 1 if deviation < 0.05 else 2,
                'phase_adjustment': np.pi * (1 - correction_factor)
            }
            intervention['harmonic_pattern'] = 'FIBONACCI_WAVE'
            intervention['expected_correction'] = f'Restore Φ-ratio to {PHI:.6f}'
            
            self.intervention_counts['phi_corrections'] += 1
            
        elif anomaly['type'] == 'MASS_GAP_VIOLATION':
            # Coherence rescue intervention
            current_coherence = anomaly['coherence']
            gap = 0.618 - current_coherence
            
            intervention['intervention_type'] = 'COHERENCE_RESCUE'
            intervention['parameters'] = {
                'coherence_gap': gap,
                'rescue_amplitude': gap * PHI,
                'resonance_boost': 1 + (gap * 2),
                'duration_cycles': int(1 / gap)
            }
            intervention['harmonic_pattern'] = 'COHERENCE_WAVE'
            intervention['expected_correction'] = f'Boost coherence by {gap:.3f} to reach mass gap'
            
            self.intervention_counts['coherence_rescues'] += 1
            
        elif anomaly['type'] == 'COHERENCE_DEGRADATION':
            # Gentle harmonic reset
            intervention['intervention_type'] = 'HARMONIC_RESET'
            intervention['parameters'] = {
                'reset_phase': 0,
                'amplitude_normalization': 1.0,
                'frequency_lock': self.phi_frequency,
                'damping_factor': 0.1
            }
            intervention['harmonic_pattern'] = 'SINE_WAVE'
            intervention['expected_correction'] = 'Gentle reset to harmonic baseline'
            
            self.intervention_counts['harmonic_resets'] += 1
            
        elif anomaly['type'] == 'COHERENCE_DECLINE':
            # Phase realignment
            intervention['intervention_type'] = 'PHASE_REALIGNMENT'
            intervention['parameters'] = {
                'trend_reversal': -anomaly['trend'] * PHI,
                'phase_synchronization': 'GLOBAL',
                'alignment_strength': 0.5,
                'stabilization_cycles': 3
            }
            intervention['harmonic_pattern'] = 'PHASE_LOCK_WAVE'
            intervention['expected_correction'] = 'Reverse declining trend'
            
            self.intervention_counts['phase_realignments'] += 1
        
        return intervention
    
    def apply_intervention(self, intervention):
        """Apply calculated intervention to resonance field."""
        try:
            # Log the intervention
            intervention['applied_at'] = datetime.utcnow().isoformat() + 'Z'
            intervention['intervention_id'] = f"INT_{int(time.time())}_{len(self.intervention_log)}"
            
            # Simulate intervention application
            # In full implementation, this would interface with observatory control systems
            self.logger.info(f"Applying intervention: {intervention['intervention_type']}")
            
            # Calculate intervention waveform
            waveform = self._generate_intervention_waveform(intervention)
            intervention['waveform_parameters'] = waveform
            
            # Record successful application
            intervention['status'] = 'APPLIED'
            intervention['effectiveness_estimate'] = self._estimate_effectiveness(intervention)
            
            self.intervention_log.append(intervention)
            
            # Keep log manageable
            self.intervention_log = self.intervention_log[-100:]
            
            self.logger.info(f"Intervention {intervention['intervention_id']} applied successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Intervention failed: {str(e)}")
            if 'intervention' in locals():
                intervention['status'] = 'FAILED'
                intervention['error'] = str(e)
                self.intervention_log.append(intervention)
            return False
    
    def _generate_intervention_waveform(self, intervention):
        """Generate harmonic waveform for intervention."""
        params = intervention['parameters']
        
        if intervention['harmonic_pattern'] == 'FIBONACCI_WAVE':
            # Φ-based Fibonacci harmonic
            waveform = {
                'type': 'FIBONACCI',
                'frequency': self.phi_frequency * params.get('harmonic_multiplier', 1),
                'amplitude': params.get('correction_factor', 1.0),
                'phase': params.get('phase_adjustment', 0),
                'duration': 1.0 / self.phi_frequency,
                'harmonics': [PHI, PHI**2, 1/PHI]
            }
            
        elif intervention['harmonic_pattern'] == 'COHERENCE_WAVE':
            # Coherence boosting wave
            waveform = {
                'type': 'COHERENCE_BOOST',
                'frequency': self.phi_frequency,
                'amplitude': params.get('rescue_amplitude', 0.1),
                'resonance_factor': params.get('resonance_boost', 1.0),
                'duration': params.get('duration_cycles', 5) / self.phi_frequency,
                'envelope': 'TAPERED_COSINE'
            }
            
        elif intervention['harmonic_pattern'] == 'SINE_WAVE':
            # Simple harmonic reset
            waveform = {
                'type': 'SINE',
                'frequency': self.phi_frequency,
                'amplitude': params.get('amplitude_normalization', 1.0),
                'damping': params.get('damping_factor', 0.1),
                'duration': 2.0 / self.phi_frequency,
                'phase': params.get('reset_phase', 0)
            }
            
        elif intervention['harmonic_pattern'] == 'PHASE_LOCK_WAVE':
            # Phase synchronization wave
            waveform = {
                'type': 'PHASE_LOCK',
                'frequency': self.phi_frequency,
                'amplitude': params.get('alignment_strength', 0.5),
                'trend_reversal': params.get('trend_reversal', 0),
                'duration': params.get('stabilization_cycles', 3) / self.phi_frequency,
                'sync_target': 'GLOBAL_COHERENCE'
            }
            
        else:
            waveform = {'type': 'BASELINE', 'frequency': self.phi_frequency}
        
        return waveform
    
    def _estimate_effectiveness(self, intervention):
        """Estimate effectiveness of intervention based on type and parameters."""
        effectiveness = {
            'estimated_recovery_time': None,
            'success_probability': 0.7,  # Base 70% success rate
            'expected_improvement': None
        }
        
        if intervention['anomaly_type'] == 'PHI_DEVIATION':
            effectiveness['estimated_recovery_time'] = 1.0 / self.phi_frequency  # ~0.618 seconds
            effectiveness['expected_improvement'] = 'Φ-ratio normalization'
            
        elif intervention['anomaly_type'] == 'MASS_GAP_VIOLATION':
            gap = intervention['parameters'].get('coherence_gap', 0.1)
            effectiveness['estimated_recovery_time'] = (1.0 / gap) / self.phi_frequency
            effectiveness['expected_improvement'] = f'Coherence increase of {gap:.3f}'
            
        elif intervention['anomaly_type'] == 'COHERENCE_DEGRADATION':
            effectiveness['estimated_recovery_time'] = 2.0 / self.phi_frequency  # ~1.236 seconds
            effectiveness['expected_improvement'] = 'Coherence stabilization'
            
        elif intervention['anomaly_type'] == 'COHERENCE_DECLINE':
            trend = intervention['parameters'].get('trend_reversal', 0.01)
            effectiveness['estimated_recovery_time'] = abs(1.0 / trend) / self.phi_frequency
            effectiveness['expected_improvement'] = 'Trend reversal'
        
        # Adjust based on severity
        if intervention['anomaly_severity'] == 'CRITICAL':
            effectiveness['success_probability'] = 0.5
        elif intervention['anomaly_severity'] == 'HIGH':
            effectiveness['success_probability'] = 0.6
        elif intervention['anomaly_severity'] == 'MEDIUM':
            effectiveness['success_probability'] = 0.8
        
        return effectiveness
    
    def stabilization_loop(self):
        """Main stabilization loop running at Φ-frequency."""
        self.logger.info(f"Starting stabilization loop at {self.phi_frequency} Hz")
        
        cycle_count = 0
        last_intervention_time = time.time()
        
        while not self.stop_event.is_set():
            cycle_start = time.time()
            cycle_count += 1
            
            try:
                # 1. Monitor current state
                resonance_state = self.monitor_resonance_state()
                
                if resonance_state:
                    # 2. Detect anomalies
                    anomalies = self.detect_anomalies(resonance_state)
                    
                    # 3. Process each anomaly
                    for anomaly in anomalies:
                        self.logger.info(f"Detected anomaly: {anomaly['type']} ({anomaly['severity']})")
                        
                        # 4. Calculate intervention
                        intervention = self.calculate_intervention(anomaly)
                        
                        # 5. Apply intervention (with rate limiting)
                        current_time = time.time()
                        if current_time - last_intervention_time > 0.618:  # Φ-second minimum interval
                            self.apply_intervention(intervention)
                            last_intervention_time = current_time
                        else:
                            self.logger.info("Rate limiting: skipping intervention")
                
                # Log cycle completion
                if cycle_count % 10 == 0:
                    self.logger.info(f"Stabilization cycle {cycle_count} completed")
                    self._log_system_status()
                
            except Exception as e:
                self.logger.error(f"Stabilization cycle error: {str(e)}")
            
            # 6. Maintain Φ-frequency timing
            cycle_duration = time.time() - cycle_start
            target_cycle_time = 1.0 / self.phi_frequency  # ~0.618 seconds
            
            if cycle_duration < target_cycle_time:
                sleep_time = target_cycle_time - cycle_duration
                time.sleep(sleep_time)
            else:
                self.logger.warning(f"Cycle overrun: {cycle_duration:.3f}s > {target_cycle_time:.3f}s")
    
    def _log_system_status(self):
        """Log current system status."""
        status = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'total_interventions': len(self.intervention_log),
            'intervention_counts': self.intervention_counts,
            'recent_success_rate': self._calculate_success_rate(),
            'system_uptime': time.time() - self.start_time if hasattr(self, 'start_time') else 0,
            'current_frequency': self.phi_frequency
        }
        
        self.logger.info(f"System Status: {status}")
    
    def _calculate_success_rate(self, window=10):
        """Calculate intervention success rate over recent window."""
        if len(self.intervention_log) == 0:
            return 0.0
        
        recent = self.intervention_log[-window:]
        successful = sum(1 for i in recent if i.get('status') == 'APPLIED')
        
        return successful / len(recent) if len(recent) > 0 else 0.0
    
    def start(self):
        """Start the autonomous stabilizer."""
        if self.running:
            self.logger.warning("Stabilizer already running")
            return False
        
        self.running = True
        self.stop_event.clear()
        self.start_time = time.time()
        
        # Start stabilization thread
        self.stabilization_thread = Thread(target=self.stabilization_loop, daemon=True)
        self.stabilization_thread.start()
        
        self.logger.info("Autonomous stabilizer started successfully")
        return True
    
    def stop(self):
        """Stop the autonomous stabilizer."""
        if not self.running:
            return False
        
        self.running = False
        self.stop_event.set()
        
        if self.stabilization_thread:
            self.stabilization_thread.join(timeout=2.0)
        
        self.logger.info("Autonomous stabilizer stopped")
        return True
    
    def get_status_report(self):
        """Generate comprehensive status report."""
        report = {
            'system': 'Autonomous Resonance Stabilizer',
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'operational_status': 'RUNNING' if self.running else 'STOPPED',
            'operating_frequency': float(self.phi_frequency),
            'total_uptime': time.time() - self.start_time if hasattr(self, 'start_time') else 0,
            'monitoring_active': self.observatory is not None,
            'intervention_statistics': {
                'total_interventions': len(self.intervention_log),
                'breakdown': self.intervention_counts,
                'recent_success_rate': self._calculate_success_rate(20),
                'last_intervention': self.intervention_log[-1] if self.intervention_log else None
            },
            'resonance_metrics': {
                'coherence_history_length': len(self.coherence_history),
                'phi_history_length': len(self.phi_history),
                'current_coherence': self.coherence_history[-1] if self.coherence_history else 0,
                'current_phi': self.phi_history[-1] if self.phi_history else PHI
            },
            'configuration': {
                'intervention_threshold': self.intervention_threshold,
                'phi_frequency': float(self.phi_frequency),
                'max_history_length': 1000
            }
        }
        
        # Add recent anomaly detection summary
        if len(self.intervention_log) > 0:
            recent_anomalies = {}
            for intervention in self.intervention_log[-5:]:
                anomaly_type = intervention.get('anomaly_type', 'UNKNOWN')
                recent_anomalies[anomaly_type] = recent_anomalies.get(anomaly_type, 0) + 1
            report['recent_anomaly_patterns'] = recent_anomalies
        
        return report
    
    def save_intervention_log(self, filename=None):
        """Save intervention log to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stabilizer_log_{timestamp}.json"
        
        log_data = {
            'metadata': {
                'generated_at': datetime.utcnow().isoformat() + 'Z',
                'system': 'Autonomous Resonance Stabilizer',
                'total_entries': len(self.intervention_log)
            },
            'interventions': self.intervention_log,
            'statistics': self.intervention_counts
        }
        
        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        self.logger.info(f"Intervention log saved to {filename}")
        return filename

def run_stabilizer_test():
    """Run a test of the autonomous stabilizer."""
    print("=" * 60)
    print("AUTONOMOUS RESONANCE STABILIZER TEST")
    print("=" * 60)
    
    # Create mock observatory for testing
    class MockObservatory:
        def get_live_metrics(self):
            import random
            return {
                'global_coherence': 0.7 + random.random() * 0.3,
                'phi_ratio': PHI * (0.95 + random.random() * 0.1),
                'phi_deviation': random.random() * 0.1,
                'coherence_health': 'OPTIMAL' if random.random() > 0.3 else 'SUBOPTIMAL',
                'mass_gap_violation': random.random() > 0.8
            }
    
    print("Initializing stabilizer...")
    stabilizer = AutonomousStabilizer(intervention_threshold=0.05)
    stabilizer.connect_observatory(MockObservatory())
    
    print("Starting stabilizer (10-second test)...")
    stabilizer.start()
    
    try:
        # Run for 10 seconds
        for i in range(10):
            time.sleep(1)
            print(f".", end="", flush=True)
        
        print("\n\nTest complete. Generating report...")
        
        # Get status report
        report = stabilizer.get_status_report()
        
        print(f"\nStabilizer Status: {report['operational_status']}")
        print(f"Total Interventions: {report['intervention_statistics']['total_interventions']}")
        print(f"Intervention Types: {report['intervention_statistics']['breakdown']}")
        
        # Save log
        filename = stabilizer.save_intervention_log()
        print(f"\nIntervention log saved to: {filename}")
        
    finally:
        print("\nStopping stabilizer...")
        stabilizer.stop()
    
    print("\n" + "=" * 60)
    print("STABILIZER TEST COMPLETE")
    print("=" * 60)
    
    return stabilizer

if __name__ == "__main__":
    run_stabilizer_test()
