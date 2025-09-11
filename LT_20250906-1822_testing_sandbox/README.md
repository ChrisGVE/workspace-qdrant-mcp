# Autonomous Stress Testing Sandbox

This directory contains a comprehensive autonomous stress testing framework for the workspace-qdrant MCP system. Designed for overnight operation with full safety monitoring and emergency stop capabilities.

## 🎯 Overview

The testing sandbox provides:

- **8-12 hour autonomous operation** with minimal human intervention
- **Progressive stress testing** with automatic escalation
- **Comprehensive safety monitoring** with emergency stops
- **Real-time resource tracking** and performance analysis
- **Decision-ready reports** with actionable recommendations

## 📁 Directory Structure

```
LT_20250906-1822_testing_sandbox/
├── baseline_metrics/          # Current performance baselines
├── qmk_integration/          # Large-scale integration tests
├── stress_scenarios/         # Synthetic load testing
├── sync_validation/          # Real-time change validation
├── monitoring_logs/          # All performance data and logs
├── results_summary/          # Final reports and analysis
├── scripts/                  # Core testing automation
└── safety_monitoring/        # System protection mechanisms
```

## 🚀 Quick Start

### Launch Autonomous Testing

```bash
# Full autonomous testing campaign
python3 launch_autonomous_testing.py

# Check prerequisites only
python3 launch_autonomous_testing.py --check-only

# Emergency stop any running tests
python3 launch_autonomous_testing.py --emergency-stop
```

### Manual Component Testing

```bash
# Run baseline collection
python3 baseline_metrics/baseline_collector.py

# Run concurrent load test
python3 stress_scenarios/concurrent_load_test.py

# Start resource monitoring
python3 scripts/resource_monitor.py

# System safety guardian
python3 safety_monitoring/system_guardian.py
```

## 🛡️ Safety Features

### Emergency Stop System

```bash
# Manual emergency stop
python3 safety_monitoring/emergency_stop.py

# Force kill all test processes
python3 safety_monitoring/emergency_stop.py --force
```

### Safety Monitoring

- **Real-time resource monitoring** (memory, CPU, disk I/O)
- **Automatic thresholds** with progressive escalation
- **Emergency circuit breakers** for runaway processes
- **System resource reservation** (20% CPU/memory kept free)
- **Automatic test suspension** on safety violations

### Safety Thresholds

- Memory usage: 80% maximum (adjustable per test phase)
- CPU usage: 85% maximum (adjustable per test phase)
- Disk usage: 90% maximum
- Maximum 3 consecutive test failures before campaign halt

## 📊 Test Phases

The autonomous system runs through progressive test phases:

1. **Warmup** (30min) - 2 connections, 10 ops, low stress
2. **Light Load** (60min) - 3 connections, 25 ops, low stress
3. **Moderate Load 1** (90min) - 5 connections, 40 ops, moderate stress
4. **Moderate Load 2** (90min) - 7 connections, 50 ops, moderate stress
5. **High Load 1** (120min) - 10 connections, 75 ops, high stress
6. **High Load 2** (120min) - 12 connections, 100 ops, high stress
7. **Stress Test** (150min) - 15 connections, 150 ops, extreme stress
8. **Endurance Test** (180min) - 8 connections, 200 ops, sustained load

Phases are automatically adjusted based on system capacity and health score.

## 📈 Monitoring & Reporting

### Real-Time Monitoring

- **monitoring_logs/** - Live logs from all components
- **Resource metrics** collected every 10 seconds
- **Safety checks** performed every 30 seconds
- **Campaign progress** logged every 30 seconds during active phases

### Final Reports

- **comprehensive_baseline_report.json** - Pre-test system state
- **autonomous_campaign_report.json** - Complete campaign analysis
- **phase_report_*.json** - Individual phase results
- **campaign_summary_*.txt** - Human-readable summary

### Key Metrics

- **Success rates** per operation type and phase
- **Response times** (average, 95th, 99th percentiles)
- **Operations per second** sustained throughout campaign
- **Resource utilization** trends and peak usage
- **Error patterns** and recovery performance

## ⚙️ Configuration

### System Requirements

- **Python 3.7+** with asyncio support
- **4GB+ available memory** (8GB+ recommended)
- **1GB+ free disk space** for logs and reports
- **psutil module** for system monitoring

### Campaign Configuration

Modify `CampaignConfig` in `scripts/autonomous_test_runner.py`:

```python
CampaignConfig(
    total_duration_hours=8,        # Campaign duration
    safety_check_interval_minutes=15,  # Safety check frequency
    baseline_health_requirement=70,     # Minimum health score
    max_consecutive_failures=3,         # Failure tolerance
    emergency_cooldown_minutes=30,      # Recovery time
    progressive_escalation=True,        # Enable phase escalation
    auto_recovery=True                  # Enable automatic recovery
)
```

### Safety Configuration

Modify `safety_monitoring/safety_config.json`:

```json
{
  "memory_threshold_percent": 80,
  "cpu_threshold_percent": 85,
  "disk_threshold_percent": 90,
  "check_interval_seconds": 30,
  "emergency_stop_on_consecutive_violations": 3,
  "max_test_duration_hours": 12
}
```

## 🔧 Troubleshooting

### Common Issues

**Campaign won't start:**
```bash
# Check prerequisites
python3 launch_autonomous_testing.py --check-only

# Check system health
python3 baseline_metrics/baseline_collector.py
```

**High resource usage:**
```bash
# Check current system state
python3 scripts/resource_monitor.py

# Emergency stop if needed
python3 safety_monitoring/emergency_stop.py
```

**Test failures:**
```bash
# Review latest logs
tail -f monitoring_logs/autonomous_test_runner_*.log

# Check safety violations
tail -f monitoring_logs/system_guardian_*.log
```

### Log Analysis

Key log files:
- `autonomous_test_runner_*.log` - Campaign orchestration
- `system_guardian_*.log` - Safety monitoring
- `resource_monitor_*.log` - System metrics
- `concurrent_load_test_*.log` - Individual test phases

## 📋 Best Practices

### Before Running

1. **Close unnecessary applications** to free system resources
2. **Ensure stable network connection** for MCP operations
3. **Check disk space** for log storage (1GB minimum)
4. **Review safety thresholds** based on your system
5. **Plan for overnight operation** - minimal interruption needed

### During Operation

1. **Monitor progress** via log files
2. **Check results_summary/** for interim reports
3. **Use emergency stop** if system becomes unresponsive
4. **Avoid system-intensive tasks** during testing

### After Completion

1. **Review campaign summary** for key findings
2. **Check performance trends** for degradation patterns
3. **Implement recommendations** for production deployment
4. **Archive test results** for future reference
5. **Clean up temp files** if disk space needed

## 🎯 Success Criteria

### Campaign Success

- **70%+ phase success rate** across all test phases
- **90%+ operation success rate** for individual operations
- **Stable response times** without significant degradation
- **No emergency stops** due to safety violations
- **Complete autonomous operation** with minimal intervention

### Performance Benchmarks

- **Low stress**: 95%+ success rate, <2s response times
- **Moderate stress**: 90%+ success rate, <3s response times
- **High stress**: 85%+ success rate, <5s response times
- **Extreme stress**: 80%+ success rate, <8s response times

## 📞 Support

For issues or questions:

1. **Check logs** in monitoring_logs/ directory
2. **Review safety reports** in results_summary/
3. **Use emergency stop** if system becomes unstable
4. **Examine configuration files** for threshold adjustments

---

**⚠️ Important**: This testing framework is designed for autonomous operation but includes comprehensive safety mechanisms. Always review system capacity and adjust configurations appropriately before launching extended testing campaigns.