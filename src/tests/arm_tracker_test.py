from ..modules.arm_tracker import ArmTracker

try:
    tracker = ArmTracker()

    tracker.track()

except Exception as e:
    print(f"Error running arm tracker test: {e}")