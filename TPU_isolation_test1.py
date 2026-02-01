#!/usr/bin/env python3
"""
Test TPU device isolation by running a computation on a specific device.
Usage: python test_tpu_device.py <device_id>
"""
import sys
import time
import jax
import jax.numpy as jnp

def test_device_isolation(device_id):
    # Get all devices
    devices = jax.devices()
    print(f"All available devices: {devices}")
    
    if device_id >= len(devices):
        print(f"ERROR: Device {device_id} doesn't exist. Only {len(devices)} devices available.")
        sys.exit(1)
    
    target_device = devices[device_id]
    print(f"\nTarget device for this process: {target_device}")
    print(f"Device coords: {target_device.coords}")
    print(f"Device core_on_chip: {target_device.core_on_chip}")
    
    # Set as default
    jax.config.update('jax_default_device', target_device)
    
    # Verify it's actually using the right device
    print(f"\nDefault device after config: {jax.devices()[0]}")
    
    # Run a computation to verify device is working
    print(f"\nRunning test computation on device {device_id}...")
    
    # Create some test data and run computation
    key = jax.random.PRNGKey(device_id)
    
    @jax.jit
    def compute(x):
        # Do something non-trivial to actually use the TPU
        for _ in range(100):
            x = jnp.sin(x) @ jnp.cos(x.T)
        return jnp.sum(x)
    
    # Allocate on specific device
    x = jax.random.normal(key, (1000, 1000))
    
    # Check where the array lives
    print(f"Array allocated on: {x.devices()}")
    
    # Time the computation
    start = time.time()
    result = compute(x)
    result.block_until_ready()  # Wait for TPU to finish
    elapsed = time.time() - start
    
    print(f"Computation result: {result}")
    print(f"Time elapsed: {elapsed:.3f}s")
    print(f"Result device: {result.devices()}")
    
    # Keep the process alive for a bit so we can check multiple in parallel
    print(f"\nSUCCESS: Device {device_id} is working!")
    print("Sleeping for 30 seconds (so you can launch other tests)...")
    time.sleep(30)
    print("Done!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_tpu_device.py <device_id>")
        sys.exit(1)
    
    device_id = int(sys.argv[1])
    test_device_isolation(device_id)
