import subprocess, os, sys
print(sys.version)
for m in ["jax", "flax", "optax", "datasets", "tokenizers", "numpy", "torch", "torch_xla"]:
    try:
        mod = __import__(m); print(m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "MISSING", type(e).__name__)
import jax
print(jax.devices())
import optax
print("optax muon:", hasattr(optax.contrib, "muon") if hasattr(optax, "contrib") else None)
print("cpus", os.cpu_count())
print(subprocess.run("df -h /kaggle/working /kaggle/tmp /tmp; free -g; nproc", shell=True, capture_output=True, text=True).stdout)
import jax.numpy as jnp, time
x = jnp.ones((8192, 8192), jnp.bfloat16)
f = jax.jit(lambda a: a @ a)
f(x).block_until_ready(); t = time.time()
for _ in range(20): y = f(x)
y.block_until_ready(); print("single-chip TFLOPs", 20 * 2 * 8192**3 / (time.time() - t) / 1e12)
