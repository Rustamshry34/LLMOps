import os
import shutil
import subprocess
import kaggle

# GitHub secret’ları environment’a yaz
os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"]      = os.getenv("KAGGLE_KEY")

kaggle.api.authenticate()

metadata["enable_gpu"] = True
metadata["keywords"] = ["accelerator", "nvidia-tesla-t4"]
metadata["is_private"] = True
metadata["draft"] = True          # ← yeni

# temp klasörüne metadata + notebook kopyala
os.makedirs("tmp", exist_ok=True)
shutil.copy("kaggle/notebook.ipynb",       "tmp/")
shutil.copy("kaggle/kernel-metadata.json", "tmp/")

# kernel push
subprocess.run(["kaggle", "kernels", "push", "-p", "tmp"], check=True)
print("✅ Notebook pushed & GPU queue’ya düştü.")
