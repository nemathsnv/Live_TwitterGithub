import subprocess

subprocess.run("python live_dash.py & python stream_twitter.py", shell=True)