"""Print the run ledger: what each recent run is, its state, and its numbers.

One markdown row per run directory, newest first: the experiment's definition read back
from its own `config.yaml`, whether it is running (with an ETA from its observed rate),
done, or stopped, the last logged metrics, and a link to its report.

Usage: python -m breeder.scripts.runs [limit]
"""

import csv
import datetime
import subprocess
import sys
from pathlib import Path

import yaml

OUTPUT = Path("/home/faldor_google_com/dev/cax/breeder/output")
BASE = "https://hp3d7589-5500.euw.devtunnels.ms/cax/breeder/output"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 20

running = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True).stdout


def _run_name(command: str) -> str:
	"""Return the name a live process runs under: its --name flag, else its config's stem."""
	fields = command.split()
	if "--name" in fields:
		return fields[fields.index("--name") + 1]
	return command.split("configs/")[1].split(".yaml")[0]


active = {
	_run_name(line)
	for line in running.splitlines()
	if "breeder.main" in line and "configs/" in line and "shell-snapshots" not in line
}

rows = []
recent = sorted(OUTPUT.glob("*_2026-*"), key=lambda p: p.stat().st_mtime, reverse=True)
for run_dir in recent[:LIMIT]:
	log = run_dir / "log.csv"
	config_path = run_dir / "config.yaml"
	if not log.exists() or not config_path.exists():
		continue
	config = yaml.safe_load(config_path.read_text())
	records = list(csv.DictReader(log.open()))
	if not records:
		continue
	last = records[-1]
	name = config["name"]
	generation = int(float(last["generation"]))
	total = config["qd"]["num_generations"]
	done = (run_dir / "report").exists() and generation >= total

	encoder = config.get("encoder") or {}
	fitness = config["fitness"]
	descriptor = ",".join(s[0] for s in config["descriptor"]["series"])
	objective = fitness.get("name", fitness.get("series", "?"))
	what = (
		f"{config['complex_system']['name']} · {encoder.get('name', 'none')}"
		f" · fit {objective} · desc {descriptor}"
	)

	if done:
		state = "done"
	elif name in active:
		rate_rows = records[len(records) // 2 :]
		g0, t0 = float(rate_rows[0]["generation"]), float(rate_rows[0]["time"])
		g1, t1 = float(rate_rows[-1]["generation"]), float(rate_rows[-1]["time"])
		rate = (t1 - t0) / max(g1 - g0, 1)
		eta = datetime.datetime.now() + datetime.timedelta(seconds=(total - generation) * rate)
		state = f"running {generation}/{total} · ETA {eta:%H:%M}"
	else:
		state = f"stopped at {generation}/{total}"

	rows.append(
		{
			"run": name,
			"dir": run_dir.name,
			"what": what,
			"state": state,
			"best": float(last["best_fitness"]),
			"valid": int(float(last["num_valid"])),
			"div": float(last["diversity"]),
			"vendi": float(last["vendi"]),
			"var": float(last["variance"]),
			"report": (run_dir / "report" / "index.html").exists(),
		}
	)

print("| run | what | state | best | valid | diversity | vendi | variance | report |")
print("|---|---|---|---|---|---|---|---|---|")
for r in rows:
	link = f"[open]({BASE}/{r['dir']}/report/index.html)" if r["report"] else "—"
	print(
		f"| `{r['run']}` | {r['what']} | {r['state']} | {r['best']:.4f} | {r['valid']} | "
		f"{r['div']:.3f} | {r['vendi']:.2f} | {r['var']:.5f} | {link} |"
	)
