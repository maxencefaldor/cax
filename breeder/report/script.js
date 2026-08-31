const population = report.population;

document.getElementById("def-fitness").textContent = report.summary.fitness;
document.getElementById("def-descriptor").textContent = report.summary.descriptor;
document.getElementById("chip-count").textContent = population.length;
document.getElementById("chip-valid").textContent = report.summary.num_valid;
document.getElementById("chip-best").textContent = report.summary.best_fitness.toFixed(4);
document.getElementById("chip-diversity").textContent = report.summary.diversity.toFixed(4);
document.getElementById("chip-vendi").textContent = report.summary.vendi.toFixed(2);
document.getElementById("chip-vendi-cos").textContent = report.summary.vendi_cos.toFixed(2);
document.getElementById("config-yaml").textContent = report.config_yaml;

// Tabs
for (const button of document.querySelectorAll("nav button")) {
	button.addEventListener("click", () => {
		document.querySelectorAll("nav button").forEach((b) => b.classList.remove("active"));
		button.classList.add("active");
		for (const section of document.querySelectorAll("section")) {
			section.hidden = section.id !== `tab-${button.dataset.tab}`;
		}
		if (button.dataset.tab === "map") layoutMap();
		if (button.dataset.tab === "progress") drawCharts();
	});
}

// Load videos as they come into view and keep them loaded: browsers cap concurrent
// video decoders (~75 in Chrome) and every element past the cap renders black, so
// once more than BUDGET are held the least recently seen are released. Videos still
// in view are never released -- evicting those is what makes tiles go blank -- and
// releasing on scroll-out would refetch a video the moment it scrolled back.
const BUDGET = 64;
const loaded = new Set(); // iteration order: least recently in view first
const visible = new Set();

const observer = new IntersectionObserver(
	(entries) => {
		for (const entry of entries) {
			const video = entry.target;
			if (entry.isIntersecting) {
				if (!video.src) video.src = video.dataset.src;
				visible.add(video);
				loaded.delete(video);
				loaded.add(video);
			} else {
				visible.delete(video);
			}
		}
		for (const video of loaded) {
			if (loaded.size <= BUDGET) break;
			if (visible.has(video)) continue;
			video.pause();
			video.removeAttribute("src");
			video.load();
			loaded.delete(video);
		}
	},
	{ rootMargin: "300px" },
);

// Sequential single-hue ramp (dark surface): brighter = fitter
function color(individual) {
	if (individual.fitness === null) return "#4a5462";
	const t = 1 - individual.rank01;
	const lerp = (a, b) => Math.round(a + (b - a) * t);
	return `rgb(${lerp(35, 110)}, ${lerp(80, 231)}, ${lerp(92, 200)})`;
}

// ---- Individuals ----
let shown = [];

function render() {
	const sortKey = document.getElementById("sort").value;
	const descending = document.getElementById("direction").textContent === "↓";
	const count = Number(document.getElementById("count").value);
	const validOnly = document.getElementById("valid-only").checked;

	const value = (individual) => {
		const v = sortKey === "fitness" ? individual.fitness : individual.metrics[sortKey];
		return v === null ? -Infinity : v;
	};
	shown = population
		.filter((individual) => !validOnly || individual.fitness !== null)
		.sort((a, b) => (descending ? value(b) - value(a) : value(a) - value(b)))
		.slice(0, count);

	const grid = document.getElementById("grid");
	observer.disconnect();
	loaded.clear();
	visible.clear();
	grid.innerHTML = "";
	for (const individual of shown) {
		const card = document.createElement("div");
		card.className = "card";
		card.id = `individual-${individual.id}`;
		const video = document.createElement("video");
		video.dataset.src = `videos/${individual.id}.mp4`;
		video.muted = video.loop = video.autoplay = true;
		video.playsInline = true;
		observer.observe(video);
		const meta = document.createElement("div");
		meta.className = "meta";
		const fitness = individual.fitness === null ? "invalid" : individual.fitness.toFixed(4);
		meta.innerHTML = `<span class="id">#${individual.id}</span><span>${fitness}</span>`;
		card.append(video, meta);
		card.title = Object.entries(individual.metrics)
			.map(([name, value]) => `${name} ${value.toFixed(4)}`)
			.join(" · ");
		grid.append(card);
	}
	drawMap();
}

// ---- Map ----
const map = document.getElementById("map");
const preview = document.getElementById("preview");
const ctx = map.getContext("2d");
let pixelRatio = 1;

function layoutMap() {
	pixelRatio = window.devicePixelRatio || 1;
	const size = map.clientWidth;
	map.width = map.height = size * pixelRatio;
	drawMap();
}

function mapXY(individual) {
	const pad = 18 * pixelRatio;
	return [
		pad + individual.x * (map.width - 2 * pad),
		pad + individual.y * (map.height - 2 * pad),
	];
}

function drawMap() {
	ctx.clearRect(0, 0, map.width, map.height);
	for (const individual of [...shown].reverse()) {
		const [x, y] = mapXY(individual);
		ctx.fillStyle = color(individual);
		ctx.beginPath();
		ctx.arc(x, y, 3.5 * pixelRatio, 0, 2 * Math.PI);
		ctx.fill();
	}
}

function nearest(event) {
	const rect = map.getBoundingClientRect();
	const px = (event.clientX - rect.left) * pixelRatio;
	const py = (event.clientY - rect.top) * pixelRatio;
	let best = null, bestDistance = Infinity;
	for (const individual of shown) {
		const [x, y] = mapXY(individual);
		const distance = Math.hypot(x - px, y - py);
		if (distance < bestDistance) { bestDistance = distance; best = individual; }
	}
	return bestDistance <= 14 * pixelRatio ? best : null;
}

map.addEventListener("mousemove", (event) => {
	const individual = nearest(event);
	if (!individual) { preview.style.display = "none"; return; }
	const video = preview.querySelector("video");
	const src = `videos/${individual.id}.mp4`;
	if (!video.src.endsWith(src)) video.src = src;
	const fitness = individual.fitness === null ? "invalid" : individual.fitness.toFixed(4);
	preview.querySelector(".meta").textContent = `#${individual.id} · ${fitness}`;
	preview.style.left = `${Math.min(event.clientX + 16, window.innerWidth - 184)}px`;
	preview.style.top = `${Math.min(event.clientY + 16, window.innerHeight - 220)}px`;
	preview.style.display = "block";
});
map.addEventListener("mouseleave", () => { preview.style.display = "none"; });

map.addEventListener("click", (event) => {
	const individual = nearest(event);
	if (!individual) return;
	document.querySelector('nav button[data-tab="individuals"]').click();
	const card = document.getElementById(`individual-${individual.id}`);
	if (card) {
		card.scrollIntoView({ behavior: "smooth", block: "center" });
		document.querySelectorAll(".card.selected").forEach((c) => c.classList.remove("selected"));
		card.classList.add("selected");
	}
});

// ---- Progress ----
const CHARTS = [
	["best_fitness", "best fitness"],
	["mean_fitness", "mean fitness"],
	["diversity", "diversity (reference space)"],
	["vendi", "vendi score (effective species)"],
	["vendi_cos", "vendi score (cosine kernel)"],
	["child_valid", "child viability"],
	["variance", "phenotype variance"],
	["num_valid", "valid individuals"],
];
let chartsBuilt = false;
const INK_2 = "#98a3b1", INK_3 = "#5f6a78", GRID_LINE = "#1d242e", ACCENT = "#6ee7c8";

function niceStep(span, target) {
	const raw = span / target;
	const magnitude = 10 ** Math.floor(Math.log10(raw));
	for (const m of [1, 2, 5, 10]) if (raw <= m * magnitude) return m * magnitude;
}

function ticks(low, high, target) {
	const step = niceStep(high - low || 1, target);
	const out = [];
	for (let v = Math.ceil(low / step) * step; v <= high + step * 1e-9; v += step) out.push(v);
	return [out, step];
}

function formatNumber(value, step) {
	if (Math.abs(value) >= 1e6) return `${+(value / 1e6).toFixed(1)}M`;
	if (Math.abs(value) >= 1e4) return `${+(value / 1e3).toFixed(0)}k`;
	const decimals = Math.max(0, -Math.floor(Math.log10(step)));
	return value.toFixed(Math.min(decimals, 6));
}

function drawChart(canvas, hover) {
	const ratio = window.devicePixelRatio || 1;
	canvas.width = canvas.clientWidth * ratio;
	canvas.height = canvas.clientHeight * ratio;
	const context = canvas.getContext("2d");
	const xs = report.progress.evaluations;
	const ys = report.progress[canvas.dataset.key];
	const points = xs.map((x, i) => [x, ys[i]]).filter(([, y]) => Number.isFinite(y));
	if (points.length < 2) return;

	const margin = { left: 56 * ratio, right: 12 * ratio, top: 10 * ratio, bottom: 40 * ratio };
	const xLow = points[0][0], xHigh = points[points.length - 1][0];
	let yLow = Math.min(...points.map(([, y]) => y));
	let yHigh = Math.max(...points.map(([, y]) => y));
	if (yLow === yHigh) { yLow -= 1; yHigh += 1; }
	const yPad = (yHigh - yLow) * 0.06;
	yLow -= yPad; yHigh += yPad;

	const plotWidth = canvas.width - margin.left - margin.right;
	const plotHeight = canvas.height - margin.top - margin.bottom;
	const sx = (x) => margin.left + ((x - xLow) / (xHigh - xLow)) * plotWidth;
	const sy = (y) => margin.top + plotHeight - ((y - yLow) / (yHigh - yLow)) * plotHeight;

	context.font = `${11 * ratio}px system-ui, sans-serif`;

	// Recessive horizontal grid + y tick labels
	const [yTicks, yStep] = ticks(yLow, yHigh, 5);
	for (const tick of yTicks) {
		const y = sy(tick);
		context.strokeStyle = GRID_LINE;
		context.lineWidth = 1 * ratio;
		context.beginPath();
		context.moveTo(margin.left, y);
		context.lineTo(canvas.width - margin.right, y);
		context.stroke();
		context.fillStyle = INK_3;
		context.textAlign = "right";
		context.textBaseline = "middle";
		context.fillText(formatNumber(tick, yStep), margin.left - 8 * ratio, y);
	}

	// Baseline + x tick labels
	context.strokeStyle = "#2b3440";
	context.lineWidth = 1 * ratio;
	context.beginPath();
	context.moveTo(margin.left, margin.top + plotHeight);
	context.lineTo(canvas.width - margin.right, margin.top + plotHeight);
	context.stroke();
	const [xTicks, xStep] = ticks(xLow, xHigh, 5);
	for (const tick of xTicks) {
		context.fillStyle = INK_3;
		context.textAlign = "center";
		context.textBaseline = "top";
		context.fillText(formatNumber(tick, xStep), sx(tick), margin.top + plotHeight + 7 * ratio);
	}
	context.fillStyle = INK_3;
	context.textAlign = "center";
	context.textBaseline = "bottom";
	context.fillText("evaluations", margin.left + plotWidth / 2, canvas.height - 3 * ratio);

	// Series
	context.strokeStyle = ACCENT;
	context.lineWidth = 2 * ratio;
	context.lineJoin = "round";
	context.beginPath();
	points.forEach(([x, y], i) => (i ? context.lineTo(sx(x), sy(y)) : context.moveTo(sx(x), sy(y))));
	context.stroke();

	// Crosshair + readout on hover
	if (hover !== null) {
		let best = points[0];
		for (const point of points) {
			if (Math.abs(sx(point[0]) - hover) < Math.abs(sx(best[0]) - hover)) best = point;
		}
		const [hx, hy] = [sx(best[0]), sy(best[1])];
		context.strokeStyle = INK_3;
		context.lineWidth = 1 * ratio;
		context.setLineDash([4 * ratio, 4 * ratio]);
		context.beginPath();
		context.moveTo(hx, margin.top);
		context.lineTo(hx, margin.top + plotHeight);
		context.stroke();
		context.setLineDash([]);
		context.fillStyle = ACCENT;
		context.beginPath();
		context.arc(hx, hy, 3.5 * ratio, 0, 2 * Math.PI);
		context.fill();
		context.strokeStyle = "#0e1116";
		context.lineWidth = 2 * ratio;
		context.stroke();
		context.fillStyle = INK_2;
		context.textAlign = "right";
		context.textBaseline = "top";
		context.fillText(
			`${best[0].toLocaleString()} evaluations · ${formatNumber(best[1], yStep / 100)}`,
			canvas.width - margin.right - 2 * ratio, margin.top,
		);
	}
}

function drawCharts() {
	const charts = document.getElementById("charts");
	if (!chartsBuilt) {
		chartsBuilt = true;
		for (const [key, label] of CHARTS) {
			const card = document.createElement("div");
			card.className = "chart";
			const series = report.progress[key];
			const last = series[series.length - 1];
			const value = Number.isFinite(last) ? (Number.isInteger(last) ? last : last.toFixed(4)) : "—";
			card.innerHTML = `<div class="head"><span class="name">${label}</span><span class="value">${value}</span></div>`;
			const canvas = document.createElement("canvas");
			canvas.dataset.key = key;
			canvas.addEventListener("mousemove", (event) => drawChart(canvas, event.offsetX * (window.devicePixelRatio || 1)));
			canvas.addEventListener("mouseleave", () => drawChart(canvas, null));
			card.append(canvas);
			charts.append(card);
		}
	}
	for (const canvas of charts.querySelectorAll("canvas")) drawChart(canvas, null);
	document.getElementById("progress-caption").textContent = "Hover a curve for exact values.";
}

document.getElementById("direction").addEventListener("click", (event) => {
	event.target.textContent = event.target.textContent === "↓" ? "↑" : "↓";
	render();
});
// Sortable metrics come from the complex system, not a hardcoded list
const sortSelect = document.getElementById("sort");
for (const name of Object.keys(population[0]?.metrics ?? {})) {
	const option = document.createElement("option");
	option.value = name;
	option.textContent = name;
	sortSelect.append(option);
}

for (const id of ["sort", "count", "valid-only"]) {
	document.getElementById(id).addEventListener("change", render);
}
window.addEventListener("resize", () => { layoutMap(); drawCharts(); });

render();
