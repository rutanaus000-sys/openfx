const $ = (id) => document.getElementById(id);

// Browsing groups. AI effects and Looks share one browser; a group can pull
// from several underlying categories.
const GROUPS = [
  { id: "featured", label: "Featured", match: (i) => i.featured },
  { id: "ai-elements", label: "Fire & Weather", match: (i) => i.kind === "ai" && ["elements", "weather"].includes(i.category) },
  { id: "ai-destruction", label: "Destruction", match: (i) => i.kind === "ai" && i.category === "destruction" },
  { id: "ai-materials", label: "Transform", match: (i) => i.kind === "ai" && ["materials", "transform"].includes(i.category) },
  { id: "ai-scifi", label: "Sci-Fi & Magic", match: (i) => i.kind === "ai" && ["scifi", "magic"].includes(i.category) },
  { id: "ai-camera", label: "Camera Moves", match: (i) => i.kind === "ai" && i.category === "camera" },
  { id: "ai-art", label: "AI Art Styles", match: (i) => i.kind === "ai" && i.category === "art" },
  { id: "color", label: "Color", match: (i) => i.kind === "look" && i.category === "color" },
  { id: "film", label: "Film", match: (i) => i.kind === "look" && i.category === "film" },
  { id: "retro", label: "Retro & VHS", match: (i) => i.kind === "look" && i.category === "retro" },
  { id: "glitch", label: "Glitch", match: (i) => i.kind === "look" && i.category === "glitch" },
  { id: "lens", label: "Lens & Blur", match: (i) => i.kind === "look" && ["lens", "blur"].includes(i.category) },
  { id: "distort", label: "Distort", match: (i) => i.kind === "look" && i.category === "distort" },
  { id: "time", label: "Time & Trails", match: (i) => i.kind === "look" && ["time", "trails"].includes(i.category) },
  { id: "art", label: "Art", match: (i) => i.kind === "look" && i.category === "art" },
  { id: "layout", label: "Frames & Layout", match: (i) => i.kind === "look" && i.category === "layout" },
];
const MAX_LOOKS = 4;

const state = {
  file: null,
  isVideo: false,
  items: [],
  index: [],
  group: "featured",
  effect: null, // AI effect item
  looks: [], // [{item, intensity}]
  polling: null,
};

async function loadCatalog() {
  const { effects, looks } = await (await fetch("/api/catalog")).json();
  state.items = [
    ...effects.map((e) => ({ ...e, kind: "ai" })),
    ...looks.map((l) => ({ ...l, kind: "look" })),
  ];
  state.index = state.items.map(indexItem);
  renderChips();
  renderBrowser();
}

function renderChips() {
  const nav = $("chips");
  nav.replaceChildren();
  for (const group of GROUPS) {
    const n = state.items.filter(group.match).length;
    if (!n) continue;
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip" + (group.id === state.group ? " active" : "");
    chip.textContent = `${group.label} ${n}`;
    chip.onclick = () => {
      state.group = group.id;
      $("search").value = "";
      renderChips();
      renderBrowser();
    };
    nav.append(chip);
  }
}

function renderBrowser() {
  const query = $("search").value;
  const results = searchCatalog(state.index, query);
  const sections = $("sections");
  sections.replaceChildren();
  if (results) {
    $("count").textContent = `${results.length} match${results.length === 1 ? "" : "es"}`;
    sections.append(section(`Results for “${query}”`, results.slice(0, 60)));
    return;
  }
  const group = GROUPS.find((g) => g.id === state.group);
  const items = state.items.filter(group.match);
  $("count").textContent = `${state.items.length} effects`;
  const ai = items.filter((i) => i.kind === "ai");
  const lk = items.filter((i) => i.kind === "look");
  if (ai.length) sections.append(section("AI effects · rendered by LTX on your GPU", ai));
  if (lk.length) sections.append(section("Looks · instant filters, stack up to 4", lk));
}

function section(title, items) {
  const wrap = document.createElement("div");
  const h = document.createElement("h3");
  h.textContent = title;
  const grid = document.createElement("div");
  grid.className = "grid";
  items.forEach((item) => grid.append(card(item)));
  wrap.append(h, grid);
  return wrap;
}

function isSelected(item) {
  return item.kind === "ai" ? state.effect?.id === item.id : state.looks.some((l) => l.item.id === item.id);
}

function card(item) {
  const el = document.createElement("button");
  el.type = "button";
  el.className = `card ${item.kind}` + (isSelected(item) ? " selected" : "");
  if (item.preview) {
    const v = document.createElement("video");
    Object.assign(v, { src: item.preview, muted: true, loop: true, playsInline: true, preload: "none" });
    el.onmouseenter = () => v.play().catch(() => {});
    el.onmouseleave = () => v.pause();
    el.append(v);
  }
  const badge = document.createElement("span");
  badge.className = "badge";
  badge.textContent = item.kind === "ai" ? "AI" : "Look";
  const name = document.createElement("strong");
  name.textContent = item.name;
  const desc = document.createElement("small");
  desc.textContent = item.description;
  el.append(badge, name, desc);
  el.onclick = () => toggle(item);
  return el;
}

function toggle(item) {
  if (item.kind === "ai") {
    state.effect = state.effect?.id === item.id ? null : item;
  } else if (state.looks.some((l) => l.item.id === item.id)) {
    state.looks = state.looks.filter((l) => l.item.id !== item.id);
  } else if (state.looks.length < MAX_LOOKS) {
    state.looks.push({ item, intensity: 1 });
  } else {
    setStatus(`You can stack up to ${MAX_LOOKS} Looks. Remove one first.`);
  }
  renderRecipe();
  renderBrowser();
}

function renderRecipe() {
  const box = $("recipe");
  box.replaceChildren();
  if (!state.effect && !state.looks.length) {
    box.innerHTML = '<p class="muted">Pick one AI effect and up to 4 Looks.</p>';
  }
  if (state.effect) box.append(recipeRow(state.effect, null));
  state.looks.forEach((entry) => box.append(recipeRow(entry.item, entry)));
  $("detail").hidden = !state.effect;
  refresh();
}

function recipeRow(item, lookEntry) {
  const row = document.createElement("div");
  row.className = "recipe-row";
  const label = document.createElement("span");
  label.textContent = `${item.kind === "ai" ? "AI" : "Look"} · ${item.name}`;
  row.append(label);
  if (lookEntry) {
    const slider = document.createElement("input");
    Object.assign(slider, { type: "range", min: 0, max: 1, step: 0.05, value: lookEntry.intensity });
    slider.setAttribute("aria-label", `${item.name} intensity`);
    slider.oninput = () => (lookEntry.intensity = Number(slider.value));
    row.append(slider);
  }
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "remove";
  remove.textContent = "×";
  remove.setAttribute("aria-label", `Remove ${item.name}`);
  remove.onclick = () => toggle(item);
  row.append(remove);
  return row;
}

function setFile(file) {
  if (!file) return;
  const isVideo = file.type.startsWith("video/");
  if (!isVideo && !file.type.startsWith("image/")) return;
  state.file = file;
  state.isVideo = isVideo;
  const url = URL.createObjectURL(file);
  $("preview-img").hidden = isVideo;
  $("preview-video").hidden = !isVideo;
  if (isVideo) $("preview-video").src = url;
  else $("preview-img").src = url;
  $("drop-hint").hidden = true;
  refresh();
}

function refresh() {
  const ready = state.file && (state.effect || state.looks.length) && state.polling === null;
  const conflict = state.isVideo && state.effect?.input === "image";
  $("go").disabled = !ready || conflict;
  if (conflict) setStatus(`${state.effect.name} starts from a photo. Upload an image or remove it.`);
}

function setStatus(text) {
  $("status").textContent = text;
}

async function generate() {
  const form = new FormData();
  form.append("upload", state.file);
  if (state.effect) form.append("effect", state.effect.id);
  if (state.looks.length) {
    form.append("looks", JSON.stringify(state.looks.map((l) => ({ look: l.item.id, intensity: l.intensity }))));
  }
  const detail = $("detail").value.trim();
  if (state.effect && detail) form.append("detail", detail);

  $("result").hidden = true;
  setStatus("Sending…");
  const res = await fetch("/api/generate", { method: "POST", body: form });
  if (!res.ok) {
    setStatus((await res.json()).detail || "Something went wrong.");
    return;
  }
  const job = await res.json();
  state.polling = setInterval(() => poll(job.id), 1500);
  refresh();
}

async function poll(id) {
  const job = await (await fetch(`/api/jobs/${id}`)).json();
  if (job.status === "queued") setStatus(`Waiting in line (#${job.position})…`);
  if (job.status === "running") setStatus(`${job.effect ? "Rendering on the GPU" : "Applying looks"}… ${job.elapsed}s`);
  if (job.status === "done" || job.status === "failed") {
    clearInterval(state.polling);
    state.polling = null;
    refresh();
  }
  if (job.status === "failed") setStatus(`Failed: ${job.error}`);
  if (job.status === "done") {
    setStatus(`Done in ${job.elapsed}s.`);
    $("video").src = job.video;
    $("download").href = job.video;
    $("result").hidden = false;
  }
}

$("file").onchange = (e) => setFile(e.target.files[0]);
$("drop").ondragover = (e) => { e.preventDefault(); $("drop").classList.add("over"); };
$("drop").ondragleave = () => $("drop").classList.remove("over");
$("drop").ondrop = (e) => {
  e.preventDefault();
  $("drop").classList.remove("over");
  setFile(e.dataTransfer.files[0]);
};
$("search").oninput = renderBrowser;
$("go").onclick = generate;
$("copy").onclick = async () => {
  await navigator.clipboard.writeText(new URL($("download").href, location.href).href);
  $("copy").textContent = "Copied";
  setTimeout(() => ($("copy").textContent = "Copy link"), 1500);
};

loadCatalog();
