const $ = (id) => document.getElementById(id);
const state = { file: null, effect: null, polling: null };

async function loadEffects() {
  const effects = await (await fetch("/api/effects")).json();
  const grid = $("effects");
  for (const fx of effects) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "card";
    card.dataset.id = fx.id;
    card.innerHTML = `${fx.preview ? `<video src="${fx.preview}" autoplay muted loop playsinline></video>` : ""}
      <strong></strong><small></small>`;
    card.querySelector("strong").textContent = fx.name;
    card.querySelector("small").textContent = fx.description;
    card.onclick = () => {
      state.effect = fx.id;
      grid.querySelectorAll(".card").forEach((c) => c.classList.toggle("selected", c === card));
      refresh();
    };
    grid.append(card);
  }
}

function setFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  state.file = file;
  $("preview").src = URL.createObjectURL(file);
  $("preview").hidden = false;
  $("drop-hint").hidden = true;
  refresh();
}

function refresh() {
  $("go").disabled = !(state.file && state.effect) || state.polling !== null;
}

function setStatus(text) {
  $("status").textContent = text;
}

async function generate() {
  const form = new FormData();
  form.append("effect", state.effect);
  form.append("image", state.file);
  const detail = $("detail").value.trim();
  if (detail) form.append("detail", detail);

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
  if (job.status === "running") setStatus(`Rendering on the GPU… ${job.elapsed}s`);
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
$("go").onclick = generate;
$("copy").onclick = async () => {
  await navigator.clipboard.writeText(new URL($("download").href, location.href).href);
  $("copy").textContent = "Copied";
  setTimeout(() => ($("copy").textContent = "Copy link"), 1500);
};

loadEffects();
