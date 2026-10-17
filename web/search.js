// Search over effects and Looks: names, tags, synonyms, descriptions, plus a
// small "vibe" vocabulary so plain-language queries ("90s home video",
// "dreamy", "scary") land on the right effects. Runs entirely in the browser.

const VIBES = {
  old: ["film", "retro", "sepia", "super-8", "silent-film", "vhs", "film-grain", "faded-blacks"],
  vintage: ["film", "retro", "sepia", "super-8", "film-35mm", "light-leaks", "polaroid"],
  retro: ["retro", "vhs", "crt-scanlines", "super-8", "posterize"],
  "90s": ["vhs", "camcorder-overlay", "tracking-errors", "chroma-bleed", "interlace"],
  "80s": ["vhs", "crt-scanlines", "chroma-bleed", "hue-cycle"],
  "home video": ["camcorder-overlay", "vhs", "super-8"],
  nostalgic: ["super-8", "light-leaks", "film-grain", "faded-blacks", "polaroid"],
  dreamy: ["soft-focus", "bloom", "halation", "light-leaks", "split-toning"],
  cinematic: ["teal-orange", "letterbox", "film-35mm", "anamorphic-streaks", "bleach-bypass"],
  movie: ["teal-orange", "letterbox", "film-35mm", "anamorphic-streaks"],
  scary: ["invert", "thermal", "glitch-bursts", "silent-film", "crushed-blacks", "ghosting"],
  horror: ["invert", "silent-film", "exposure-flicker", "crushed-blacks", "ghosting"],
  creepy: ["ghosting", "exposure-flicker", "invert", "vertical-roll"],
  trippy: ["kaleidoscope", "hue-cycle", "vortex", "color-trails", "droste", "solarize"],
  psychedelic: ["kaleidoscope", "hue-cycle", "solarize", "color-trails"],
  glitchy: ["glitch", "rgb-split", "pixel-sort", "slice-shift", "glitch-bursts"],
  broken: ["glitch-bursts", "digital-tear", "macroblock", "tape-dropout"],
  cyberpunk: ["rgb-split", "glitch-bursts", "hologram", "teal-orange", "crt-scanlines"],
  futuristic: ["hologram", "scifi", "rgb-split"],
  magic: ["scifi", "hologram", "bloom", "light-leaks"],
  fire: ["on-fire", "film-burn", "thermal"],
  hot: ["on-fire", "thermal", "heat"],
  cold: ["freeze", "day-for-night", "monochrome-tint"],
  winter: ["freeze"],
  comic: ["comic", "halftone", "cel-shade", "ink-outline"],
  cartoon: ["cel-shade", "comic", "posterize", "ink-outline"],
  drawing: ["pencil-sketch", "ink-outline", "comic"],
  painting: ["oil-paint", "pointillism", "watercolor"],
  art: ["art"],
  arcade: ["crt-curvature", "crt-scanlines", "pixelate", "bit-crush"],
  pixel: ["pixelate", "bit-crush", "mosaic", "ascii"],
  "8-bit": ["pixelate", "bit-crush", "posterize"],
  smooth: ["soft-focus", "gaussian-blur"],
  dramatic: ["bleach-bypass", "crushed-blacks", "vignette", "black-and-white"],
  moody: ["bleach-bypass", "crushed-blacks", "teal-orange", "vignette"],
  warm: ["sepia", "light-leaks", "teal-orange"],
  disappear: ["disintegrate", "melt"],
  vanish: ["disintegrate"],
  destroy: ["destruction", "disintegrate"],
  instagram: ["faded-blacks", "split-toning", "light-leaks", "polaroid"],
  tiktok: ["featured"],
  viral: ["featured"],
};

const STOP = new Set(["a", "an", "the", "it", "make", "look", "like", "with", "and", "of", "my", "me", "into", "in", "to", "style", "effect", "video", "please"]);

function tokens(text) {
  return text.toLowerCase().replace(/[^a-z0-9\- ]+/g, " ").split(/\s+/).filter((t) => t && !STOP.has(t));
}

function close(a, b) {
  // true when a and b differ by at most one edit (typo tolerance for longer words)
  if (Math.abs(a.length - b.length) > 1 || a.length < 4) return false;
  let i = 0, j = 0, edits = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) { i++; j++; continue; }
    if (++edits > 1) return false;
    if (a.length > b.length) i++;
    else if (b.length > a.length) j++;
    else { i++; j++; }
  }
  return edits + (a.length - i) + (b.length - j) <= 1;
}

function fieldScore(query, words, weight) {
  let score = 0;
  for (const w of words) {
    if (w === query) score = Math.max(score, weight * 1.5);
    else if (w.startsWith(query)) score = Math.max(score, weight);
    else if (close(query, w)) score = Math.max(score, weight * 0.6);
  }
  return score;
}

function indexItem(item) {
  return {
    item,
    id: item.id,
    category: item.category,
    name: tokens(item.name + " " + item.id.replace(/-/g, " ")),
    tags: tokens([...(item.tags || []), ...(item.synonyms || [])].join(" ")),
    text: tokens((item.description || "") + " " + (item.category || "")),
  };
}

function vibeTargets(query) {
  const q = query.toLowerCase();
  const targets = new Set();
  for (const [phrase, ids] of Object.entries(VIBES)) {
    if (q.includes(phrase)) ids.forEach((id) => targets.add(id));
  }
  return targets;
}

// Returns items sorted by relevance, or null for an empty query.
function searchCatalog(index, query) {
  const qTokens = tokens(query);
  if (!qTokens.length) return null;
  const vibes = vibeTargets(query);
  const results = [];
  for (const entry of index) {
    let score = 0;
    for (const t of qTokens) {
      score += Math.max(fieldScore(t, entry.name, 6), fieldScore(t, entry.tags, 4), fieldScore(t, entry.text, 1.5));
    }
    if (vibes.has(entry.id)) score += 5;
    if (vibes.has(entry.category)) score += 2.5;
    if (vibes.has("featured") && entry.item.featured) score += 4;
    if (score > 0) results.push({ entry, score });
  }
  results.sort((a, b) => b.score - a.score);
  return results.map((r) => r.entry.item);
}
