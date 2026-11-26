"use strict";

const ui = Object.fromEntries(
  ["clip-title", "clip-info", "gallery-link", "status", "review", "choices", "videos", "play", "scrub", "time", "legend",
   "plot", "duration", "segments"].map(id => [id, document.getElementById(id)])
);
const mobile = window.matchMedia("(max-width: 690px)");
const colors = {original: "#62cefc", retime: "#74edbf", uncertain: "#efbd6f", generated: "#ce9df8"};
const labels = {"push-in": "Push-in", pullback: "Pullback", pan: "Pan", tilt: "Tilt", slide: "Slide",
                hold: "Hold", cut: "Cut", unknown: "Low confidence", mixed: "Mixed motion"};
let report;
let selected = 0;
let time = 0;
let playing = false;
const views = [];

function stamp(seconds, precision = 0) {
  const ticks = Math.floor(seconds * 10 ** precision + 1e-7);
  const integer = Math.floor(ticks / 10 ** precision);
  const tenths = precision ? "." + String(ticks % 10 ** precision).padStart(precision, "0") : "";
  return Math.floor(integer / 60) + ":" + String(integer % 60).padStart(2, "0") + tenths;
}

function paint() {
  const canvas = ui.plot;
  const {width, height} = canvas.getBoundingClientRect();
  if (!width || !height) return;
  const ratio = devicePixelRatio || 1;
  const w = Math.round(width * ratio), h = Math.round(height * ratio);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const end = height - 14;
  const at = t => t * width / report.duration;
  ctx.strokeStyle = "#2a394a";
  for (let offset = 1; offset <= 3; offset++) {
    const y = end * offset / 3;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  for (const segment of report.segments) {
    if (segment.reliable) continue;
    ctx.fillStyle = segment.kind === "cut" ? colors.uncertain : "#efbd6f40";
    ctx.fillRect(at(segment.start), 0, Math.max(segment.kind === "cut" ? 2 : 1,
                                               at(segment.end - segment.start)), end);
  }
  const hasOriginal = report.variants.some(v => v.id === "original");
  const hasRetime = report.variants.some(v => v.id !== "original");
  const max = Math.max(.001, ...report.series.map(point => Math.max(hasOriginal ? point[2] : 0,
                                                                    hasRetime ? point[3] : 0)));
  const drawSeries = (coordinate, speed, color) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (const [index, row] of report.series.entries()) {
      const x = at(row[coordinate]), y = end - row[speed] / max * (end - 9);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  };
  if (hasOriginal) drawSeries(0, 2, colors.original);
  if (hasRetime) drawSeries(1, 3, colors.retime);
  ctx.fillStyle = colors.generated;
  for (const frame of report.generated) ctx.fillRect(at(frame / report.fps), end + 4, 1.6, 9);
  ctx.fillStyle = "#f2f7fc";
  ctx.fillRect(Math.min(width - 1, at(time)), 0, 1, height);
}

function display() {
  ui.scrub.value = time;
  ui.time.value = stamp(time, 1) + " / " + stamp(report.duration, 1);
  paint();
}

function move(video, target) {
  if (!Number.isFinite(video.duration)) return;
  video.currentTime = Math.min(target, Math.max(0, video.duration - .001));
}

function outputFrame(position) {
  const last = Math.max(0, Math.round(report.duration * report.fps) - 1);
  return Math.max(0, Math.min(last, Math.floor(position * report.fps + 1e-7)));
}

function sharedSeekTime() {
  // Stay inside the paused frame even when a video loads late or becomes visible.
  return playing ? time : (outputFrame(time) + .5) / report.fps;
}

function seek(target) {
  time = Math.min(report.duration, Math.max(0, target));
  for (const {video} of views) move(video, sharedSeekTime());
  display();
}

function visibleViews() {
  return mobile.matches ? [views[selected]] : views;
}

function stop(atEnd = false) {
  playing = false;
  ui.play.textContent = "Play";
  const leader = views[selected].video;
  const position = !leader.seeking && Number.isFinite(leader.duration) ? leader.currentTime : time;
  time = atEnd ? report.duration : outputFrame(position) / report.fps;
  for (const {video} of views) {
    video.pause();
    move(video, sharedSeekTime());
  }
  display();
}

function start() {
  if (time >= report.duration - 1 / report.fps) seek(0);
  const leader = views[selected].video;
  playing = true;
  ui.play.textContent = "Pause";
  for (const {video} of visibleViews()) {
    if (Math.abs(video.currentTime - time) > .09) move(video, time);
    video.play().catch(() => {
      if (video !== leader || video !== views[selected].video || !playing) return;
      stop();
      ui.status.hidden = false;
      ui.status.textContent = "The browser could not start playback. Try Play again.";
    });
  }
}

function choose(index) {
  selected = index;
  for (const [i, {figure, button, video}] of views.entries()) {
    figure.classList.toggle("selected", i === index);
    button.setAttribute("aria-pressed", i === index);
    if (mobile.matches && i !== index) video.pause();
  }
  move(views[index].video, sharedSeekTime());
  if (playing) start();
  display();
}

function update() {
  if (playing) {
    const leader = views[selected].video;
    if (leader.ended) {
      stop(true);
    } else if (!leader.seeking && !leader.paused) {
      time = leader.currentTime;
      for (const {video} of visibleViews()) {
        if (video !== leader && !video.seeking && Math.abs(video.currentTime - time) > 1 / report.fps) move(video, time);
      }
      display();
    }
  }
  requestAnimationFrame(update);
}

function element(tag, value) {
  const result = document.createElement(tag);
  if (value) result.textContent = value;
  return result;
}

async function init() {
  const response = await fetch("api/preview.json");
  if (!response.ok) throw new Error("Preview data could not be loaded.");
  report = await response.json();
  if (report.title) {
    ui["clip-title"].textContent = report.title;
    document.title = report.title;
    ui["gallery-link"].hidden = false;
  }
  ui["clip-info"].textContent = stamp(report.duration, 1) + " · " + Number(report.fps.toFixed(3)) + " fps";
  for (const [index, variant] of report.variants.entries()) {
    const figure = element("figure");
    const caption = element("figcaption", variant.name);
    const video = element("video");
    video.src = variant.url;
    video.muted = true;
    video.defaultMuted = true;
    video.playsInline = true;
    video.preload = "metadata";
    video.setAttribute("aria-label", variant.name);
    video.addEventListener("loadedmetadata", () => move(video, sharedSeekTime()));
    video.addEventListener("error", () => {
      ui.status.hidden = false;
      ui.status.textContent = variant.name + " could not be loaded.";
    });
    const count = variant.counts;
    figure.append(caption, video, element("p", count.original_frames + " original frames" +
                  (variant.id === "interpolated" ? " · " + count.generated_frames + " generated" : "")));
    ui.videos.append(figure);
    const button = element("button", variant.name);
    button.type = "button";
    button.addEventListener("click", () => choose(index));
    ui.choices.append(button);
    views.push({figure, button, video});
  }
  const legend = [
    ["original", report.variants.find(v => v.id === "original")?.name,
     report.variants.some(v => v.id === "original")],
    ["retime", "Retime", report.variants.some(v => v.id !== "original")],
    ["uncertain", "Cuts / low confidence", true], ["generated", "Generated frames", report.generated.length > 0]
  ];
  for (const [key, label, show] of legend) {
    if (!show) continue;
    const row = element("span");
    const swatch = element("i");
    swatch.className = key;
    row.append(swatch, label);
    ui.legend.append(row);
  }
  for (const item of report.segments) {
    const end = item.kind === "cut" ? "" : "–" + stamp(item.end, 2);
    const status = item.retimed ? " · retimed" : item.reliable ? "" : " · original timing";
    ui.segments.append(element("li", stamp(item.start, 2) + end + " · " + labels[item.kind] + status));
  }
  ui.status.hidden = true;
  ui.review.hidden = false;
  ui.duration.textContent = stamp(report.duration, 1);
  ui.scrub.max = report.duration;
  ui.scrub.addEventListener("input", () => seek(Number(ui.scrub.value)));
  ui.play.addEventListener("click", () => playing ? stop() : start());
  ui.plot.addEventListener("pointerdown", event => {
    const rect = ui.plot.getBoundingClientRect();
    seek((event.clientX - rect.left) / rect.width * report.duration);
  });
  new ResizeObserver(paint).observe(ui.plot);
  mobile.addEventListener("change", () => choose(selected));
  choose(0);
  requestAnimationFrame(update);
}

init().catch(error => { ui.status.textContent = error.message; });
