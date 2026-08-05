"use strict";

const cameraLabels = {
  operation_backview: "Back",
  operation_leftview: "Left",
  operation_rightview: "Right",
  operation_topview: "Top",
};

const movementVectors = {
  forward: [0, 0, -1],
  backward: [0, 0, 1],
  left: [-1, 0, 0],
  right: [1, 0, 0],
  down: [0, -1, 0],
  up: [0, 1, 0],
};

const keyDirections = {
  w: "forward",
  s: "backward",
  a: "left",
  d: "right",
  q: "down",
  e: "up",
};

let state = null;
let activeCamera = "operation_backview";
let movementSpeed = 0.01;
let requestChain = Promise.resolve();
let toastTimer = null;
let dragStart = null;
const imageUrls = new Map();

const elements = {
  app: document.querySelector("#app"),
  taskContext: document.querySelector("#task-context"),
  progressCount: document.querySelector("#progress-count"),
  progressBar: document.querySelector("#progress-bar"),
  taskList: document.querySelector("#task-list"),
  cameraTabs: document.querySelector("#camera-tabs"),
  mainPreview: document.querySelector("#main-preview"),
  previewLoading: document.querySelector("#preview-loading"),
  imageFrame: document.querySelector(".image-frame"),
  activeCameraName: document.querySelector("#active-camera-name"),
  thumbnailStrip: document.querySelector("#thumbnail-strip"),
  demoSelect: document.querySelector("#demo-select"),
  demoMeta: document.querySelector("#demo-meta"),
  frameSlider: document.querySelector("#frame-slider"),
  frameNumber: document.querySelector("#frame-number"),
  frameOutput: document.querySelector("#frame-output"),
  posePosition: document.querySelector("#pose-position"),
  poseQuaternion: document.querySelector("#pose-quaternion"),
  resetCamera: document.querySelector("#reset-camera"),
  speedControl: document.querySelector("#speed-control"),
  saveState: document.querySelector("#save-state"),
  confirmTask: document.querySelector("#confirm-task"),
  toast: document.querySelector("#toast"),
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

function enqueueMutation(path, body, camerasToRefresh = null) {
  requestChain = requestChain
    .catch(() => undefined)
    .then(async () => {
      const nextState = await api(path, {
        method: "POST",
        body: JSON.stringify(body),
      });
      applyState(nextState);
      await refreshImages(
        nextState.revision,
        camerasToRefresh || nextState.camera_names,
      );
    })
    .catch((error) => showToast(error.message, true));
  return requestChain;
}

function applyState(nextState) {
  state = nextState;
  elements.app.setAttribute("aria-busy", "false");
  const task = state.current_task;
  elements.taskContext.textContent = `${task.suite} / ${task.name.replaceAll("_", " ")}`;
  elements.progressCount.textContent = `${state.progress.confirmed} / ${state.progress.total}`;
  elements.progressBar.style.width = `${(state.progress.confirmed / state.progress.total) * 100}%`;
  renderTaskList();
  renderCameraControls();
  renderSceneControls();
  renderPose();
  elements.saveState.classList.toggle("dirty", state.dirty);
  elements.saveState.querySelector("strong").textContent = state.dirty
    ? "Unsaved changes"
    : state.current_task.confirmed
      ? "Confirmed"
      : "Unchanged";
  elements.confirmTask.disabled = state.completed;
  elements.confirmTask.querySelector("span").textContent = state.completed
    ? "All 40 tasks confirmed"
    : "Confirm and next task";
}

function renderTaskList() {
  const groups = new Map();
  state.tasks.forEach((task) => {
    if (!groups.has(task.suite)) groups.set(task.suite, []);
    groups.get(task.suite).push(task);
  });
  elements.taskList.replaceChildren();
  groups.forEach((tasks, suite) => {
    const heading = document.createElement("div");
    heading.className = "suite-heading";
    heading.textContent = suite.replace("libero_", "LIBERO ");
    elements.taskList.append(heading);
    tasks.forEach((task) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `task-item${task.current ? " active" : ""}${task.confirmed ? " confirmed" : ""}`;
      button.dataset.taskId = task.id;
      button.innerHTML = `<span class="task-number">${String(task.index + 1).padStart(2, "0")}</span><span class="task-name"></span><span class="task-status" aria-hidden="true"></span>`;
      button.querySelector(".task-name").textContent = task.name.replaceAll("_", " ");
      button.addEventListener("click", () => loadTask(task.id));
      elements.taskList.append(button);
    });
  });
  document.querySelector(".task-item.active")?.scrollIntoView({ block: "nearest" });
}

function renderCameraControls() {
  if (!state.camera_names.includes(activeCamera)) activeCamera = state.camera_names[0];
  elements.cameraTabs.replaceChildren();
  elements.thumbnailStrip.replaceChildren();
  state.camera_names.forEach((camera) => {
    const tab = document.createElement("button");
    tab.type = "button";
    tab.className = `camera-tab${camera === activeCamera ? " active" : ""}`;
    tab.role = "tab";
    tab.setAttribute("aria-selected", String(camera === activeCamera));
    tab.textContent = cameraLabels[camera];
    tab.addEventListener("click", () => selectCamera(camera));
    elements.cameraTabs.append(tab);

    const thumbnail = document.createElement("button");
    thumbnail.type = "button";
    thumbnail.className = `thumbnail${camera === activeCamera ? " active" : ""}`;
    thumbnail.dataset.camera = camera;
    thumbnail.innerHTML = `<img alt="${cameraLabels[camera]} camera thumbnail"><span>${cameraLabels[camera]}</span>`;
    thumbnail.addEventListener("click", () => selectCamera(camera));
    elements.thumbnailStrip.append(thumbnail);
  });
  elements.activeCameraName.textContent = cameraLabels[activeCamera];
  updateVisibleImages();
}

function renderSceneControls() {
  elements.demoSelect.replaceChildren();
  state.demo.keys.forEach((key, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${key} (${state.demo.lengths[index]} frames)`;
    option.selected = index === state.demo.index;
    elements.demoSelect.append(option);
  });
  elements.demoMeta.textContent = `Demo ${state.demo.index + 1} / ${state.demo.keys.length}`;
  const maxFrame = Math.max(0, state.demo.frame_count - 1);
  elements.frameSlider.max = String(maxFrame);
  elements.frameSlider.value = String(state.demo.frame_index);
  elements.frameNumber.max = String(maxFrame);
  elements.frameNumber.value = String(state.demo.frame_index);
  elements.frameOutput.textContent = `${state.demo.frame_index + 1} / ${state.demo.frame_count}`;
}

function renderPose() {
  const pose = state.cameras[activeCamera];
  elements.posePosition.textContent = pose.position.map(formatNumber).join("  ");
  elements.poseQuaternion.textContent = pose.quaternion_wxyz.map(formatNumber).join("  ");
}

function formatNumber(value) {
  return Number(value).toFixed(4);
}

function selectCamera(camera) {
  activeCamera = camera;
  renderCameraControls();
  renderPose();
}

async function refreshImages(revision, cameras = state.camera_names) {
  const uniqueCameras = [...new Set(cameras)];
  if (uniqueCameras.includes(activeCamera)) {
    elements.previewLoading.classList.remove("hidden");
  }
  await Promise.all(
    uniqueCameras.map(async (camera) => {
      const response = await fetch(`/api/render/${camera}.jpg?revision=${revision}`, {
        cache: "no-store",
      });
      if (!response.ok) throw new Error(`Render failed: ${camera}`);
      const blob = await response.blob();
      if (!state || state.revision !== revision) return;
      const previousUrl = imageUrls.get(camera);
      if (previousUrl) URL.revokeObjectURL(previousUrl);
      imageUrls.set(camera, URL.createObjectURL(blob));
      updateVisibleImages();
    }),
  );
  if (state && state.revision === revision) {
    updateVisibleImages();
    elements.previewLoading.classList.add("hidden");
  }
}

function updateVisibleImages() {
  const mainUrl = imageUrls.get(activeCamera);
  if (mainUrl) elements.mainPreview.src = mainUrl;
  elements.thumbnailStrip.querySelectorAll(".thumbnail").forEach((thumbnail) => {
    const url = imageUrls.get(thumbnail.dataset.camera);
    if (url) thumbnail.querySelector("img").src = url;
  });
}

function loadTask(taskId) {
  if (state.dirty && !window.confirm("Discard unsaved camera changes?")) return;
  enqueueMutation("/api/task", { task_id: taskId });
}

function move(direction, multiplier = 1) {
  const camera = activeCamera;
  const vector = movementVectors[direction].map((value) => value * movementSpeed * multiplier);
  enqueueMutation("/api/adjust", {
    camera,
    translation: vector,
    rotation_degrees: [0, 0, 0],
  }, [camera]);
}

function showToast(message, isError = false) {
  clearTimeout(toastTimer);
  elements.toast.textContent = message;
  elements.toast.classList.toggle("error", isError);
  elements.toast.classList.add("visible");
  toastTimer = setTimeout(() => elements.toast.classList.remove("visible"), 2600);
}

elements.demoSelect.addEventListener("change", (event) => {
  enqueueMutation("/api/demo", { demo_index: Number(event.target.value) });
});

function commitFrame(value) {
  const frame = Math.max(0, Math.min(Number(value), Number(elements.frameSlider.max)));
  enqueueMutation("/api/frame", { frame_index: frame });
}

elements.frameSlider.addEventListener("change", (event) => commitFrame(event.target.value));
elements.frameNumber.addEventListener("change", (event) => commitFrame(event.target.value));

elements.resetCamera.addEventListener("click", () => {
  const camera = activeCamera;
  enqueueMutation("/api/camera/reset", { camera }, [camera]);
});

elements.speedControl.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-speed]");
  if (!button) return;
  movementSpeed = Number(button.dataset.speed);
  elements.speedControl.querySelectorAll("button").forEach((item) => {
    item.classList.toggle("active", item === button);
  });
});

document.querySelector(".nudge-grid").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-move]");
  if (button) move(button.dataset.move);
});

elements.confirmTask.addEventListener("click", async () => {
  await enqueueMutation("/api/confirm", {});
  if (state.completed) showToast("Calibration complete: 40 / 40");
  else showToast("Task cameras confirmed");
});

window.addEventListener("keydown", (event) => {
  if (event.repeat || event.ctrlKey || event.metaKey || event.altKey) return;
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
  const direction = keyDirections[event.key.toLowerCase()];
  if (!direction) return;
  event.preventDefault();
  move(direction, event.shiftKey ? 5 : 1);
});

elements.imageFrame.addEventListener("wheel", (event) => {
  event.preventDefault();
  move(event.deltaY < 0 ? "forward" : "backward", Math.min(Math.abs(event.deltaY) / 80, 3));
}, { passive: false });

elements.imageFrame.addEventListener("pointerdown", (event) => {
  dragStart = { x: event.clientX, y: event.clientY };
  elements.imageFrame.setPointerCapture(event.pointerId);
  elements.imageFrame.classList.add("dragging");
});

elements.imageFrame.addEventListener("pointerup", (event) => {
  if (!dragStart) return;
  const dx = event.clientX - dragStart.x;
  const dy = event.clientY - dragStart.y;
  dragStart = null;
  elements.imageFrame.classList.remove("dragging");
  if (Math.abs(dx) + Math.abs(dy) < 2) return;
  const camera = activeCamera;
  enqueueMutation("/api/adjust", {
    camera,
    translation: [0, 0, 0],
    rotation_degrees: [dy * 0.16, -dx * 0.16, 0],
  }, [camera]);
});

elements.imageFrame.addEventListener("pointercancel", () => {
  dragStart = null;
  elements.imageFrame.classList.remove("dragging");
});

window.addEventListener("beforeunload", (event) => {
  if (!state?.dirty) return;
  event.preventDefault();
  event.returnValue = "";
});

api("/api/bootstrap")
  .then((initialState) => {
    applyState(initialState);
    return refreshImages(initialState.revision);
  })
  .catch((error) => {
    elements.previewLoading.textContent = "Unable to load";
    showToast(error.message, true);
  });
