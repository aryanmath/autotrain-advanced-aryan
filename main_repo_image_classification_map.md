# Main Repo (AutoTrain Advanced) - Image Classification Task & Common Logic: Deep-Dive Map

**Repository:** [https://github.com/huggingface/autotrain-advanced/tree/main](https://github.com/huggingface/autotrain-advanced/tree/main)

---

## 1. Directory & File Structure (Relevant to All Tasks)

- `src/autotrain/app/static/scripts/listeners.js` — Frontend JS (UI logic, **common for all tasks**)
- `src/autotrain/app/ui_routes.py` — UI backend routes (API endpoints, **common for all tasks**)
- `src/autotrain/dataset.py` — Dataset extraction, validation, path updates (**common for all tasks**)
- `src/autotrain/params.py` — Parameter definitions (**common for all tasks**)
- `src/autotrain/tasks.py` — Task registry (**common for all tasks**)
- `src/autotrain/app/templates/index.html` — Main UI template (**common for all tasks**)
- `src/autotrain/logger.py`, `src/autotrain/utils.py`, etc. — Logging, utility functions (**common for all tasks**)
- `src/autotrain/trainers/image_classification/` — Training logic (**image-specific**)
- `src/autotrain/preprocessor/vision.py` — Image preprocessing (**image-specific**)
- `configs/image_classification/` — Config templates (**image-specific**)
- `docs/source/tasks/image_classification_regression.mdx` — Documentation (**image-specific**)

---

## 2. Common/Shared Files: Roles, Integration, and Extension

### A. `src/autotrain/app/static/scripts/listeners.js`
- **Role:** Handles all UI events for task selection, dataset upload, parameter rendering, and training start.
- **Used by:** All tasks (image, text, tabular, ASR, etc.)
- **How it works:**
  - Task dropdown (`task` select): Triggers parameter fetch and UI update for any selected task.
  - Parameter fetch: Calls `/ui/params/{task}/{param_type}` endpoint.
  - Dataset source handling: Shows/hides local/hub upload UI based on task.
  - File upload: Handles zip upload for local datasets (all tasks).
  - Start Training: Gathers all form data and sends to backend.
- **How to extend:** Add new task-specific UI logic with clear checks (e.g., `if (task === 'ASR') { ... }`).

### B. `src/autotrain/app/ui_routes.py`
- **Role:** All backend API endpoints for the UI.
- **Used by:** All tasks.
- **How it works:**
  - Receives the `task` parameter and routes to the correct logic (using `tasks.py`, `params.py`, etc.)
  - Handles parameter fetching, model choices, project creation, dataset upload, and training start for every task.
  - Calls shared dataset handling and parameter validation.
- **How to extend:** Add new endpoints or extend existing ones with task-specific logic using `if task == 'ASR': ...` or by updating the task registry.

### C. `src/autotrain/dataset.py`
- **Role:** Dataset extraction, validation, and path updates for all tasks.
- **Used by:** All tasks.
- **How it works:**
  - Unzips uploaded files, validates CSV/JSON, updates file paths.
  - Has task-specific classes/functions (e.g., `AutoTrainImageClassificationDataset`, `AutoTrainDataset`).
- **How to extend:** Add new dataset handler classes/functions for new tasks, or extend shared logic with task checks.

### D. `src/autotrain/params.py`
- **Role:** Defines parameter schemas for all tasks.
- **Used by:** All tasks.
- **How it works:**
  - Contains parameter definitions for image, text, tabular, ASR, etc.
  - Used by both frontend (for rendering) and backend (for validation).
- **How to extend:** Add new parameter schemas for new tasks.

### E. `src/autotrain/tasks.py`
- **Role:** Task registry and mapping.
- **Used by:** All tasks.
- **How it works:**
  - Maps task names (e.g., `"image-classification"`, `"text-classification"`, `"ASR"`) to the correct preprocessor, trainer, and parameter schema.
  - Central "hub" for task-specific logic.
- **How to extend:** Register new tasks by adding them to the mapping.

### F. `src/autotrain/app/templates/index.html`
- **Role:** Main UI template for all tasks.
- **Used by:** All tasks.
- **How it works:**
  - Renders the task dropdown, dataset upload, parameter forms, etc.
- **How to extend:** Add new UI elements or forms for new tasks.

### G. `src/autotrain/logger.py`, `src/autotrain/utils.py`, etc.
- **Role:** Logging, utility functions, helpers.
- **Used by:** All tasks.
- **How it works:**
  - Provides shared logging, error handling, and utility functions.
- **How to extend:** Add new utilities or logging as needed.

---

## 3. Task-Specific Files (Image Classification Example)

### A. `src/autotrain/trainers/image_classification/`
- **Entrypoint:** `__main__.py` (runs the training loop)
- **Dataset loading:** `dataset.py`
- **Parameter handling:** `params.py`
- **Utilities:** `utils.py`
- **How to extend:** Add new trainers for new tasks in their own folders.

### B. `src/autotrain/preprocessor/vision.py`
- **Role:** Image preprocessing (resize, normalize, augment)
- **How to extend:** Add new preprocessors for new tasks.

### C. `configs/image_classification/`
- **Role:** Config templates (YAML) for image classification
- **How to extend:** Add new config templates for new tasks.

### D. `docs/source/tasks/image_classification_regression.mdx`
- **Role:** Documentation for image classification workflow
- **How to extend:** Add new docs for new tasks.

---

## 4. Workflow: Start Training to Model Training (Step-by-Step, All Tasks)

1. **User selects a task in UI.**
2. **UI fetches parameters, shows upload UI.**
3. **User uploads data, sets params, clicks "Start Training".**
4. **Frontend sends data to `/ui/create_project`.**
5. **Backend extracts data, validates, updates paths, preprocesses as needed.**
6. **Backend loads parameter schema, calls the correct trainer for the task.**
7. **Trainer runs, logs progress, saves model.**
8. **UI shows logs, results, and download options.**

---

## 5. File/Function/Logic Reference Table (Common + Image Example)

| File/Folder                                      | Common/Task-Specific | What/Where/How                                                                 |
|--------------------------------------------------|:--------------------:|---------------------------------------------------------------------------------|
| listeners.js                                     | Common               | UI events, task selection, param fetch, upload, training start                  |
| ui_routes.py                                     | Common               | API endpoints, param/model fetch, project creation, dataset handling, training  |
| dataset.py                                       | Common               | Zip extraction, CSV/image validation, path update, dataset prep                 |
| params.py                                        | Common               | Parameter schemas for all tasks                                                 |
| tasks.py                                         | Common               | Task registry, maps task to preprocessor/trainer/params                        |
| templates/index.html                             | Common               | Main UI template (task dropdown, upload, params)                               |
| logger.py, utils.py, etc.                        | Common               | Logging, helpers                                                               |
| trainers/image_classification/                   | Image-specific        | Training loop, dataset loading, param parsing, utils                           |
| preprocessor/vision.py                           | Image-specific        | Image preprocessing (resize, normalize, augment)                               |
| configs/image_classification/                    | Image-specific        | Config templates (YAML)                                                         |
| docs/source/tasks/image_classification_regression.mdx | Image-specific   | Documentation for workflow, requirements, usage                        |

---

## 6. Best Practices for Extending Common Logic
- **Keep shared logic generic and reusable.**
- **Add task-specific logic only in task-specific files or with clear task checks.**
- **Register new tasks in `tasks.py` and add their schemas in `params.py`.**
- **Update UI and backend only where needed for new tasks.**
- **Document all changes and keep code organized.**

---

## 7. Notes
- This document is a reference for implementing ASR or any new task in the same style.
- All code hygiene, structure, and workflow should match this map for consistency.
- If you need a line-by-line breakdown of any specific file, let me know! 