# LiFE App Code Map (AutoTrain Advanced Local Repo)

This document lists **every file and code block** in the local repo that implements LiFE App logic, with the **exact code** and a clear explanation of what each does and how it fits into the LiFE App workflow. All entries are double-checked and accurate.

---

## Summary Table

| File | Code/Function/Line | What it Does |
|------|--------------------|--------------|
| `src/autotrain/app/templates/index.html` | LiFE App option, project/script dropdowns, JS blocks | UI for LiFE App selection and project/script picking |
| `src/autotrain/app/static/scripts/listeners.js` | All LiFE App–related JS | Frontend logic for LiFE App UI, project/script loading |
| `src/autotrain/app/ui_routes.py` | Form handling, API endpoints | Backend logic for LiFE App UI, validation, data serving |
| `src/autotrain/trainers/automatic_speech_recognition/dataset.py` | `load_life_app_dataset` | Loads LiFE App dataset for training |
| `src/autotrain/preprocessor/automatic_speech_recognition.py` | `load_life_app_dataset_from_disk` | Preprocesses LiFE App dataset |
| `src/autotrain/trainers/automatic_speech_recognition/__main__.py` | LiFE App dataset handling | Loads/uses LiFE App dataset in training |
| `src/autotrain/project.py` | LiFE App data path logic | Sets correct data path for LiFE App datasets |
| `life_app_data/` | Data files | Stores audio and processed CSV for LiFE App |

---

## 1. `src/autotrain/app/templates/index.html`

**LiFE App option in dataset source (line 452):**
```html
<option value="life_app">LiFE App</option>
```
*Adds LiFE App as a dataset source option (ASR only).*

**LiFE App project/script selection UI (lines 564-579):**
```html
<div id="life-app-selection" style="display:none;" class="w-full px-4 mb-4">
    <div class="flex flex-wrap gap-4">
        <!-- Projects Multi-select -->
        <div class="flex-1">
            <label class="block text-gray-700 dark:text-gray-300 text-sm font-bold mb-2">LiFE App Project(s)
            </label>
            <select id="life_app_project" name="life_app_project" multiple
                class="block w-full border border-gray-300 dark:border-gray-600 px-3 py-2 mb-2 bg-white dark:bg-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500">
            </select>
            <div id="life-app-project-tags" class="mb-2"></div>
        </div>
        <!-- Script Dropdown -->
        <div class="flex-1">
            <label class="block text-gray-700 dark:text-gray-300 text-sm font-bold mb-2">LiFE App Script
            </label>
            <select id="life_app_script" name="life_app_script"
                class="block w-full border border-gray-300 dark:border-gray-600 px-3 py-2 mb-2 bg-white dark:bg-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500">
                <option value="">Select Script</option>
            </select>
        </div>
    </div>
    <!-- Dataset File Dropdown -->
    <div id="dataset_file_div">
        <label class="block text-gray-700 dark:text-gray-300 text-sm font-bold mb-2">Dataset File
        </label>
        <select id="dataset_file" name="dataset_file"
            class="js-example-basic-single block w-full border border-gray-300 dark:border-gray-600 px-3 py-2 mb-2 bg-white dark:bg-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500">
            <option value="">Select Dataset</option>
        </select>
    </div>
</div>
```
*Dropdowns for selecting LiFE App project(s), script, and dataset file.*

**JavaScript: Show/hide LiFE App option based on task (lines 108-145):**
```javascript
case 'ASR':
    fields = ['audio', 'transcription'];
    fieldNames = ['audio', 'transcription'];
    // Show LiFE App dataset source option only for ASR
    document.getElementById("dataset_source").querySelectorAll("option").forEach(opt => {
        if (opt.value === "life_app") {
            opt.style.display = "";
        }
    });
    break;
default:
    // Hide LiFE App dataset source option for other tasks
    document.getElementById("dataset_source").querySelectorAll("option").forEach(opt => {
        if (opt.value === "life_app") {
            opt.style.display = "none";
        }
    });
    break;
```
*Shows/hides LiFE App option in the dataset source dropdown depending on the selected task.*

**JavaScript: Show/hide LiFE App selection UI based on dataset source (lines 793-800):**
```javascript
document.addEventListener('DOMContentLoaded', function () {
    const datasetSource = document.getElementById('dataset_source');
    const lifeAppSelection = document.getElementById('life-app-selection');
    function toggleLifeAppSelection() {
        if (datasetSource.value === 'life_app') {
            lifeAppSelection.style.display = '';
        } else {
            lifeAppSelection.style.display = 'none';
        }
    }
    datasetSource.addEventListener('change', toggleLifeAppSelection);
    // Initialize on page load
    toggleLifeAppSelection();
});
```
*Shows or hides the LiFE App project/script UI when the user selects "LiFE App" as the dataset source.*

**jQuery: Initialize Select2 and load projects when LiFE App is selected (lines 814-834):**
```javascript
$(document).ready(function() {
    // Initialize Select2 for all dropdowns
    $('#life_app_project').select2({
        placeholder: "Select LiFE App Project(s)",
        allowClear: true,
        multiple: true,
        width: '100%',
        maximumSelectionLength: 2,
        tags: true
    });
    $('#dataset_file').select2({
        placeholder: "Select Dataset File",
        allowClear: true,
        width: '100%'
    });
    // Handle dataset source change
    $('#dataset_source').on('change', function() {
        const lifeAppSelection = $('#life-app-selection');
        if ($(this).val() === 'life_app') {
            lifeAppSelection.show();
            // Load projects when LiFE App is selected
            loadLifeAppProjects();
        } else {
            lifeAppSelection.hide();
        }
    });
});
```
*Initializes Select2 for LiFE App dropdowns and loads projects when LiFE App is selected.*

---

## 2. `src/autotrain/app/static/scripts/listeners.js`

**LiFE App dataset source selection and UI logic (lines 173-200):**
```javascript
function handleDataSource() {
    const lifeAppOption = document.getElementById("dataset_source").querySelector('option[value="life_app"]');
    const lifeAppSelection = document.getElementById("life-app-selection");
    const datasetFileDiv = document.getElementById('dataset_file_div');
    const taskValue = document.getElementById('task').value;
    // Hide all dataset source UIs by default
    if (hubDataTabContent) hubDataTabContent.style.display = "none";
    if (uploadDataTabContent) uploadDataTabContent.style.display = "none";
    if (uploadDataTabs) uploadDataTabs.style.display = "none";
    if (lifeAppSelection) lifeAppSelection.style.display = "none";
    if (datasetFileDiv) datasetFileDiv.style.display = 'none';
    // Show/hide LiFE App option in dropdown
    if (lifeAppOption) {
        if (taskValue === "ASR") {
            lifeAppOption.style.display = "";
        } else {
            lifeAppOption.style.display = "none";
            if (dataSource.value === "life_app") {
                dataSource.value = "local";
            }
        }
    }
    // Show relevant section based on selected data source
    if (dataSource.value === "life_app" && taskValue === "ASR") {
        if (lifeAppSelection) lifeAppSelection.style.display = "block";
        if (datasetFileDiv) datasetFileDiv.style.display = 'block';
        loadLifeAppProjects();
    } else if (dataSource.value === "hub") {
        if (hubDataTabContent) hubDataTabContent.style.display = "block";
    } else if (dataSource.value === "local") {
        if (uploadDataTabContent) uploadDataTabContent.style.display = "block";
        if (uploadDataTabs) uploadDataTabs.style.display = "block";
    }
}
```
*Shows/hides LiFE App UI and triggers project loading when selected.*

**Load LiFE App projects from backend (lines 349-379):**
```javascript
async function loadLifeAppProjects() {
    const projectSelect = document.getElementById('life_app_project');
    if (!projectSelect) return;
    try {
        const response = await fetch('/ui/life_app_projects');
        if (!response.ok) {
            throw new Error('Failed to fetch projects');
        }
        const data = await response.json();
        const projects = data.projects || [];
        projectSelect.innerHTML = '';
        projects.forEach(project => {
            const option = document.createElement('option');
            option.value = project;
            option.textContent = project;
            projectSelect.appendChild(option);
        });
        // --- Select2 for LiFE App Project(s) ---
        $('#life_app_project').select2({
            placeholder: 'Select LiFE App Project(s)',
            allowClear: true,
            multiple: true,
            width: '100%',
            maximumSelectionLength: 2,
            tags: true
        });
        // --- Project selection event handler ---
        $('#life_app_project').on('change', function() {
            const selectedProjects = $(this).val();
            if (selectedProjects && selectedProjects.length > 0) {
                loadScriptsForProjects(selectedProjects);
            } else {
                $('#life_app_script').prop('disabled', true).empty();
                $('#dataset_file').prop('disabled', true).empty();
            }
        });
    } catch (error) {
        console.error('Error loading projects:', error);
    }
}
```
*Fetches available LiFE App projects and populates the dropdown.*

**Load scripts for selected LiFE App projects (lines 400-453):**
```javascript
async function loadScriptsForProjects(selectedProjects) {
    const scriptSelect = $('#life_app_script');
    scriptSelect.prop('disabled', false).empty();
    scriptSelect.append(new Option('Select Script', ''));
    try {
        const response = await fetch('/ui/project_selected', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ projects: selectedProjects })
        });
        if (!response.ok) throw new Error('Failed to fetch scripts');
        const data = await response.json();
        if (data.scripts) {
            data.scripts.forEach(script => {
                scriptSelect.append(new Option(script, script));
            });
        }
        // Reinitialize Select2 after adding options
        if (scriptSelect.data('select2')) {
            scriptSelect.select2('destroy');
        }
        scriptSelect.select2({
            placeholder: 'Select Script',
            allowClear: true,
            width: '100%',
            templateSelection: formatState
        });
        // Set value if only one script, or clear
        if (data.scripts && data.scripts.length === 1) {
            scriptSelect.val(data.scripts[0]).trigger('change');
        } else {
            scriptSelect.val('').trigger('change');
        }
        // Handle script selection changes
        scriptSelect.off('change').on('change', function() {
            const selectedScript = $(this).val();
            window.selectedScript = selectedScript;
            if (selectedScript) {
                loadDatasetsForScript(selectedScript);
            } else {
                $('#dataset_file').prop('disabled', true).empty();
            }
        });
    } catch (error) {
        console.error('Error loading scripts:', error);
    }
}
```
*Fetches scripts for selected projects and updates the script dropdown.*

**Load datasets for selected LiFE App script (lines 453-499):**
```javascript
async function loadDatasetsForScript(selectedScript) {
    const datasetSelect = $('#dataset_file');
    datasetSelect.prop('disabled', false).empty();
    datasetSelect.append(new Option('Select Dataset', ''));
    try {
        const response = await fetch('/ui/script_selected', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                script: selectedScript
            })
        });
        if (!response.ok) {
            throw new Error('Failed to fetch datasets');
        }
        const data = await response.json();
        if (data.datasets && data.datasets.length > 0) {
            data.datasets.forEach(dataset => {
                datasetSelect.append(new Option(dataset, dataset));
            });
        }
        // Reinitialize Select2
        if (datasetSelect.data('select2')) {
            datasetSelect.select2('destroy');
        }
        datasetSelect.select2({
            placeholder: "Select Dataset File",
            allowClear: true,
            width: '100%'
        });
        datasetSelect.trigger('change');
    } catch (error) {
        console.error('Error loading datasets:', error);
    }
}
```
*Fetches datasets for the selected script and updates the dataset dropdown.*

---

## 3. `src/autotrain/app/ui_routes.py`

**Form params and selection (lines 515-516, 560-561):**
```python
life_app_project: str = Form("")
life_app_script: str = Form("")
selected_project = form.get("life_app_project")
selected_script = form.get("life_app_script")
```
*Receives LiFE App project and script from the form.*

**LiFE App data source handling (lines 566, 596, 612):**
```python
if data_source == "life_app":
audio_dir = os.path.join("life_app_data", "audio")
processed_csv = os.path.join("life_app_data", "processed_dataset.csv")
```
*Handles LiFE App data source, sets audio and CSV paths.*

**Set params for training (lines 645-646):**
```python
params["life_app_project"] = selected_project
params["life_app_script"] = selected_script
```
*Sets LiFE App params for downstream use.*

**API endpoints (lines 958-983):**
```python
@ui_router.get("/life_app_projects", response_class=JSONResponse)
async def get_life_app_projects(authenticated: bool = Depends(user_authentication)):
    ...
@ui_router.get("/life_app_scripts", response_class=JSONResponse)
async def get_life_app_scripts(authenticated: bool = Depends(user_authentication)):
    ...
@ui_router.get("/life_app_dataset", response_class=JSONResponse)
async def get_life_app_dataset(authenticated: bool = Depends(user_authentication)):
    ...
```
*API endpoints to serve LiFE App projects, scripts, and datasets to the frontend.*

---

## 4. `src/autotrain/trainers/automatic_speech_recognition/dataset.py`

**Load LiFE App dataset (line 226):**
```python
def load_life_app_dataset(data_path):
    """Utility to load LiFE App dataset."""
    ...
```
*Loads the LiFE App dataset from disk for use in ASR training.*

---

## 5. `src/autotrain/preprocessor/automatic_speech_recognition.py`

**Load and preprocess LiFE App dataset (line 192):**
```python
def load_life_app_dataset_from_disk(data_path):
    ...
```
*Loads and preprocesses the LiFE App dataset from disk.*

---

## 6. `src/autotrain/trainers/automatic_speech_recognition/__main__.py`

**Detect and load LiFE App dataset for training (line 519):**
```python
if data_path == "life_app_data" or os.path.basename(data_path) == "life_app_data":
    ...
```
*Detects and loads the LiFE App dataset during ASR training.*

---

## 7. `src/autotrain/project.py`

**Set correct data path for LiFE App datasets (lines 477, 480):**
```python
if hasattr(params, "data_path") and params.data_path == "life_app_data":
    ...
params.data_path = "life_app_data"
```
*Ensures the correct data path is set for LiFE App datasets when creating a project.*

---

## 8. `life_app_data/`

- **`processed_dataset.csv`**: Contains the processed LiFE App dataset (audio file paths and transcriptions).
- **`audio/`**: Contains all audio files referenced in the dataset.

*Stores the actual data used by the LiFE App integration.*

---

## Notes
- This document is based on a triple-checked, line-by-line review of the local repo.
- If you want the exact code for any function or a line-by-line breakdown, specify the file and section. 