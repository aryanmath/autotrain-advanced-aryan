/**
 * AutoTrain Advanced - Frontend Event Listeners and UI Logic
 *
 * This file handles all user interactions in the AutoTrain UI:
 * - Parameter management (basic/full mode, JSON editing)
 * - Dataset source selection (Local, Hugging Face Hub, LiFE App for ASR)
 * - Dynamic UI rendering based on task and parameters
 * - LiFE App integration for ASR task only
 *
 * File Structure:
 * 1. Variable Declarations
 * 2. Parameter Management Functions
 * 3. UI Rendering Functions
 * 4. Dataset Source Handling
 * 5. Event Listeners Setup
 * 6. ASR/LiFE App Specific Logic
 */

document.addEventListener('DOMContentLoaded', function () {

    // ============================================================================
    // 1. VARIABLE DECLARATIONS - Get references to important HTML elements
    // ============================================================================

    // Dataset source related elements
    const dataSource = document.getElementById("dataset_source");
    const uploadDataTabContent = document.getElementById("upload-data-tab-content");
    const hubDataTabContent = document.getElementById("hub-data-tab-content");
    const uploadDataTabs = document.getElementById("upload-data-tabs");

    // Parameter management elements
    const jsonCheckbox = document.getElementById('show-json-parameters');
    const jsonParametersDiv = document.getElementById('json-parameters');
    const dynamicUiDiv = document.getElementById('dynamic-ui');
    const paramsTextarea = document.getElementById('params_json');

    // ============================================================================
    // 2. PARAMETER MANAGEMENT FUNCTIONS - Handle parameter editing and JSON mode
    // ============================================================================

    /**
     * Updates the JSON textarea with current parameter values
     * Called whenever any parameter input changes
     */
    const updateTextarea = () => {
        const paramElements = document.querySelectorAll('[id^="param_"]');
        const params = {};
        paramElements.forEach(el => {
            const key = el.id.replace('param_', '');
            params[key] = el.value;
        });
        paramsTextarea.value = JSON.stringify(params, null, 2);
        paramsTextarea.style.height = '600px';
    };

    /**
     * Adds change listeners to all parameter input fields
     * This ensures the JSON textarea stays in sync with the UI
     */
    const observeParamChanges = () => {
        const paramElements = document.querySelectorAll('[id^="param_"]');
        paramElements.forEach(el => {
            el.addEventListener('input', updateTextarea);
        });
    };

    /**
     * Updates parameter input fields from JSON textarea content
     * Called when user edits the JSON directly
     */
    const updateParamsFromTextarea = () => {
        try {
            const params = JSON.parse(paramsTextarea.value);
            Object.keys(params).forEach(key => {
                const el = document.getElementById('param_' + key);
                if (el) {
                    el.value = params[key];
                }
            });
        } catch (e) {
            console.error('Invalid JSON:', e);
        }
    };

    /**
     * Switches between dynamic UI mode and JSON editing mode
     * When JSON checkbox is checked, hides the form and shows JSON textarea
     */
    function switchToJSON() {
        if (jsonCheckbox.checked) {
            dynamicUiDiv.style.display = 'none';
            jsonParametersDiv.style.display = 'block';
        } else {
            dynamicUiDiv.style.display = 'block';
            jsonParametersDiv.style.display = 'none';
        }
    }

    // ============================================================================
    // 3. UI RENDERING FUNCTIONS - Create and display parameter input fields
    // ============================================================================

    /**
     * Fetches parameter configuration from backend based on task and mode
     * Returns the parameter definitions (type, label, default value, etc.)
     */
    async function fetchParams() {
        const taskValue = document.getElementById('task').value;
        const parameterMode = document.getElementById('parameter_mode').value;
        const response = await fetch(`/ui/params/${taskValue}/${parameterMode}`);
        const params = await response.json();
        return params;
    }

    /**
     * Creates HTML element for a single parameter based on its type
     * Supports: number, dropdown, checkbox, and string input types
     */
    function createElement(param, config) {
        let element = '';
        switch (config.type) {
            case 'number':
                element = `<div>
                    <label for="param_${param}" class="text-sm font-medium text-gray-700 dark:text-gray-300">${config.label}</label>
                    <input type="number" name="param_${param}" id="param_${param}" value="${config.default}"
                        class="mt-1 p-1 text-xs font-medium w-full border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500">
                </div>`;
                break;
            case 'dropdown':
                let options = config.options.map(option => `<option value="${option}" ${option === config.default ? 'selected' : ''}>${option}</option>`).join('');
                element = `<div>
                    <label for="param_${param}" class="text-sm font-medium text-gray-700 dark:text-gray-300">${config.label}</label>
                    <select name="param_${param}" id="param_${param}"
                        class="mt-1 p-1 text-xs font-medium w-full border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500">
                        ${options}
                    </select>
                </div>`;
                break;
            case 'checkbox':
                element = `<div>
                    <label for="param_${param}" class="text-sm font-medium text-gray-700 dark:text-gray-300">${config.label}</label>
                    <input type="checkbox" name="param_${param}" id="param_${param}" ${config.default ? 'checked' : ''}
                        class="mt-1 text-xs font-medium border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500">
                </div>`;
                break;
            case 'string':
                element = `<div>
                    <label for="param_${param}" class="text-sm font-medium text-gray-700 dark:text-gray-300">${config.label}</label>
                    <input type="text" name="param_${param}" id="param_${param}" value="${config.default}"
                        class="mt-1 p-1 text-xs font-medium w-full border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500">
                </div>`;
                break;
        }
        return element;
    }

    /**
     * Renders the complete parameter UI in a grid layout
     * Groups parameters by type and arranges them in rows of 3
     */
    function renderUI(params) {
        const uiContainer = document.getElementById('dynamic-ui');
        let rowDiv = null;
        let rowIndex = 0;
        let lastType = null;

        Object.keys(params).forEach((param, index) => {
            const config = params[param];
            if (lastType !== config.type || rowIndex >= 3) {
                if (rowDiv) uiContainer.appendChild(rowDiv);
                rowDiv = document.createElement('div');
                rowDiv.className = 'grid grid-cols-3 gap-2 mb-2';
                rowIndex = 0;
            }
            rowDiv.innerHTML += createElement(param, config);
            rowIndex++;
            lastType = config.type;
        });
        if (rowDiv) uiContainer.appendChild(rowDiv);
    }

    // ============================================================================
    // 4. DATASET SOURCE HANDLING - Manage different dataset input methods
    // ============================================================================

    /**
     * Handles dataset source selection and shows/hides appropriate UI sections
     * Supports: Local upload, Hugging Face Hub, and LiFE App (ASR only)
     */
    function handleDataSource() {
        // Main repo logic for hub/local dataset sources
        if (dataSource.value === "hub") {
            uploadDataTabContent.style.display = "none";
            uploadDataTabs.style.display = "none";
            hubDataTabContent.style.display = "block";
        } else if (dataSource.value === "local") {
            uploadDataTabContent.style.display = "block";
            uploadDataTabs.style.display = "block";
            hubDataTabContent.style.display = "none";
        }

        // === ASR-SPECIFIC LOGIC START ===
        // Show/hide LiFE App option and UI only for ASR task
        const lifeAppOption = document.getElementById("dataset_source").querySelector('option[value="life_app"]');
        const lifeAppSelection = document.getElementById("life-app-selection");
        const datasetFileDiv = document.getElementById('dataset_file_div');
        const taskValue = document.getElementById('task').value;

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

        if (dataSource.value === "life_app" && taskValue === "ASR") {
            if (lifeAppSelection) lifeAppSelection.style.display = "block";
            if (datasetFileDiv) datasetFileDiv.style.display = 'block';
            loadLifeAppProjects();
        } else {
            if (lifeAppSelection) lifeAppSelection.style.display = "none";
            if (datasetFileDiv) datasetFileDiv.style.display = 'none';
        }
        // === ASR-SPECIFIC LOGIC END ===
    }

    // ============================================================================
    // 5. EVENT LISTENERS SETUP - Connect user actions to functions
    // ============================================================================

    // Initialize the UI with current parameters
    fetchParams().then(params => renderUI(params));

    // Handle task changes (e.g., switching from text classification to ASR)
    document.getElementById('task').addEventListener('change', function () {
        fetchParams().then(params => {
            document.getElementById('dynamic-ui').innerHTML = '';
            let jsonCheckBoxFlag = false;
            if (jsonCheckbox.checked) {
                jsonCheckbox.checked = false;
                jsonCheckBoxFlag = true;
            }
            renderUI(params);
            if (jsonCheckBoxFlag) {
                jsonCheckbox.checked = true;
                updateTextarea();
                observeParamChanges();
            }
            // === ASR-SPECIFIC LOGIC START ===
            handleDataSource(); // Update dataset source options for new task
            // === ASR-SPECIFIC LOGIC END ===
        });
    });

    // Handle parameter mode changes (basic vs full)
    document.getElementById('parameter_mode').addEventListener('change', function () {
        fetchParams().then(params => {
            document.getElementById('dynamic-ui').innerHTML = '';
            let jsonCheckBoxFlag = false;
            if (jsonCheckbox.checked) {
                jsonCheckbox.checked = false;
                jsonCheckBoxFlag = true;
            }
            renderUI(params);
            if (jsonCheckBoxFlag) {
                jsonCheckbox.checked = true;
                updateTextarea();
                observeParamChanges();
            }
        });
    });

    // Handle JSON mode toggle
    jsonCheckbox.addEventListener('change', function () {
        if (jsonCheckbox.checked) {
            updateTextarea();
            observeParamChanges();
        }
    });

    // Handle task changes for JSON mode
    document.getElementById('task').addEventListener('change', function () {
        if (jsonCheckbox.checked) {
            updateTextarea();
            observeParamChanges();
        }
    });

    // Attach event listeners to dataset source dropdown
    dataSource.addEventListener("change", handleDataSource);
    jsonCheckbox.addEventListener('change', switchToJSON);
    paramsTextarea.addEventListener('input', updateParamsFromTextarea);

    // Initialize the UI state
    handleDataSource();
    observeParamChanges();
    updateTextarea();

    // ============================================================================
    // 6. ASR/LiFE APP SPECIFIC LOGIC - Only active for ASR task
    // ============================================================================
    // This section handles LiFE App integration for ASR tasks only
    // Includes: Project selection, script loading, dataset fetching, and JSON upload

    // === ASR-SPECIFIC LOGIC START ===

    /**
     * LiFE App Dataset Source Selection
     * Allows users to choose between API access or JSON file upload
     */
    if (document.getElementById('life-app-source')) {
        document.getElementById('life-app-source').addEventListener('change', function() {
            const apiContainer = document.getElementById('life-app-api-container');
            const jsonContainer = document.getElementById('life-app-json-container');
            if (this.value === 'api') {
                apiContainer.style.display = 'block';
                jsonContainer.style.display = 'none';
            } else if (this.value === 'json') {
                apiContainer.style.display = 'none';
                jsonContainer.style.display = 'block';
            } else {
                apiContainer.style.display = 'none';
                jsonContainer.style.display = 'none';
            }
        });
    }

    /**
     * LiFE App JSON File Upload Handler
     * Validates and processes uploaded JSON files with audio/transcription data
     */
    if (document.getElementById('life-app-json-file')) {
        document.getElementById('life-app-json-file').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    try {
                        const data = JSON.parse(e.target.result);
                        if (!Array.isArray(data)) {
                            throw new Error('JSON must be an array');
                        }
                        if (data.length === 0) {
                            throw new Error('JSON array is empty');
                        }
                        const firstItem = data[0];
                        if (!firstItem.audio || !firstItem.transcription) {
                            throw new Error('Each item must have audio and transcription fields');
                        }
                        window.lifeAppData = data;
                    } catch (error) {
                        alert('Invalid JSON file: ' + error.message);
                        e.target.value = '';
                    }
                };
                reader.readAsText(file);
            }
        });
    }

    /**
     * Loads LiFE App projects from backend API
     * Populates the project dropdown with available projects
     */
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

            // Initialize Select2 for enhanced dropdown functionality
            if (!$(projectSelect).data('select2')) {
                $(projectSelect).select2({
                    placeholder: "Select LiFE App Project(s)",
                    allowClear: true,
                    multiple: true,
                    width: '100%',
                    maximumSelectionLength: 2,
                    tags: true
                });
            }

            updateProjectTags();

            // Handle project selection changes
            $(projectSelect).off('change').on('change', function() {
                updateProjectTags();
                const selectedProjects = $(this).val();
                if (selectedProjects && selectedProjects.length > 0) {
                    loadScriptsForProjects(selectedProjects);
                } else {
                    $('#life_app_script').prop('disabled', true).empty();
                }
            });
        } catch (error) {
            console.error('Error loading projects:', error);
        }
    }

    /**
     * Initialize Select2 for script dropdown
     * Provides enhanced dropdown functionality with search and clear options
     */
    $('#life_app_script').select2({
        placeholder: "Select Script",
        allowClear: true,
        width: '100%'
    });

    /**
     * Loads scripts for selected LiFE App projects
     * Fetches available scripts from backend and populates dropdown
     */
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
                placeholder: "Select Script",
                allowClear: true,
                width: '100%'
            });

            // Restore previous selection if possible
            if (window.selectedScript && data.scripts.includes(window.selectedScript)) {
                scriptSelect.val(window.selectedScript).trigger('change');
            } else if (data.scripts && data.scripts.length === 1) {
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

    /**
     * Project selection event handler
     * Triggers script loading when projects are selected
     */
    $('#life_app_project').on('change', function() {
        const selectedProjects = $(this).val();
        if (selectedProjects && selectedProjects.length > 0) {
            loadScriptsForProjects(selectedProjects);
        } else {
            $('#life_app_script').prop('disabled', true).empty();
            $('#dataset_file').prop('disabled', true).empty();
        }
    });

    /**
     * Loads datasets for selected LiFE App script
     * Fetches available datasets from backend and populates dropdown
     */
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

    /**
     * Updates the visual tag display for selected projects
     * Shows selected projects as removable tags below the dropdown
     */
    function updateProjectTags() {
        const projectSelect = document.getElementById('life_app_project');
        const tagContainer = document.getElementById('life-app-project-tags');
        if (!projectSelect || !tagContainer) return;

        tagContainer.innerHTML = '';
        const selectedOptions = $(projectSelect).select2('data');

        selectedOptions.forEach(option => {
            const tag = document.createElement('span');
            tag.className = 'inline-flex items-center px-2 py-1 mr-2 mb-2 text-sm font-medium text-blue-800 bg-blue-100 rounded-full dark:bg-blue-900 dark:text-blue-300';
            tag.textContent = option.text;

            const removeButton = document.createElement('button');
            removeButton.className = 'ml-1 text-blue-800 hover:text-blue-600 dark:text-blue-300 dark:hover:text-blue-100';
            removeButton.innerHTML = '×';
            removeButton.onclick = () => {
                const optionElement = Array.from(projectSelect.options).find(opt => opt.value === option.id);
                if (optionElement) {
                    optionElement.selected = false;
                    $(projectSelect).trigger('change');
                }
            };

            tag.appendChild(removeButton);
            tagContainer.appendChild(tag);
        });
    }

    // === ASR-SPECIFIC LOGIC END ===

});