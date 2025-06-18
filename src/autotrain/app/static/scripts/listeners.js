document.addEventListener('DOMContentLoaded', function () {
    const dataSource = document.getElementById("dataset_source");
    const uploadDataTabContent = document.getElementById("upload-data-tab-content");
    const hubDataTabContent = document.getElementById("hub-data-tab-content");
    const uploadDataTabs = document.getElementById("upload-data-tabs");

    const jsonCheckbox = document.getElementById('show-json-parameters');
    const jsonParametersDiv = document.getElementById('json-parameters');
    const dynamicUiDiv = document.getElementById('dynamic-ui');

    const paramsTextarea = document.getElementById('params_json');

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

    const observeParamChanges = () => {
        const paramElements = document.querySelectorAll('[id^="param_"]');
        paramElements.forEach(el => {
            el.addEventListener('input', updateTextarea);
        });
    };

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

    function switchToJSON() {
        if (jsonCheckbox.checked) {
            dynamicUiDiv.style.display = 'none';
            jsonParametersDiv.style.display = 'block';
        } else {
            dynamicUiDiv.style.display = 'block';
            jsonParametersDiv.style.display = 'none';
        }
    }

    // Function to handle changes in dataset_source and task
    function handleDataSource() {
        const lifeAppSelection = document.getElementById("life-app-selection");
        const datasetFileDiv = document.getElementById('dataset_file_div');
        const scriptDiv = document.getElementById('life_app_script').parentElement;
        const scriptSelect = document.getElementById('life_app_script');
        const datasetSelect = document.getElementById('dataset_file');
        const taskValue = document.getElementById('task').value;

        // Hide all data source related sections by default
        if (hubDataTabContent) hubDataTabContent.style.display = "none";
        if (uploadDataTabContent) uploadDataTabContent.style.display = "none";
        if (uploadDataTabs) uploadDataTabs.style.display = "none";
        if (lifeAppSelection) lifeAppSelection.style.display = "none";
        if (datasetFileDiv) datasetFileDiv.style.display = 'none';

        // Show/hide LiFE App option based on task
        const lifeAppOption = document.getElementById("dataset_source").querySelector('option[value="life_app"]');
        if (lifeAppOption) {
            if (taskValue === "automatic-speech-recognition") {
                lifeAppOption.style.display = ""; // Show option
            } else {
                lifeAppOption.style.display = "none"; // Hide option
                // If LiFE App was selected, switch to local
                if (dataSource.value === "life_app") {
                    dataSource.value = "local";
                }
            }
        }

        // Show relevant section based on selected data source
        if (dataSource.value === "life_app" && taskValue === "automatic-speech-recognition") {
            if (lifeAppSelection) lifeAppSelection.style.display = "block";
            scriptDiv.style.display = '';
            scriptSelect.innerHTML = '<option value="">Select Script</option>';
            scriptSelect.disabled = true;
            datasetFileDiv.style.display = '';
            datasetSelect.innerHTML = '<option value="">Select Dataset</option>';
            datasetSelect.disabled = true;
            loadLifeAppProjects();
        } else if (dataSource.value === "huggingface") {
            if (hubDataTabContent) hubDataTabContent.style.display = "block";
        } else if (dataSource.value === "local") {
            if (uploadDataTabContent) uploadDataTabContent.style.display = "block";
            if (uploadDataTabs) uploadDataTabs.style.display = "block";
        }
    }

    async function fetchParams() {
        const taskValue = document.getElementById('task').value;
        const parameterMode = document.getElementById('parameter_mode').value;
        const response = await fetch(`/ui/params/${taskValue}/${parameterMode}`);
        const params = await response.json();
        return params;
    }

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

    fetchParams().then(params => renderUI(params));

    document.getElementById('task').addEventListener('change', function () {
        fetchParams().then(params => {
            document.getElementById('dynamic-ui').innerHTML = '';
            let jsonCheckboxFlag = false;
            if (jsonCheckbox.checked) {
                jsonCheckbox.checked = false;
                jsonCheckboxFlag = true;
            }
            renderUI(params);
            if (jsonCheckboxFlag) {
                jsonCheckbox.checked = true;
                updateTextarea();
                observeParamChanges();
            }
            handleDataSource(); // Call handleDataSource on task change
        });
    });

    document.getElementById('parameter_mode').addEventListener('change', function () {
        fetchParams().then(params => {
            document.getElementById('dynamic-ui').innerHTML = '';
            let jsonCheckboxFlag = false;
            if (jsonCheckbox.checked) {
                jsonCheckbox.checked = false;
                jsonCheckboxFlag = true;
            }
            renderUI(params);
            if (jsonCheckboxFlag) {
                jsonCheckbox.checked = true;
                updateTextarea();
                observeParamChanges();
            }
        });
    });

    jsonCheckbox.addEventListener('change', function () {
        if (jsonCheckbox.checked) {
            updateTextarea();
            observeParamChanges();
        }
    });

    document.getElementById('task').addEventListener('change', function () {
        if (jsonCheckbox.checked) {
            updateTextarea();
            observeParamChanges();
        }
    });

    // Attach event listeners to dataset_source dropdown and task dropdown
    dataSource.addEventListener("change", handleDataSource);
    document.getElementById('task').addEventListener('change', handleDataSource); // Ensure handleDataSource is called on task change
    jsonCheckbox.addEventListener('change', switchToJSON);
    paramsTextarea.addEventListener('input', updateParamsFromTextarea);

    // Trigger the event listener to set the initial state
    handleDataSource();
    observeParamChanges();
    updateTextarea();

    // LiFE App Dataset Selection (API vs JSON)
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

    // Handle JSON file selection
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

    // --- Project multi-select tags update ---
    function updateProjectTags() {
        const projectSelect = document.getElementById('life_app_project');
        const tagContainer = document.getElementById('life-app-project-tags');
        if (!projectSelect || !tagContainer) return;

        tagContainer.innerHTML = '';
        Array.from(projectSelect.selectedOptions).forEach(option => {
            const tag = document.createElement('span');
            tag.className = 'bg-blue-200 text-blue-800 text-sm font-semibold mr-2 px-2.5 py-0.5 rounded dark:bg-blue-200 dark:text-blue-900';
            tag.textContent = option.textContent;
            const removeBtn = document.createElement('button');
            removeBtn.className = 'ml-1 text-blue-800 hover:text-blue-600 focus:outline-none';
            removeBtn.innerHTML = '&times;';
            removeBtn.onclick = () => {
                option.selected = false;
                updateProjectTags(); // Refresh tags after removal
            };
            tag.appendChild(removeBtn);
            tagContainer.appendChild(tag);
        });
    }

    document.getElementById('life_app_project').addEventListener('change', function() {
        updateProjectTags();
        const scriptSelect = document.getElementById('life_app_script');
        if (this.selectedOptions.length > 0) {
            scriptSelect.innerHTML = '<option value="">Select Script</option>';
            scriptSelect.disabled = false;
            // Plain JS: load scripts instantly
            fetch('/static/scriptList.json')
                .then(response => response.json())
                .then(scripts => {
                    scripts.forEach(script => {
                        const option = document.createElement('option');
                        option.value = script;
                        option.textContent = script;
                        scriptSelect.appendChild(option);
                    });
                });
        } else {
            scriptSelect.innerHTML = '<option value="">Select Script</option>';
            scriptSelect.disabled = true;
        }
    });

    document.getElementById('life_app_script').addEventListener('change', function() {
        const datasetSelect = document.getElementById('dataset_file');
        if (this.value) {
            datasetSelect.disabled = false;
            $(datasetSelect).prop('disabled', false);
            $(datasetSelect).select2();
            $(datasetSelect).trigger('change.select2');
            loadDatasetFiles();
        } else {
            datasetSelect.innerHTML = '<option value="">Select Dataset</option>';
            datasetSelect.disabled = true;
            $(datasetSelect).prop('disabled', true);
            $(datasetSelect).select2();
            $(datasetSelect).trigger('change.select2');
            if ($(datasetSelect).data('select2')) { $(datasetSelect).val(null).trigger('change'); }
        }
    });

    // Function to load projects
    async function loadLifeAppProjects() {
        const projectSelect = document.getElementById('life_app_project');
        if (!projectSelect) return;
        projectSelect.innerHTML = '';

        try {
            // Simulate API call by loading from static file
            const response = await fetch('/static/projectList.json');
            if (!response.ok) {
                throw new Error('Failed to load projects');
            }
            const projects = await response.json();
            
            // Add projects to select
            projects.forEach(project => {
                const option = document.createElement('option');
                option.value = project;
                option.textContent = project;
                projectSelect.appendChild(option);
            });
        } catch (error) {
            console.error('Error loading projects:', error);
            // Show error to user
            const errorDiv = document.createElement('div');
            errorDiv.className = 'text-red-500 text-sm mt-2';
            errorDiv.textContent = 'Failed to load projects. Please try again.';
            projectSelect.parentNode.appendChild(errorDiv);
        }
    }

    // Function to load scripts
    async function loadLifeAppScripts() {
        // No-op for script dropdown, handled inline above
    }

    // Function to load dataset files
    async function loadDatasetFiles() {
        const container = document.getElementById('dataset_file_div');
        const select = document.getElementById('dataset_file');
        
        if (!container || !select) {
            console.error('Dataset elements not found');
            return;
        }

        try {
            // Destroy existing Select2 instance if it exists
            if ($(select).data('select2')) {
                $(select).select2('destroy');
            }

            // Clear current options
            select.innerHTML = '';
            
            // Simulate API call by loading from static file
            const response = await fetch('/static/dataset.json');
            if (!response.ok) {
                throw new Error('Failed to load dataset');
            }
            const dataset = await response.json();
            
            // Add dataset option
            select.innerHTML = `
                <option value="">Select Dataset</option>
                <option value="dataset.json">Current Dataset</option>
            `;

            // Make sure container is visible
            container.style.display = 'block';

            // Reinitialize Select2
            $(select).select2({
                placeholder: "Select Dataset File",
                width: '100%'
            });
        } catch (error) {
            console.error('Error loading dataset:', error);
            // Show error to user
            const errorDiv = document.createElement('div');
            errorDiv.className = 'text-red-500 text-sm mt-2';
            errorDiv.textContent = 'Failed to load dataset. Please try again.';
            select.parentNode.appendChild(errorDiv);
        }
    }

    // Add validation before form submission
    function validateLifeAppData() {
        const projectSelect = document.getElementById('life_app_project');
        const scriptSelect = document.getElementById('life_app_script');
        const datasetSelect = document.getElementById('dataset_file');
        
        // Check if project is selected
        if (!projectSelect.value) {
            showError('Please select at least one project');
            return false;
        }
        
        // Check if script is selected
        if (!scriptSelect.value) {
            showError('Please select a script');
            return false;
        }
        
        // Check if dataset is selected
        if (!datasetSelect.value) {
            showError('Please select a dataset');
            return false;
        }
        
        return true;
    }

    // Helper function to show errors
    function showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'text-red-500 text-sm mt-2';
        errorDiv.textContent = message;
        
        // Remove any existing error messages
        const existingError = document.querySelector('.text-red-500');
        if (existingError) {
            existingError.remove();
        }
        
        // Add new error message
        document.getElementById('life-app-selection').appendChild(errorDiv);
        
        // Remove error after 5 seconds
        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
    }
});