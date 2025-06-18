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

    // Initialize Select2 for all dropdowns
    if (typeof $ !== 'undefined') {
        // Initialize project dropdown with Select2
        $('#life_app_project').select2({
            placeholder: 'Select Project(s)',
            allowClear: true,
            multiple: true,
            width: '100%',
            theme: 'classic'
        });

        // Initialize script dropdown with Select2
        $('#life_app_script').select2({
            placeholder: 'Select Script',
            allowClear: true,
            width: '100%',
            theme: 'classic'
        });

        // Initialize dataset dropdown with Select2
        $('#dataset_file').select2({
            placeholder: 'Select Dataset',
            allowClear: true,
            width: '100%',
            theme: 'classic'
        });

        // Function to load projects
        async function loadLifeAppProjects() {
            try {
                const response = await fetch('/ui/life_app_projects');
                if (!response.ok) {
                    throw new Error('Failed to load projects');
                }
                const data = await response.json();
                const projects = data.projects;
                
                // Clear and populate project dropdown
                $('#life_app_project').empty();
                projects.forEach(project => {
                    $('#life_app_project').append(new Option(project, project));
                });
                $('#life_app_project').trigger('change');
            } catch (error) {
                console.error('Error loading projects:', error);
                showError('Failed to load projects. Please try again.');
            }
        }

        // On project change, load scripts
        $('#life_app_project').on('change', function() {
            const selectedProjects = $(this).val();
            if (selectedProjects && selectedProjects.length > 0) {
                fetch('/ui/life_app_scripts')
                    .then(res => res.json())
                    .then(data => {
                        const scripts = data.scripts;
                        // Clear and populate script dropdown
                        $('#life_app_script').empty();
                        scripts.forEach(script => {
                            $('#life_app_script').append(new Option(script, script));
                        });
                        $('#life_app_script').prop('disabled', false).trigger('change');
                    })
                    .catch(error => {
                        console.error('Error loading scripts:', error);
                        showError('Failed to load scripts. Please try again.');
                    });
            } else {
                $('#life_app_script').empty().prop('disabled', true).trigger('change');
            }
        });

        // On script change, load dataset files
        $('#life_app_script').on('change', function() {
            const datasetSelect = $('#dataset_file');
            if (this.value) {
                datasetSelect.prop('disabled', false);
                loadDatasetFiles();
            } else {
                datasetSelect.empty().prop('disabled', true).trigger('change');
            }
        });

        // Function to load dataset files
        async function loadDatasetFiles() {
            const datasetSelect = $('#dataset_file');
            try {
                const response = await fetch('/ui/life_app_dataset');
                if (!response.ok) {
                    throw new Error('Failed to load dataset');
                }
                const data = await response.json();
                const dataset = data.dataset;
                
                // Clear and populate dataset dropdown
                datasetSelect.empty();
                datasetSelect.append(new Option('Current Dataset', 'dataset.json'));
                datasetSelect.prop('disabled', false).trigger('change');
            } catch (error) {
                console.error('Error loading dataset:', error);
                showError('Failed to load dataset. Please try again.');
            }
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

        // Call loadLifeAppProjects when LiFE App is selected
        $('#dataset_source').on('change', function() {
            if (this.value === 'life_app') {
                loadLifeAppProjects();
            }
        });
    } else {
        console.error('jQuery is not loaded. Select2 initialization failed.');
    }
});