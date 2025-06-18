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
            if (datasetFileDiv) datasetFileDiv.style.display = 'block';
            loadLifeAppProjects();
            loadLifeAppScripts();
            loadDatasetFiles();
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

    // Helper for Select2 templating (simple passthrough for now)
    function formatState (state) {
        if (!state.id) {
            return state.text;
        }
        return state.text;
    }

    // Initialize Select2 for LiFE App Project (multi-select, max 2)
    $(document).ready(function() {
        $('.js-example-basic-multiple-limit').select2({
            maximumSelectionLength: 2,
            placeholder: 'Select Project(s)'
        });
        $('.js-example-templating').select2({
            templateSelection: formatState,
            placeholder: 'Select Script or Dataset'
        });

        // Load projects into project select
        fetch('/static/projectList.json')
            .then(res => res.json())
            .then(projects => {
                const $project = $('#life_app_project');
                $project.empty();
                projects.forEach(p => {
                    $project.append(new Option(p, p));
                });
                $project.trigger('change.select2');
            });

        // On project change, update script options
        $('#life_app_project').on('change', function() {
            const selectedProjects = $(this).val() || [];
            const $script = $('#life_app_script');
            $script.empty();
            if (selectedProjects.length > 0) {
                fetch('/static/scriptList.json')
                    .then(res => res.json())
                    .then(scripts => {
                        $script.append(new Option('Select Script', ''));
                        scripts.forEach(s => {
                            $script.append(new Option(s, s));
                        });
                        $script.val('').trigger('change.select2');
                    });
            } else {
                $script.append(new Option('Select Script', ''));
                $script.val('').trigger('change.select2');
            }
        });

        // On script change, update dataset options (dummy logic for now)
        $('#life_app_script').on('change', function() {
            const selectedScript = $(this).val();
            const $dataset = $('#dataset_file');
            $dataset.empty();
            if (selectedScript) {
                fetch('/static/dataset.json')
                    .then(res => res.json())
                    .then(dataset => {
                        $dataset.append(new Option('Select Dataset', ''));
                        $dataset.append(new Option('Current Dataset', 'dataset.json'));
                        $dataset.val('').trigger('change.select2');
                    });
            } else {
                $dataset.append(new Option('Select Dataset', ''));
                $dataset.val('').trigger('change.select2');
            }
        });
    });
});