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
        //paramsTextarea.className = 'p-2.5 w-full text-sm text-gray-600 border-white border-transparent focus:border-transparent focus:ring-0'
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

    function handleDataSource() {
        const lifeAppSelection = document.getElementById("life-app-selection");

        if (dataSource.value === "life_app") {
            // Show only LiFE App selection UI
            if (lifeAppSelection) lifeAppSelection.style.display = "block";
            // Hide hub and local upload sections
            if (hubDataTabContent) hubDataTabContent.style.display = "none";
            if (uploadDataTabContent) uploadDataTabContent.style.display = "none";
            if (uploadDataTabs) uploadDataTabs.style.display = "none";
            loadLifeAppProjects();
            loadLifeAppScripts();
        } else if (dataSource.value === "huggingface") {
            // Show hub dataset section
            if (hubDataTabContent) hubDataTabContent.style.display = "block";
            // Hide LiFE App and local upload sections
            if (lifeAppSelection) lifeAppSelection.style.display = "none";
            if (uploadDataTabContent) uploadDataTabContent.style.display = "none";
            if (uploadDataTabs) uploadDataTabs.style.display = "none";
        } else if (dataSource.value === "local") {
            // Show local upload sections
            if (uploadDataTabContent) uploadDataTabContent.style.display = "block";
            if (uploadDataTabs) uploadDataTabs.style.display = "block";
            // Hide LiFE App and hub dataset sections
            if (hubDataTabContent) hubDataTabContent.style.display = "none";
            if (lifeAppSelection) lifeAppSelection.style.display = "none";
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
    // Attach event listeners to dataset_source dropdown
    dataSource.addEventListener("change", handleDataSource);
    jsonCheckbox.addEventListener('change', switchToJSON);
    paramsTextarea.addEventListener('input', updateParamsFromTextarea);

    // Trigger the event listener to set the initial state
    handleDataSource();
    observeParamChanges();
    updateTextarea();

    // Add event listener for task changes
    document.getElementById('task').addEventListener('change', function() {
        if (dataSource.value === "life_app" && this.value !== "automatic-speech-recognition") {
            alert("LiFE app datasets can only be used with Automatic Speech Recognition tasks");
            dataSource.value = "local";
            handleDataSource();
        }
    });

    // LiFE App Dataset Selection
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
                    // Store the validated data
                    window.lifeAppData = data;
                } catch (error) {
                    alert('Invalid JSON file: ' + error.message);
                    e.target.value = '';
                }
            };
            reader.readAsText(file);
        }
    });

    // Update dataset source change handler
    document.getElementById('dataset_source').addEventListener('change', function() {
        console.log('Dataset source changed to:', this.value);
        const lifeAppSelection = document.getElementById('life-app-selection');
        const projectSelect = document.getElementById('life_app_project');
        const scriptSelect = document.getElementById('life_app_script');
        const tagContainer = document.getElementById('life-app-project-tags');

        if (this.value === 'life_app') {
            console.log('Showing LiFE App selection');
            lifeAppSelection.style.display = 'block';

            // --- Script dropdown (single select) ---
            if (scriptSelect) {
                console.log('Loading scripts...');
                fetch('/static/scriptList.json')
                    .then(response => {
                        console.log('Script list response:', response.status);
                        return response.json();
                    })
                    .then(data => {
                        console.log('Scripts loaded:', data);
                        scriptSelect.innerHTML = '<option value="">Select Script</option>';
                        data.forEach(script => {
                            const option = document.createElement('option');
                            option.value = script;
                            option.textContent = script;
                            scriptSelect.appendChild(option);
                        });
                    })
                    .catch(error => {
                        console.error('Error loading scripts:', error);
                        alert('Failed to load scripts. Please check console for details.');
                    });
            }
        } else {
            lifeAppSelection.style.display = 'none';
        }
    });

    // --- LiFE App Dataset Selection: Show/Hide and Populate ---
    document.getElementById('dataset_source').addEventListener('change', function() {
        const lifeAppSelection = document.getElementById('life-app-selection');
        const projectSelect = document.getElementById('life_app_project');
        const scriptSelect = document.getElementById('life_app_script');
        const tagContainer = document.getElementById('life-app-project-tags');

        if (this.value === 'life_app') {
            lifeAppSelection.style.display = 'block';

            // --- Script dropdown (single select) ---
            if (scriptSelect) {
                fetch('/static/scriptList.json')
                    .then(response => response.json())
                    .then(data => {
                        scriptSelect.innerHTML = '<option value="">Select Script</option>';
                        data.forEach(script => {
                            const option = document.createElement('option');
                            option.value = script;
                            option.textContent = script;
                            scriptSelect.appendChild(option);
                        });
                    })
                    .catch(error => {
                        console.error('Error loading scripts:', error);
                        alert('Failed to load scripts. Please check console for details.');
                    });
            }
        } else {
            lifeAppSelection.style.display = 'none';
        }
    });

    // --- Project multi-select tags update ---
    document.getElementById('life_app_project').addEventListener('change', function() {
        const projectSelect = document.getElementById('life_app_project');
        const tagContainer = document.getElementById('life-app-project-tags');
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
                updateProjectTags(); // Refresh tags
            };
            tag.appendChild(removeBtn);
            tagContainer.appendChild(tag);
        });
    });

    // Function to load projects
    async function loadLifeAppProjects() {
        const projectSelect = document.getElementById('life_app_project');
        const projectTagsContainer = document.getElementById('life-app-project-tags');
        if (!projectSelect) return;
        projectSelect.innerHTML = '';

        try {
            const response = await fetch('/static/projectList.json');
            const projects = await response.json();
            projects.forEach(project => {
                const option = document.createElement('option');
                option.value = project;
                option.textContent = project;
                projectSelect.appendChild(option);
            });
            // Trigger change event to update tags
            updateProjectTags();
        } catch (error) {
            console.error('Error loading projects:', error);
        }
    }

    // Function to load scripts
    async function loadLifeAppScripts() {
        const scriptSelect = document.getElementById('life_app_script');
        if (!scriptSelect) return;
        scriptSelect.innerHTML = '<option value="">Select Script</option>';
        try {
            const response = await fetch('/static/scriptList.json');
            const scripts = await response.json();
            scripts.forEach(script => {
                const option = document.createElement('option');
                option.value = script;
                option.textContent = script;
                scriptSelect.appendChild(option);
            });
        } catch (error) {
            console.error('Error loading scripts:', error);
        }
    }

    // Function to load dataset files
    async function loadDatasetFiles() {
        const datasetSelect = document.getElementById('dataset_file');
        if (!datasetSelect) return;
        
        try {
            // First destroy any existing select2
            if ($(datasetSelect).hasClass('select2-hidden-accessible')) {
                $(datasetSelect).select2('destroy');
            }
            
            // Clear and add options directly to the select element
            datasetSelect.innerHTML = `
                <option value="">Select Dataset</option>
                <option value="dataset.json">dataset.json</option>
            `;

            // Initialize select2 with proper container
            $(datasetSelect).select2({
                dropdownParent: $('#dataset_file_div'),
                width: '100%',
                placeholder: 'Select Dataset',
                allowClear: true,
                templateResult: (data) => {
                    if (!data.id) return data.text;
                    return $(`<span><i class="fas fa-file-code mr-2"></i>${data.text}</span>`);
                },
                templateSelection: (data) => {
                    if (!data.id) return data.text;
                    return $(`<span><i class="fas fa-file-code mr-2"></i>${data.text}</span>`);
                }
            });

            // Make sure container is visible
            document.getElementById('dataset_file_div').style.display = 'block';
        } catch (error) {
            console.error('Error loading dataset files:', error);
        }
    }

    // Function to update project tags
    function updateProjectTags() {
        const projectSelect = document.getElementById('life_app_project');
        const projectTagsContainer = document.getElementById('life-app-project-tags');
        if (!projectSelect || !projectTagsContainer) return;
        projectTagsContainer.innerHTML = '';

        Array.from(projectSelect.selectedOptions).forEach(option => {
            const tag = document.createElement('span');
            tag.className = 'bg-blue-200 text-blue-800 text-sm font-semibold mr-2 px-2.5 py-0.5 rounded dark:bg-blue-200 dark:text-blue-900';
            tag.textContent = option.textContent;

            const removeBtn = document.createElement('button');
            removeBtn.className = 'ml-1 text-blue-800 hover:text-blue-600 focus:outline-none';
            removeBtn.innerHTML = '&times;';
            removeBtn.onclick = () => {
                // Deselect the option in the dropdown
                option.selected = false;
                projectSelect.dispatchEvent(new Event('change')); // Trigger change event
                updateProjectTags(); // Refresh tags
            };
            tag.appendChild(removeBtn);
            projectTagsContainer.appendChild(tag);
        });
    }

    // Event listener for project selection change
    document.getElementById('life_app_project').addEventListener('change', () => {
        updateProjectTags();
    });

    // When LiFE App is selected, load projects/scripts
    document.getElementById('dataset_source').addEventListener('change', function() {
        if (this.value === 'life_app') {
            loadLifeAppProjects();
            loadLifeAppScripts();
            $('#dataset_file_div').show();
            loadDatasetFiles();
        } else {
            $('#dataset_file_div').hide();
        }
    });

    // --- On page load, hide LiFE App dataset source if not ASR ---
    document.getElementById('task').addEventListener('change', function() {
        const taskValue = this.value;
        const lifeAppOption = document.getElementById("dataset_source").querySelector('option[value="life_app"]');
        if (taskValue === "automatic-speech-recognition") {
            if (lifeAppOption) lifeAppOption.style.display = "";
        } else {
            if (lifeAppOption) lifeAppOption.style.display = "none";
            if (dataSource.value === "life_app") {
                dataSource.value = "local";
                handleDataSource();
            }
        }
    });

    handleDataSource(); // ADD THIS LINE
});