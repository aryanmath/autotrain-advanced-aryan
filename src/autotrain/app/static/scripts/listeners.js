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
        const dataSource = document.getElementById("dataset_source");
        const taskSelect = document.getElementById('task');
        const lifeAppSelection = document.getElementById("life-app-selection");
        const hubDataTabContent = document.getElementById('hub-data-tab-content');
        const uploadDataTabContent = document.getElementById("upload-data-tab-content");
        const uploadDataTabs = document.getElementById("upload-data-tabs");

        // Always hide both first
        if (lifeAppSelection) lifeAppSelection.style.display = "none";
        if (hubDataTabContent) hubDataTabContent.style.display = "none";
        if (uploadDataTabContent) uploadDataTabContent.style.display = "none";
        if (uploadDataTabs) uploadDataTabs.style.display = "none";

        if (dataSource.value === "life_app") {
            if (taskSelect.value !== "automatic-speech-recognition") {
                alert("LiFE App Dataset can only be used with Automatic Speech Recognition task.");
                dataSource.value = "local";
                if (uploadDataTabContent) uploadDataTabContent.style.display = "block";
                if (uploadDataTabs) uploadDataTabs.style.display = "block";
                return;
            }
            if (lifeAppSelection) lifeAppSelection.style.display = "block";

            // --- Multi-select with tags for Project ---
            const projectSelect = document.getElementById('life_app_project');
            const scriptSelect = document.getElementById('life_app_script');
            let tagContainer = document.getElementById('life-app-project-tags');
            if (!tagContainer) {
                tagContainer = document.createElement('div');
                tagContainer.id = 'life-app-project-tags';
                tagContainer.style.marginTop = '8px';
                projectSelect.parentElement.appendChild(tagContainer);
            } else {
                tagContainer.innerHTML = '';
            }
            let hiddenInput = document.getElementById('life_app_project_hidden');
            if (!hiddenInput) {
                hiddenInput = document.createElement('input');
                hiddenInput.type = 'hidden';
                hiddenInput.id = 'life_app_project_hidden';
                hiddenInput.name = 'life_app_project';
                projectSelect.parentElement.appendChild(hiddenInput);
            }

            // Fetch and populate projects
            fetch('/static/projectList.json')
                .then(response => response.json())
                .then(data => {
                    let availableProjects = [...data];
                    let selectedProjects = [];
                    function updateTags() {
                        tagContainer.innerHTML = '';
                        selectedProjects.forEach(project => {
                            const tag = document.createElement('span');
                            tag.textContent = project;
                            tag.style.display = 'inline-block';
                            tag.style.background = '#e5e7eb';
                            tag.style.color = '#111827';
                            tag.style.borderRadius = '12px';
                            tag.style.padding = '2px 10px 2px 8px';
                            tag.style.marginRight = '6px';
                            tag.style.marginBottom = '4px';
                            tag.style.fontSize = '0.95em';
                            tag.style.position = 'relative';
                            // Remove button
                            const removeBtn = document.createElement('span');
                            removeBtn.textContent = '×';
                            removeBtn.style.marginLeft = '8px';
                            removeBtn.style.cursor = 'pointer';
                            removeBtn.style.color = '#ef4444';
                            removeBtn.onclick = function() {
                                selectedProjects = selectedProjects.filter(p => p !== project);
                                availableProjects.push(project);
                                updateDropdown();
                                updateTags();
                            };
                            tag.appendChild(removeBtn);
                            tagContainer.appendChild(tag);
                        });
                        hiddenInput.value = selectedProjects.join(',');
                    }
                    function updateDropdown() {
                        projectSelect.innerHTML = '<option value=\"\">Select Project</option>';
                        availableProjects.forEach(project => {
                            const option = document.createElement('option');
                            option.value = project;
                            option.textContent = project;
                            projectSelect.appendChild(option);
                        });
                    }
                    updateDropdown();
                    updateTags();
                    projectSelect.onchange = function() {
                        const val = projectSelect.value;
                        if (val && !selectedProjects.includes(val)) {
                            selectedProjects.push(val);
                            availableProjects = availableProjects.filter(p => p !== val);
                            updateDropdown();
                            updateTags();
                        }
                        projectSelect.value = '';
                    };
                });

            // --- Script dropdown (single select) ---
            if (scriptSelect) {
                scriptSelect.innerHTML = '<option value=\"\">Select Script</option>';
                fetch('/static/scriptList.json')
                    .then(response => response.json())
                    .then(data => {
                        data.forEach(script => {
                            const option = document.createElement('option');
                            option.value = script;
                            option.textContent = script;
                            scriptSelect.appendChild(option);
                        });
                    });
            }
        } else if (dataSource.value === "hub") {
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

            // --- Project dropdown (multi-select) ---
            if (projectSelect) {
                console.log('Loading projects...');
                fetch('/static/projectList.json')
                    .then(response => {
                        console.log('Project list response:', response.status);
                        return response.json();
                    })
                    .then(data => {
                        console.log('Projects loaded:', data);
                        projectSelect.innerHTML = '<option value="">Select Project</option>';
                        data.forEach(project => {
                            const option = document.createElement('option');
                            option.value = project;
                            option.textContent = project;
                            projectSelect.appendChild(option);
                        });
                    })
                    .catch(error => {
                        console.error('Error loading projects:', error);
                        alert('Failed to load projects. Please check console for details.');
                    });
            }

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

    // Function to load projects
    async function loadLifeAppProjects() {
        const projectSelect = document.getElementById('life-app-project');
        const projectTagsContainer = document.getElementById('life-app-project-tags');
        projectSelect.innerHTML = ''; // Clear existing options
        projectTagsContainer.innerHTML = ''; // Clear existing tags

        try {
            const response = await fetch('/static/projectList.json');
            const projects = await response.json();
            projects.forEach(project => {
                const option = document.createElement('option');
                option.value = project.project_id;
                option.textContent = project.project_name;
                projectSelect.appendChild(option);
            });
            updateProjectTags(); // Update tags after loading projects
        } catch (error) {
            console.error('Error loading projects:', error);
            alert('Failed to load projects. Please check console for details.');
        }
    }

    // Function to load scripts based on selected projects
    async function loadLifeAppScripts() {
        const projectSelect = document.getElementById('life-app-project');
        const scriptSelect = document.getElementById('life-app-script');
        scriptSelect.innerHTML = '<option value="">Select Script</option>'; // Clear and reset

        const selectedProjectIds = Array.from(projectSelect.selectedOptions).map(option => option.value);

        if (selectedProjectIds.length === 0) {
            return; // No projects selected, so no scripts to load
        }

        try {
            const response = await fetch('/static/scriptList.json');
            const allScripts = await response.json();

            // Filter scripts based on selected projects
            const filteredScripts = allScripts.filter(script => 
                selectedProjectIds.includes(script.project_id)
            );

            filteredScripts.forEach(script => {
                const option = document.createElement('option');
                option.value = script.script_id;
                option.textContent = script.script_name;
                scriptSelect.appendChild(option);
            });
        } catch (error) {
            console.error('Error loading scripts:', error);
            alert('Failed to load scripts. Please check console for details.');
        }
    }

    // Function to update project tags
    function updateProjectTags() {
        const projectSelect = document.getElementById('life-app-project');
        const projectTagsContainer = document.getElementById('life-app-project-tags');
        projectTagsContainer.innerHTML = ''; // Clear existing tags

        Array.from(projectSelect.selectedOptions).forEach(option => {
            const tag = document.createElement('span');
            tag.className = 'bg-blue-200 text-blue-800 text-sm font-semibold mr-2 px-2.5 py-0.5 rounded dark:bg-blue-200 dark:text-blue-900';
            tag.textContent = option.textContent;

            const removeBtn = document.createElement('button');
            removeBtn.className = 'ml-1 text-blue-800 hover:text-blue-600 focus:outline-none';
            removeBtn.innerHTML = '&times;';
            removeBtn.onclick = () => {
                // Deselect the option in the dropdown
                for (let i = 0; i < projectSelect.options.length; i++) {
                    if (projectSelect.options[i].value === option.value) {
                        projectSelect.options[i].selected = false;
                        break;
                    }
                }
                updateProjectTags(); // Refresh tags
                loadLifeAppScripts(); // Re-filter scripts
            };
            tag.appendChild(removeBtn);
            projectTagsContainer.appendChild(tag);
        });
    }

    // Event listener for project selection change
    document.getElementById('life-app-project').addEventListener('change', () => {
        loadLifeAppScripts();
        updateProjectTags();
    });

    // Initial call to load projects if already on LiFE App selection (e.g., after a refresh)
    if (document.getElementById('dataset_source').value === 'life_app') {
        loadLifeAppProjects();
    }
});