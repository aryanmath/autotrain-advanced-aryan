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
        const datasetFileDiv = document.getElementById("dataset_file_div"); // ADD THIS LINE

        if (dataSource.value === "life_app") {
            // Show only LiFE App selection UI
            if (lifeAppSelection) lifeAppSelection.style.display = "block";
            // Show dataset file dropdown
            if (datasetFileDiv) datasetFileDiv.style.display = ""; // ADD THIS LINE
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
            if (datasetFileDiv) datasetFileDiv.style.display = "none"; // ADD THIS LINE
            if (uploadDataTabContent) uploadDataTabContent.style.display = "none";
            if (uploadDataTabs) uploadDataTabs.style.display = "none";
        } else if (dataSource.value === "local") {
            // Show local upload sections
            if (uploadDataTabContent) uploadDataTabContent.style.display = "block";
            if (uploadDataTabs) uploadDataTabs.style.display = "block";
            // Hide LiFE App and hub dataset sections
            if (hubDataTabContent) hubDataTabContent.style.display = "none";
            if (lifeAppSelection) lifeAppSelection.style.display = "none";
            if (datasetFileDiv) datasetFileDiv.style.display = "none"; // ADD THIS LINE
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
                        ${options}ig.options.map(option => `<option value="${option}" ${option === config.default ? 'selected' : ''}>${option}</option>`).join('');
                    </select>iv>
                </div>`;el for="param_${param}" class="text-sm font-medium text-gray-700 dark:text-gray-300">${config.label}</label>
                break;elect name="param_${param}" id="param_${param}"
            case 'checkbox':s="mt-1 p-1 text-xs font-medium w-full border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500">
                element = `<div>s}
                    <label for="param_${param}" class="text-sm font-medium text-gray-700 dark:text-gray-300">${config.label}</label>
                    <input type="checkbox" name="param_${param}" id="param_${param}" ${config.default ? 'checked' : ''}
                        class="mt-1 text-xs font-medium border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500">
                </div>`;ox':
                break;t = `<div>
            case 'string': for="param_${param}" class="text-sm font-medium text-gray-700 dark:text-gray-300">${config.label}</label>
                element = `<div>"checkbox" name="param_${param}" id="param_${param}" ${config.default ? 'checked' : ''}
                    <label for="param_${param}" class="text-sm font-medium text-gray-700 dark:text-gray-300">${config.label}</label>adow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500">
                    <input type="text" name="param_${param}" id="param_${param}" value="${config.default}"
                        class="mt-1 p-1 text-xs font-medium w-full border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500">
                </div>`;':
                break;t = `<div>
        }           <label for="param_${param}" class="text-sm font-medium text-gray-700 dark:text-gray-300">${config.label}</label>
        return element;put type="text" name="param_${param}" id="param_${param}" value="${config.default}"
    }                   class="mt-1 p-1 text-xs font-medium w-full border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500">
                </div>`;
    function renderUI(params) {
        const uiContainer = document.getElementById('dynamic-ui');
        let rowDiv = null;
        let rowIndex = 0;
        let lastType = null;
    function renderUI(params) {
        Object.keys(params).forEach((param, index) => {namic-ui');
            const config = params[param];
            if (lastType !== config.type || rowIndex >= 3) {
                if (rowDiv) uiContainer.appendChild(rowDiv);
                rowDiv = document.createElement('div');
                rowDiv.className = 'grid grid-cols-3 gap-2 mb-2';
                rowIndex = 0;rams[param];
            }f (lastType !== config.type || rowIndex >= 3) {
            rowDiv.innerHTML += createElement(param, config);
            rowIndex++;= document.createElement('div');
            lastType = config.type;'grid grid-cols-3 gap-2 mb-2';
        });     rowIndex = 0;
        if (rowDiv) uiContainer.appendChild(rowDiv);
    }       rowDiv.innerHTML += createElement(param, config);
            rowIndex++;
            lastType = config.type;
    fetchParams().then(params => renderUI(params));
    document.getElementById('task').addEventListener('change', function () {
        fetchParams().then(params => {
            document.getElementById('dynamic-ui').innerHTML = '';
            let jsonCheckBoxFlag = false;
            if (jsonCheckbox.checked) {UI(params));
                jsonCheckbox.checked = false;istener('change', function () {
                jsonCheckBoxFlag = true;
            document.getElementById('dynamic-ui').innerHTML = '';
            }et jsonCheckBoxFlag = false;
            renderUI(params);checked) {
            if (jsonCheckBoxFlag) {d = false;
                jsonCheckbox.checked = true;
                updateTextarea();
                observeParamChanges();
            }enderUI(params);
        }); if (jsonCheckBoxFlag) {
    });         jsonCheckbox.checked = true;
    document.getElementById('parameter_mode').addEventListener('change', function () {
        fetchParams().then(params => {
            document.getElementById('dynamic-ui').innerHTML = '';
            let jsonCheckBoxFlag = false;
            if (jsonCheckbox.checked) {
                jsonCheckbox.checked = false;.addEventListener('change', function () {
                jsonCheckBoxFlag = true;
            document.getElementById('dynamic-ui').innerHTML = '';
            }et jsonCheckBoxFlag = false;
            renderUI(params);checked) {
            if (jsonCheckBoxFlag) {d = false;
                jsonCheckbox.checked = true;
                updateTextarea();
                observeParamChanges();
            }enderUI(params);
        }); if (jsonCheckBoxFlag) {
    });         jsonCheckbox.checked = true;
                updateTextarea();
    jsonCheckbox.addEventListener('change', function () {
        if (jsonCheckbox.checked) {
            updateTextarea();
            observeParamChanges();
        }
    });nCheckbox.addEventListener('change', function () {
    document.getElementById('task').addEventListener('change', function () {
        if (jsonCheckbox.checked) {
            updateTextarea();es();
            observeParamChanges();
        }
    });ument.getElementById('task').addEventListener('change', function () {
    // Attach event listeners to dataset_source dropdown
    dataSource.addEventListener("change", handleDataSource);
    jsonCheckbox.addEventListener('change', switchToJSON);
    paramsTextarea.addEventListener('input', updateParamsFromTextarea);
    });
    // Trigger the event listener to set the initial state
    handleDataSource();Listener("change", handleDataSource);
    observeParamChanges();istener('change', switchToJSON);
    updateTextarea();dEventListener('input', updateParamsFromTextarea);

    // Add event listener for task changeshe initial state
    document.getElementById('task').addEventListener('change', function() {
        if (dataSource.value === "life_app" && this.value !== "automatic-speech-recognition") {
            alert("LiFE app datasets can only be used with Automatic Speech Recognition tasks");
            dataSource.value = "local";
            handleDataSource();ask changes
        }ent.getElementById('task').addEventListener('change', function() {
    }); if (dataSource.value === "life_app" && this.value !== "automatic-speech-recognition") {
            alert("LiFE app datasets can only be used with Automatic Speech Recognition tasks");
    // LiFE App Dataset Selectionocal";
    document.getElementById('life-app-source').addEventListener('change', function() {
        const apiContainer = document.getElementById('life-app-api-container');
        const jsonContainer = document.getElementById('life-app-json-container');
        
        if (this.value === 'api') {
            apiContainer.style.display = 'block';dEventListener('change', function() {
            jsonContainer.style.display = 'none';yId('life-app-api-container');
        } else if (this.value === 'json') {lementById('life-app-json-container');
            apiContainer.style.display = 'none';
            jsonContainer.style.display = 'block';
        } else {ontainer.style.display = 'block';
            apiContainer.style.display = 'none';;
            jsonContainer.style.display = 'none';
        }   apiContainer.style.display = 'none';
    });     jsonContainer.style.display = 'block';
        } else {
    // Handle JSON file selectionsplay = 'none';
    document.getElementById('life-app-json-file').addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                try {ntById('life-app-json-file').addEventListener('change', function(e) {
                    const data = JSON.parse(e.target.result);
                    if (!Array.isArray(data)) {
                        throw new Error('JSON must be an array');
                    }load = function(e) {
                    if (data.length === 0) {
                        throw new Error('JSON array is empty');
                    }f (!Array.isArray(data)) {
                    const firstItem = data[0];must be an array');
                    if (!firstItem.audio || !firstItem.transcription) {
                        throw new Error('Each item must have audio and transcription fields');
                    }   throw new Error('JSON array is empty');
                    // Store the validated data
                    window.lifeAppData = data;
                } catch (error) {m.audio || !firstItem.transcription) {
                    alert('Invalid JSON file: ' + error.message);o and transcription fields');
                    e.target.value = '';
                }   // Store the validated data
            };      window.lifeAppData = data;
            reader.readAsText(file);
        }           alert('Invalid JSON file: ' + error.message);
    });             e.target.value = '';
                }
    // Update dataset source change handler
    document.getElementById('dataset_source').addEventListener('change', function() {
        console.log('Dataset source changed to:', this.value);
        const lifeAppSelection = document.getElementById('life-app-selection');
        const projectSelect = document.getElementById('life_app_project');
        const scriptSelect = document.getElementById('life_app_script');
        const tagContainer = document.getElementById('life-app-project-tags');ion() {
        console.log('Dataset source changed to:', this.value);
        if (this.value === 'life_app') {t.getElementById('life-app-selection');
            console.log('Showing LiFE App selection');'life_app_project');
            lifeAppSelection.style.display = 'block';'life_app_script');
        const tagContainer = document.getElementById('life-app-project-tags');
            // --- Script dropdown (single select) ---
            if (scriptSelect) {e_app') {
                console.log('Loading scripts...');n');
                fetch('/static/scriptList.json')ock';
                    .then(response => {
                        console.log('Script list response:', response.status);
                        return response.json();
                    })e.log('Loading scripts...');
                    .then(data => {ptList.json')
                        console.log('Scripts loaded:', data);
                        scriptSelect.innerHTML = '<option value="">Select Script</option>';
                        data.forEach(script => {
                            const option = document.createElement('option');
                            option.value = script;
                            option.textContent = script;ata);
                            scriptSelect.appendChild(option);ue="">Select Script</option>';
                        });a.forEach(script => {
                    })      const option = document.createElement('option');
                    .catch(error => {lue = script;
                        console.error('Error loading scripts:', error);
                        alert('Failed to load scripts. Please check console for details.');
                    }); });
            }       })
        } else {    .catch(error => {
            lifeAppSelection.style.display = 'none'; scripts:', error);
        }               alert('Failed to load scripts. Please check console for details.');
    });             });
            }
    // --- LiFE App Dataset Selection: Show/Hide and Populate ---
    document.getElementById('dataset_source').addEventListener('change', function() {
        const lifeAppSelection = document.getElementById('life-app-selection');
        const projectSelect = document.getElementById('life_app_project');
        const scriptSelect = document.getElementById('life_app_script');
        const tagContainer = document.getElementById('life-app-project-tags');
    document.getElementById('dataset_source').addEventListener('change', function() {
        if (this.value === 'life_app') {t.getElementById('life-app-selection');
            lifeAppSelection.style.display = 'block';('life_app_project');
        const scriptSelect = document.getElementById('life_app_script');
            // --- Script dropdown (single select) ---life-app-project-tags');
            if (scriptSelect) {
                fetch('/static/scriptList.json')
                    .then(response => response.json())
                    .then(data => {
                        scriptSelect.innerHTML = '<option value="">Select Script</option>';
                        data.forEach(script => {
                            const option = document.createElement('option');
                            option.value = script;n())
                            option.textContent = script;
                            scriptSelect.appendChild(option);ue="">Select Script</option>';
                        });a.forEach(script => {
                    })      const option = document.createElement('option');
                    .catch(error => {lue = script;
                        console.error('Error loading scripts:', error);
                        alert('Failed to load scripts. Please check console for details.');
                    }); });
            }       })
        } else {    .catch(error => {
            lifeAppSelection.style.display = 'none'; scripts:', error);
        }               alert('Failed to load scripts. Please check console for details.');
    });             });
            }
    // --- Project multi-select tags update ---
    document.getElementById('life_app_project').addEventListener('change', function() {
        const projectSelect = document.getElementById('life_app_project');
        const tagContainer = document.getElementById('life-app-project-tags');
        tagContainer.innerHTML = '';
        Array.from(projectSelect.selectedOptions).forEach(option => {
            const tag = document.createElement('span');tListener('change', function() {
            tag.className = 'bg-blue-200 text-blue-800 text-sm font-semibold mr-2 px-2.5 py-0.5 rounded dark:bg-blue-200 dark:text-blue-900';
            tag.textContent = option.textContent;yId('life-app-project-tags');
            const removeBtn = document.createElement('button');
            removeBtn.className = 'ml-1 text-blue-800 hover:text-blue-600 focus:outline-none';
            removeBtn.innerHTML = '&times;';nt('span');
            removeBtn.onclick = () => {0 text-blue-800 text-sm font-semibold mr-2 px-2.5 py-0.5 rounded dark:bg-blue-200 dark:text-blue-900';
                option.selected = false;tContent;
                updateProjectTags(); // Refresh tags('button');
            };moveBtn.className = 'ml-1 text-blue-800 hover:text-blue-600 focus:outline-none';
            tag.appendChild(removeBtn);es;';
            tagContainer.appendChild(tag);
        });     option.selected = false;
    });         updateProjectTags(); // Refresh tags
            };
    // Function to load projectsveBtn);
    async function loadLifeAppProjects() {
        const projectSelect = document.getElementById('life_app_project');
        const projectTagsContainer = document.getElementById('life-app-project-tags');
        if (!projectSelect) return;
        projectSelect.innerHTML = '';
    async function loadLifeAppProjects() {
        try { projectSelect = document.getElementById('life_app_project');
            const response = await fetch('/static/projectList.json');p-project-tags');
            const projects = await response.json();
            projects.forEach(project => {
                const option = document.createElement('option');
                option.value = project;
                option.textContent = project;atic/projectList.json');
                projectSelect.appendChild(option);;
            });jects.forEach(project => {
            // Trigger change event to update tagsent('option');
            updateProjectTags();roject;
        } catch (error) {xtContent = project;
            console.error('Error loading projects:', error);
        }   });
    }       // Trigger change event to update tags
            updateProjectTags();
    // Function to load scripts
    async function loadLifeAppScripts() {projects:', error);
        const scriptSelect = document.getElementById('life_app_script');
        if (!scriptSelect) return;
        scriptSelect.innerHTML = '<option value="">Select Script</option>';
        try {on to load scripts
            const response = await fetch('/static/scriptList.json');
            const scripts = await response.json();Id('life_app_script');
            scripts.forEach(script => {
                const option = document.createElement('option');</option>';
                option.value = script;
                option.textContent = script;tatic/scriptList.json');
                scriptSelect.appendChild(option);;
            });ipts.forEach(script => {
        } catch (error) {ion = document.createElement('option');
            console.error('Error loading scripts:', error);
        }       option.textContent = script;
    }           scriptSelect.appendChild(option);
            });
    // Function to load dataset files
    async function loadDatasetFiles() {g scripts:', error);
        const datasetSelect = document.getElementById('dataset_file');
        if (!datasetSelect) return;
        datasetSelect.innerHTML = '<option value="">Select Dataset File</option>';
    // Function to load dataset files
        try {ction loadDatasetFiles() {
            const datasetFiles = ["dataset.json"]; // Directly use the datasetFiles array
            datasetFiles.forEach(datasetFile => {
                const option = document.createElement('option');et File</option>';
                option.value = datasetFile;
                option.textContent = datasetFile;
                datasetSelect.appendChild(option); // Directly use the datasetFiles array
            });asetFiles.forEach(datasetFile => {
        } catch (error) {ion = document.createElement('option');
            console.error('Error loading dataset files:', error);
        }       option.textContent = datasetFile;
    }           datasetSelect.appendChild(option);
            });
    // Function to update project tags
    function updateProjectTags() {oading dataset files:', error);
        const projectSelect = document.getElementById('life_app_project');
        const projectTagsContainer = document.getElementById('life-app-project-tags');
        if (!projectSelect || !projectTagsContainer) return;
        projectTagsContainer.innerHTML = '';
    function updateProjectTags() {
        Array.from(projectSelect.selectedOptions).forEach(option => {ct');
            const tag = document.createElement('span');tById('life-app-project-tags');
            tag.className = 'bg-blue-200 text-blue-800 text-sm font-semibold mr-2 px-2.5 py-0.5 rounded dark:bg-blue-200 dark:text-blue-900';
            tag.textContent = option.textContent;

            const removeBtn = document.createElement('button');n => {
            removeBtn.className = 'ml-1 text-blue-800 hover:text-blue-600 focus:outline-none';
            removeBtn.innerHTML = '&times;';t-blue-800 text-sm font-semibold mr-2 px-2.5 py-0.5 rounded dark:bg-blue-200 dark:text-blue-900';
            removeBtn.onclick = () => {xtContent;
                // Deselect the option in the dropdown
                option.selected = false;reateElement('button');
                projectSelect.dispatchEvent(new Event('change')); // Trigger change eventone';
                updateProjectTags(); // Refresh tags
            };moveBtn.onclick = () => {
            tag.appendChild(removeBtn);in the dropdown
            projectTagsContainer.appendChild(tag);
        });     projectSelect.dispatchEvent(new Event('change')); // Trigger change event
    }           updateProjectTags(); // Refresh tags
            };
    // Event listener for project selection change
    document.getElementById('life_app_project').addEventListener('change', () => {
        updateProjectTags();
    });

    // When LiFE App is selected, load projects/scripts
    document.getElementById('dataset_source').addEventListener('change', function() {
        const datasetFileDiv = document.getElementById('dataset_file_div'); // ADD THIS LINE
        if (this.value === 'life_app') {
            loadLifeAppProjects();
            loadLifeAppScripts(); load projects/scripts
            loadDatasetFiles();taset_source').addEventListener('change', function() {
            if (datasetFileDiv) { // ADD THIS CONDITION'dataset_file_div'); // ADD THIS LINE
                datasetFileDiv.style.display = 'block'; // CHANGED FROM '' TO 'block'
            }oadLifeAppProjects();
        } else {LifeAppScripts();
            if (datasetFileDiv) { // ADD THIS CONDITION
                datasetFileDiv.style.display = 'none';N
            }   datasetFileDiv.style.display = 'block'; // CHANGED FROM '' TO 'block'
        }   }
    });     handleDataSource();
        } else {
    // --- On page load, hide LiFE App dataset source if not ASR ---
    document.getElementById('task').addEventListener('change', function() {
        const taskValue = this.value;
        const lifeAppOption = document.getElementById("dataset_source").querySelector('option[value="life_app"]');
        if (taskValue === "automatic-speech-recognition") {
            if (lifeAppOption) lifeAppOption.style.display = "";
        } else {
            if (lifeAppOption) lifeAppOption.style.display = "none";
            if (dataSource.value === "life_app") {er('change', function() {
                dataSource.value = "local";
                handleDataSource();ent.getElementById("dataset_source").querySelector('option[value="life_app"]');
            }askValue === "automatic-speech-recognition") {
        }   if (lifeAppOption) lifeAppOption.style.display = "";
    }); } else {
            if (lifeAppOption) lifeAppOption.style.display = "none";
    handleDataSource(); // ADD THIS LINEfe_app") {
});             dataSource.value = "local";
                handleDataSource();
            }
        }
    });

    handleDataSource(); // ADD THIS LINE
});