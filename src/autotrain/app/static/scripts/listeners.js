document.addEventListener('DOMContentLoaded', function () {
    const dataSource = document.getElementById("dataset_source");
    const taskSelect = document.getElementById('task');
    const jsonCheckbox = document.getElementById('show-json-parameters');
    const customTrainingConfigDiv = document.getElementById('custom_training_config');
    const jsonParametersDiv = document.getElementById('json-parameters');

    // Event listener for dataset source and task changes
    dataSource.addEventListener('change', handleDataSource);
    taskSelect.addEventListener('change', handleDataSource);

    // Initial call to set up the UI on page load
    handleDataSource();

    if (jsonCheckbox) {
        jsonCheckbox.addEventListener('change', function () {
            if (this.checked) {
                if (customTrainingConfigDiv) customTrainingConfigDiv.style.display = 'block';
                if (jsonParametersDiv) jsonParametersDiv.style.display = 'block';
            } else {
                if (customTrainingConfigDiv) customTrainingConfigDiv.style.display = 'none';
                if (jsonParametersDiv) jsonParametersDiv.style.display = 'none';
            }
        });
    }

    // Function to handle dynamic UI based on dataset source and task
    function handleDataSource() {
        const lifeAppSelection = document.getElementById("life-app-selection");
        const hubDataTabContent = document.getElementById("hub-data-tab-content");
        const uploadDataTabContent = document.getElementById("upload-data-tab-content");
        const uploadDataTabs = document.getElementById("upload-data-tabs");

        // Always hide all specific sections first
        if (lifeAppSelection) lifeAppSelection.style.display = "none";
        if (hubDataTabContent) hubDataTabContent.style.display = "none";
        if (uploadDataTabContent) uploadDataTabContent.style.display = "none";
        if (uploadDataTabs) uploadDataTabs.style.display = "none";

        if (dataSource.value === "life_app") {
            if (taskSelect.value !== "automatic-speech-recognition") {
                alert("LiFE App Dataset can only be used with Automatic Speech Recognition task.");
                dataSource.value = "local"; // Reset to local if task is not ASR
                // Then show local upload UI
                if (uploadDataTabContent) uploadDataTabContent.style.display = "block";
                if (uploadDataTabs) uploadDataTabs.style.display = "block";
                return; // Exit function after reset
            }
            // Show LiFE App UI and related components
            if (lifeAppSelection) lifeAppSelection.style.display = "block";
            loadLifeAppProjects(); // Load projects with tag functionality
            loadLifeAppScripts(); // Load scripts
            loadLifeAppDatasets(); // Load datasets into the dropdown

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

    // Function to load projects and handle multi-select tags
    async function loadLifeAppProjects() {
        const projectSelect = document.getElementById('life_app_project');
        let tagContainer = document.getElementById('life-app-project-tags');

        if (!tagContainer) {
            tagContainer = document.createElement('div');
            tagContainer.id = 'life-app-project-tags';
            tagContainer.className = 'mt-2 flex flex-wrap gap-2';
            if (projectSelect && projectSelect.parentElement) {
                projectSelect.parentElement.appendChild(tagContainer);
            }
        } else {
            tagContainer.innerHTML = ''; // Clear existing tags
        }

        let hiddenInput = document.getElementById('life_app_project_hidden');
        if (!hiddenInput) {
            hiddenInput = document.createElement('input');
            hiddenInput.type = 'hidden';
            hiddenInput.id = 'life_app_project_hidden';
            hiddenInput.name = 'life_app_project';
            if (projectSelect && projectSelect.parentElement) {
                projectSelect.parentElement.appendChild(hiddenInput);
            }
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
                        tag.className = 'inline-flex items-center px-2 py-1 rounded-full text-sm font-medium bg-gray-100 text-gray-800';
                        tag.innerHTML = `
                            ${project}
                            <button type="button" class="ml-1 text-gray-500 hover:text-red-500 focus:outline-none">
                                <span class="sr-only">Remove</span>
                                <svg class="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
                                    <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"/>
                                </svg>
                            </button>
                        `;

                        const removeBtn = tag.querySelector('button');
                        if (removeBtn) {
                            removeBtn.onclick = function() {
                                selectedProjects = selectedProjects.filter(p => p !== project);
                                availableProjects.push(project);
                                availableProjects.sort(); // Sort after adding back
                                updateDropdown();
                                updateTags();
                            };
                        }
                        tagContainer.appendChild(tag);
                    });
                    hiddenInput.value = selectedProjects.join(',');
                }

                function updateDropdown() {
                    projectSelect.innerHTML = '<option value="">Select Project</option>';
                    availableProjects.sort(); // Ensure dropdown is always sorted
                    availableProjects.forEach(project => {
                        const option = document.createElement('option');
                        option.value = project;
                        option.textContent = project;
                        projectSelect.appendChild(option);
                    });
                }

                // Initial setup
                updateDropdown();
                updateTags();

                // Handle project selection
                projectSelect.onchange = function() {
                    const selectedValue = projectSelect.value;
                    if (selectedValue && !selectedProjects.includes(selectedValue)) {
                        selectedProjects.push(selectedValue);
                        availableProjects = availableProjects.filter(p => p !== selectedValue);
                        updateDropdown();
                        updateTags();
                    }
                    projectSelect.value = ''; // Reset dropdown selection
                };
            })
            .catch(error => {
                console.error('Error loading projects:', error);
                alert('Failed to load projects. Please try again.');
            });
    }

    // Function to load scripts
    async function loadLifeAppScripts() {
        const scriptSelect = document.getElementById('life_app_script');
        if (scriptSelect) {
            scriptSelect.innerHTML = '<option value="">Select Script</option>'; // Clear and reset
            fetch('/static/scriptList.json')
                .then(response => response.json())
                .then(data => {
                    data.forEach(script => {
                        const option = document.createElement('option');
                        option.value = script;
                        option.textContent = script;
                        scriptSelect.appendChild(option);
                    });
                })
                .catch(error => {
                    console.error('Error loading scripts:', error);
                    alert('Failed to load scripts. Please try again.');
                });
        }
    }

    // Function to load datasets (re-added)
    async function loadLifeAppDatasets() {
        const datasetSelect = document.getElementById('life_app_dataset');
        if (datasetSelect) {
            datasetSelect.innerHTML = '<option value="">Select Dataset</option>';
            fetch('/static/dataset.json')
                .then(response => response.json())
                .then(data => {
                    data.forEach(dataset => {
                        const option = document.createElement('option');
                        // Store the entire object as a string and display the transcription
                        option.value = JSON.stringify(dataset);
                        option.textContent = dataset.transcription; 
                        datasetSelect.appendChild(option);
                    });
                })
                .catch(error => {
                    console.error('Error loading datasets:', error);
                    alert('Failed to load datasets. Please try again.');
                });
        }
    }
});