document.addEventListener('DOMContentLoaded', function () {
    const dataSource = document.getElementById("dataset_source");
    const uploadDataTabContent = document.getElementById("upload-data-tab-content");
    const hubDataTabContent = document.getElementById("hub-data-tab-content");
    const uploadDataTabs = document.getElementById("upload-data-tabs");
    const lifeAppDataTabContent = document.getElementById('life-app-data-tab-content');

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
        const dataSource = document.getElementById('dataset_source');
        const uploadDataTabContent = document.getElementById('upload-data-tab-content');
        const uploadDataTabs = document.getElementById('upload-data-tabs');
        const hubDataTabContent = document.getElementById('hub-data-tab-content');
        const lifeAppDataTabContent = document.getElementById('life-app-data-tab-content');

        if (dataSource.value === "hub") {
            uploadDataTabContent.style.display = "none";
            uploadDataTabs.style.display = "none";
            hubDataTabContent.style.display = "block";
            lifeAppDataTabContent.style.display = "none";
        } else if (dataSource.value === "life_app") {
            uploadDataTabContent.style.display = "none";
            uploadDataTabs.style.display = "none";
            hubDataTabContent.style.display = "none";
            lifeAppDataTabContent.style.display = "block";
        } else if (dataSource.value === "local") {
            uploadDataTabContent.style.display = "block";
            uploadDataTabs.style.display = "block";
            hubDataTabContent.style.display = "none";
            lifeAppDataTabContent.style.display = "none";
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
});

document.getElementById('start-training-button').addEventListener('click', async function() {
    const formData = new FormData();
    formData.append('project_name', document.getElementById('project_name').value);
    formData.append('task', document.getElementById('task').value);
    formData.append('base_model', document.getElementById('base_model').value);
    formData.append('hardware', document.getElementById('hardware').value);
    formData.append('params', document.getElementById('params_json').value);
    formData.append('autotrain_user', document.getElementById('autotrain_user').value);
    formData.append('column_mapping', JSON.stringify(columnMapping));
    formData.append('dataset_source', document.getElementById('dataset_source').value);

    const dataSource = document.getElementById('dataset_source').value;
    if (dataSource === 'life_app') {
        formData.append('life_app_api_key', document.getElementById('life_app_api_key').value);
        formData.append('life_app_dataset_id', document.getElementById('life_app_dataset_id').value);
    } else if (dataSource === 'hub') {
        formData.append('hub_dataset', document.getElementById('hub_dataset').value);
        formData.append('train_split', document.getElementById('train_split').value);
        formData.append('valid_split', document.getElementById('valid_split').value);
    } else {
        const trainingFiles = document.getElementById('data_files_training').files;
        const validationFiles = document.getElementById('data_files_valid').files;
        for (let i = 0; i < trainingFiles.length; i++) {
            formData.append('data_files_training', trainingFiles[i]);
        }
        for (let i = 0; i < validationFiles.length; i++) {
            formData.append('data_files_valid', validationFiles[i]);
        }
    }

    try {
        const response = await fetch('/ui/create_project', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (data.success) {
            window.location.href = data.monitor_url;
        } else {
            alert('Error: ' + data.detail);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while submitting the form.');
    }
});