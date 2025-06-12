document.addEventListener('DOMContentLoaded', function () {
    window.submitLifeAppTraining = function () {
        const loadingSpinner = document.getElementById('loading-spinner');
        loadingSpinner.classList.remove('hidden');

        const formData = new FormData();
        formData.append('project_name', document.getElementById('project_name').value);
        formData.append('task', document.getElementById('task').value);
        formData.append('base_model', document.getElementById('base_model').value);
        formData.append('hardware', document.getElementById('hardware').value);
        formData.append('params', document.getElementById('params').value);
        formData.append('autotrain_user', document.getElementById('autotrain_user').value);

        const projectSelect = document.getElementById('life_app_project');
        const scriptSelect = document.getElementById('life_app_script');
        const datasetSelect = document.getElementById('life_app_dataset');
        
        // Get selected project tags
        const selectedTags = Array.from(document.querySelectorAll('#life-app-project-tags input:checked'))
            .map(input => input.value);
            
        formData.append('life-app-project', JSON.stringify(selectedTags));
        formData.append('life-app-script', scriptSelect.value);
        formData.append('life-app-dataset-file', datasetSelect.value);

        const endpoint = '/ui/start_life_app_training';

        const xhr = new XMLHttpRequest();
        xhr.open('POST', endpoint, true);

        xhr.onload = function () {
            loadingSpinner.classList.add('hidden');
            var finalModalContent = document.querySelector('#final-modal .text-center');

            if (xhr.status === 200) {
                var responseObj = JSON.parse(xhr.responseText);
                if (responseObj.status === 'success') {
                    var monitorURL = responseObj.monitor_url;
                    if (monitorURL && monitorURL.startsWith('http')) {
                        finalModalContent.innerHTML = '<p>Success!</p>' +
                            '<p>You can check the progress of your training here: <a href="' + monitorURL + '" target="_blank">' + monitorURL + '</a></p>';
                    } else {
                        finalModalContent.innerHTML = '<p>Success!</p>' +
                            '<p>' + (monitorURL || responseObj.message) + '</p>';
                    }
                } else {
                    finalModalContent.innerHTML = '<p>Error: ' + responseObj.message + '</p>';
                }
                showFinalModal();
            } else {
                finalModalContent.innerHTML = '<p>Error: ' + xhr.status + ' ' + xhr.statusText + '</p>' + 
                    '<p>Please check the logs for more information.</p>';
                console.error('Error:', xhr.status, xhr.statusText);
                showFinalModal();
            }
        };

        xhr.send(formData);
    };
}); 