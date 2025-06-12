// LiFE App Dataset Selection Handler
class LifeAppDatasetHandler {
    constructor() {
        this.datasetSelect = document.getElementById('life-app-dataset-select');
        this.apiUrlInput = document.getElementById('life-app-api-url');
        this.apiTokenInput = document.getElementById('life-app-api-token');
        this.jsonFileInput = document.getElementById('life-app-json-file');
        
        this.initializeEventListeners();
    }
    
    initializeEventListeners() {
        if (this.datasetSelect) {
            this.datasetSelect.addEventListener('change', () => this.handleDatasetSelection());
        }
        
        if (this.jsonFileInput) {
            this.jsonFileInput.addEventListener('change', () => this.handleFileSelection());
        }
    }
    
    handleDatasetSelection() {
        const selectedOption = this.datasetSelect.value;
        
        // Hide all input containers
        this.hideAllInputs();
        
        // Show relevant input based on selection
        switch(selectedOption) {
            case 'api':
                this.showApiInputs();
                break;
            case 'json':
                this.showJsonInput();
                break;
            default:
                // Handle default case
                break;
        }
    }
    
    hideAllInputs() {
        const containers = [
            'api-input-container',
            'json-input-container'
        ];
        
        containers.forEach(id => {
            const container = document.getElementById(id);
            if (container) {
                container.style.display = 'none';
            }
        });
    }
    
    showApiInputs() {
        const container = document.getElementById('api-input-container');
        if (container) {
            container.style.display = 'block';
        }
    }
    
    showJsonInput() {
        const container = document.getElementById('json-input-container');
        if (container) {
            container.style.display = 'block';
        }
    }
    
    handleFileSelection() {
        const file = this.jsonFileInput.files[0];
        if (file) {
            // Validate file type
            if (file.type !== 'application/json') {
                alert('Please select a JSON file');
                this.jsonFileInput.value = '';
                return;
            }
            
            // Read and validate JSON content
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const data = JSON.parse(e.target.result);
                    // Validate JSON structure
                    if (!this.validateJsonStructure(data)) {
                        alert('Invalid JSON structure. Please check the file format.');
                        this.jsonFileInput.value = '';
                    }
                } catch (error) {
                    alert('Error reading JSON file: ' + error.message);
                    this.jsonFileInput.value = '';
                }
            };
            reader.readAsText(file);
        }
    }
    
    validateJsonStructure(data) {
        // Check if data is an array
        if (!Array.isArray(data)) {
            return false;
        }
        
        // Check each item in the array
        return data.every(item => {
            return (
                typeof item === 'object' &&
                'audio' in item &&
                'transcription' in item &&
                typeof item.audio === 'string' &&
                typeof item.transcription === 'string'
            );
        });
    }
    
    getDatasetConfig() {
        const selectedOption = this.datasetSelect.value;
        
        switch(selectedOption) {
            case 'api':
                return {
                    type: 'api',
                    api_url: this.apiUrlInput.value,
                    api_token: this.apiTokenInput.value
                };
            case 'json':
                return {
                    type: 'json',
                    json_file: this.jsonFileInput.files[0]?.name
                };
            default:
                return null;
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.lifeAppDatasetHandler = new LifeAppDatasetHandler();
}); 