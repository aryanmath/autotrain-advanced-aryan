from flask import Blueprint, render_template, request, jsonify
from autotrain.trainers.automatic_speech_recognition.life_app.data_fetcher import LifeAppDataFetcher
from autotrain.trainers.automatic_speech_recognition.life_app.preprocessor import LifeAppASRPreprocessor
import os
import json

life_app_bp = Blueprint('life_app', __name__)

@life_app_bp.route('/life-app-dataset', methods=['GET'])
def get_dataset_ui():
    """Render the LiFE App dataset selection UI."""
    return render_template('life_app_dataset.html')

@life_app_bp.route('/life-app-dataset/validate', methods=['POST'])
def validate_dataset():
    """Validate the selected dataset source."""
    try:
        data = request.json
        source_type = data.get('type')
        
        if source_type == 'api':
            # Validate API connection
            api_url = data.get('api_url')
            api_token = data.get('api_token')
            
            fetcher = LifeAppDataFetcher(api_url=api_url, api_token=api_token)
            sample_data = fetcher.fetch_data()
            
            return jsonify({
                'status': 'success',
                'message': 'API connection successful',
                'sample_size': len(sample_data)
            })
            
        elif source_type == 'json':
            # Validate JSON file
            file = request.files.get('file')
            if not file:
                return jsonify({
                    'status': 'error',
                    'message': 'No file provided'
                }), 400
                
            # Save file temporarily
            temp_path = os.path.join('temp', file.filename)
            os.makedirs('temp', exist_ok=True)
            file.save(temp_path)
            
            # Validate file
            fetcher = LifeAppDataFetcher(json_file_path=temp_path)
            sample_data = fetcher.fetch_data()
            
            # Clean up
            os.remove(temp_path)
            
            return jsonify({
                'status': 'success',
                'message': 'JSON file validated',
                'sample_size': len(sample_data)
            })
            
        else:
            return jsonify({
                'status': 'error',
                'message': 'Invalid source type'
            }), 400
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@life_app_bp.route('/life-app-dataset/prepare', methods=['POST'])
def prepare_dataset():
    """Prepare the dataset for training."""
    try:
        data = request.json
        source_type = data.get('type')
        project_name = data.get('project_name')
        username = data.get('username')
        token = data.get('token')
        
        if source_type == 'api':
            api_url = data.get('api_url')
            api_token = data.get('api_token')
            
            preprocessor = LifeAppASRPreprocessor(
                api_url=api_url,
                api_token=api_token,
                project_name=project_name,
                username=username,
                token=token
            )
            
        elif source_type == 'json':
            json_file = data.get('json_file')
            
            preprocessor = LifeAppASRPreprocessor(
                json_file_path=json_file,
                project_name=project_name,
                username=username,
                token=token
            )
            
        else:
            return jsonify({
                'status': 'error',
                'message': 'Invalid source type'
            }), 400
            
        # Prepare dataset
        data_path = preprocessor.prepare()
        
        return jsonify({
            'status': 'success',
            'message': 'Dataset prepared successfully',
            'data_path': data_path
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500 