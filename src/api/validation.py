#!/usr/bin/env python3
"""
Input Validation Module for CineSync v2 APIs
Provides validation functions and decorators for API endpoints
"""

import re
import logging
from functools import wraps
from typing import Any, Dict, List, Optional, Callable, Union
from flask import request, jsonify

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom validation error"""
    def __init__(self, message: str, field: str = None):
        self.message = message
        self.field = field
        super().__init__(self.message)


def sanitize_string(value: str, max_length: int = 1000, allow_html: bool = False) -> str:
    """
    Sanitize a string input.

    Args:
        value: Input string
        max_length: Maximum allowed length
        allow_html: Whether to allow HTML tags

    Returns:
        Sanitized string
    """
    if not isinstance(value, str):
        raise ValidationError("Expected string value", "type")

    # Trim and limit length
    value = value.strip()[:max_length]

    # Remove HTML tags if not allowed
    if not allow_html:
        value = re.sub(r'<[^>]+>', '', value)

    # Remove null bytes and control characters
    value = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', value)

    return value


def validate_int(value: Any, min_val: int = None, max_val: int = None, field_name: str = "value") -> int:
    """
    Validate and convert to integer.

    Args:
        value: Input value
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        field_name: Field name for error messages

    Returns:
        Validated integer
    """
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{field_name} must be a valid integer", field_name)

    if min_val is not None and result < min_val:
        raise ValidationError(f"{field_name} must be at least {min_val}", field_name)

    if max_val is not None and result > max_val:
        raise ValidationError(f"{field_name} must be at most {max_val}", field_name)

    return result


def validate_float(value: Any, min_val: float = None, max_val: float = None, field_name: str = "value") -> float:
    """
    Validate and convert to float.
    """
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{field_name} must be a valid number", field_name)

    if min_val is not None and result < min_val:
        raise ValidationError(f"{field_name} must be at least {min_val}", field_name)

    if max_val is not None and result > max_val:
        raise ValidationError(f"{field_name} must be at most {max_val}", field_name)

    return result


def validate_bool(value: Any, field_name: str = "value") -> bool:
    """
    Validate and convert to boolean.
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        if value.lower() in ('true', '1', 'yes', 'on'):
            return True
        if value.lower() in ('false', '0', 'no', 'off'):
            return False

    if isinstance(value, (int, float)):
        return bool(value)

    raise ValidationError(f"{field_name} must be a boolean value", field_name)


def validate_list(value: Any, item_validator: Callable = None, max_items: int = 1000, field_name: str = "value") -> List:
    """
    Validate a list.

    Args:
        value: Input value
        item_validator: Optional function to validate each item
        max_items: Maximum number of items
        field_name: Field name for error messages

    Returns:
        Validated list
    """
    if not isinstance(value, list):
        raise ValidationError(f"{field_name} must be a list", field_name)

    if len(value) > max_items:
        raise ValidationError(f"{field_name} cannot have more than {max_items} items", field_name)

    if item_validator:
        return [item_validator(item) for item in value]

    return value


def validate_model_name(name: str) -> str:
    """
    Validate a model name.
    """
    name = sanitize_string(name, max_length=100)

    # Only allow alphanumeric, underscores, and hyphens
    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
        raise ValidationError("Model name can only contain letters, numbers, underscores, and hyphens", "model_name")

    return name


def validate_user_id(user_id: Any) -> int:
    """
    Validate a user ID.
    """
    return validate_int(user_id, min_val=0, max_val=2**63, field_name="user_id")


def validate_content_type(content_type: str) -> str:
    """
    Validate content type.
    """
    valid_types = {'movie', 'tv', 'mixed', 'both'}
    content_type = sanitize_string(content_type, max_length=20).lower()

    if content_type not in valid_types:
        raise ValidationError(f"Content type must be one of: {', '.join(valid_types)}", "content_type")

    return content_type


def validate_genre(genre: str, valid_genres: List[str] = None) -> str:
    """
    Validate a genre.
    """
    genre = sanitize_string(genre, max_length=50)

    if valid_genres and genre not in valid_genres:
        raise ValidationError(f"Invalid genre: {genre}", "genre")

    return genre


def validate_rating(rating: Any) -> float:
    """
    Validate a rating value (1-5).
    """
    return validate_float(rating, min_val=1.0, max_val=5.0, field_name="rating")


def validate_json_request(required_fields: List[str] = None, optional_fields: Dict[str, Any] = None):
    """
    Decorator to validate JSON request data.

    Args:
        required_fields: List of required field names
        optional_fields: Dict of optional fields with default values
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                data = request.get_json()

                if data is None:
                    return jsonify({'success': False, 'error': 'Invalid JSON data'}), 400

                # Check required fields
                if required_fields:
                    for field in required_fields:
                        if field not in data:
                            return jsonify({
                                'success': False,
                                'error': f'Missing required field: {field}'
                            }), 400

                # Add optional fields with defaults
                if optional_fields:
                    for field, default in optional_fields.items():
                        if field not in data:
                            data[field] = default

                # Store validated data in request context
                request.validated_data = data

                return func(*args, **kwargs)

            except ValidationError as e:
                return jsonify({
                    'success': False,
                    'error': e.message,
                    'field': e.field
                }), 400

        return wrapper
    return decorator


def validate_query_params(params: Dict[str, Dict[str, Any]]):
    """
    Decorator to validate query parameters.

    Args:
        params: Dict mapping param names to validation specs
            e.g., {'limit': {'type': 'int', 'min': 1, 'max': 100, 'default': 10}}
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                validated = {}

                for param_name, spec in params.items():
                    value = request.args.get(param_name, spec.get('default'))

                    if value is None:
                        if spec.get('required', False):
                            return jsonify({
                                'success': False,
                                'error': f'Missing required parameter: {param_name}'
                            }), 400
                        continue

                    param_type = spec.get('type', 'str')

                    if param_type == 'int':
                        validated[param_name] = validate_int(
                            value,
                            min_val=spec.get('min'),
                            max_val=spec.get('max'),
                            field_name=param_name
                        )
                    elif param_type == 'float':
                        validated[param_name] = validate_float(
                            value,
                            min_val=spec.get('min'),
                            max_val=spec.get('max'),
                            field_name=param_name
                        )
                    elif param_type == 'bool':
                        validated[param_name] = validate_bool(value, field_name=param_name)
                    else:
                        validated[param_name] = sanitize_string(
                            str(value),
                            max_length=spec.get('max_length', 1000)
                        )

                request.validated_params = validated
                return func(*args, **kwargs)

            except ValidationError as e:
                return jsonify({
                    'success': False,
                    'error': e.message,
                    'field': e.field
                }), 400

        return wrapper
    return decorator


# Training preferences validation schema
TRAINING_PREFERENCES_SCHEMA = {
    'auto_retrain': {'type': 'bool'},
    'min_feedback_threshold': {'type': 'int', 'min': 1, 'max': 10000},
    'excluded_genres': {'type': 'list', 'max_items': 100},
    'excluded_users': {'type': 'list', 'max_items': 10000},
    'quality_filters': {'type': 'dict'}
}


def validate_training_preferences(data: Dict) -> Dict:
    """
    Validate training preferences data.
    """
    validated = {}

    for key, value in data.items():
        if key not in TRAINING_PREFERENCES_SCHEMA:
            continue  # Skip unknown fields

        spec = TRAINING_PREFERENCES_SCHEMA[key]

        if spec['type'] == 'bool':
            validated[key] = validate_bool(value, field_name=key)
        elif spec['type'] == 'int':
            validated[key] = validate_int(
                value,
                min_val=spec.get('min'),
                max_val=spec.get('max'),
                field_name=key
            )
        elif spec['type'] == 'list':
            validated[key] = validate_list(
                value,
                max_items=spec.get('max_items', 1000),
                field_name=key
            )
        elif spec['type'] == 'dict':
            if not isinstance(value, dict):
                raise ValidationError(f"{key} must be an object", key)
            validated[key] = value

    return validated


# Model upload validation
def validate_model_upload(file, model_name: str, model_type: str) -> tuple:
    """
    Validate model file upload.

    Returns:
        Tuple of (validated_model_name, validated_model_type)
    """
    # Validate model name
    model_name = validate_model_name(model_name)

    # Validate model type
    valid_model_types = {'ncf', 'sequential', 'two_tower', 'bert4rec', 'transformer', 'hybrid'}
    model_type = sanitize_string(model_type, max_length=50).lower()

    if model_type not in valid_model_types:
        raise ValidationError(f"Model type must be one of: {', '.join(valid_model_types)}", "model_type")

    # Validate file
    if file is None:
        raise ValidationError("No file provided", "file")

    if file.filename == '':
        raise ValidationError("No file selected", "file")

    # Check file extension
    allowed_extensions = {'.pt', '.pth', '.pkl', '.pickle'}
    file_ext = '.' + file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''

    if file_ext not in allowed_extensions:
        raise ValidationError(
            f"Invalid file type. Allowed: {', '.join(allowed_extensions)}",
            "file"
        )

    # Check file size (max 500MB)
    file.seek(0, 2)  # Seek to end
    size = file.tell()
    file.seek(0)  # Reset to beginning

    if size > 500 * 1024 * 1024:
        raise ValidationError("File too large. Maximum size is 500MB", "file")

    return model_name, model_type
