# CineSync Test Suite

This directory contains comprehensive tests for the CineSync movie recommendation system.

## Test Structure

The test suite covers the following modules:

### Core Tests
- `test_config.py` - Configuration loading and validation
- `test_models.py` - Hybrid recommender model functionality
- `test_database.py` - Database operations and utilities
- `test_inference.py` - Movie recommendation inference system
- `test_main.py` - Training pipeline and data processing

### Test Categories

#### Unit Tests
- Model forward/backward passes
- Configuration parsing
- Database connection and queries
- Data preprocessing functions
- ID mapping and validation

#### Integration Tests
- End-to-end recommendation generation
- Model saving and loading
- Database integration with models
- Training pipeline with real data

#### Mock Tests
- Database operations (avoiding real DB connections)
- File I/O operations
- GPU availability scenarios
- Error handling paths

## Setup Instructions

### Option 1: Using Virtual Environment (Recommended)
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows

# Install dependencies including dev dependencies
pip install -r requirements.txt
pip install pytest pytest-asyncio black flake8 mypy

# Run tests
pytest tests/ -v
```

### Option 2: Using Project Dependencies
If pytest is already installed in your project environment:
```bash
pytest tests/ -v
```

### Option 3: Install with --break-system-packages (Not Recommended)
```bash
python3 -m pip install pytest --break-system-packages
python3 -m pytest tests/ -v
```

## Running Tests

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test File
```bash
pytest tests/test_models.py -v
```

### Run Specific Test Class
```bash
pytest tests/test_models.py::TestHybridRecommenderModel -v
```

### Run Specific Test Method
```bash
pytest tests/test_models.py::TestHybridRecommenderModel::test_forward_pass -v
```

### Run Tests with Coverage
```bash
pytest tests/ --cov=. --cov-report=html
```

## Test Features

### Mocking Strategy
- Database operations are mocked to avoid requiring a real database
- File I/O operations use temporary files
- GPU operations are mocked for consistent testing across environments
- External API calls are mocked

### Fixtures
- Sample data fixtures for consistent test data
- Temporary file/directory management
- Mock configuration objects
- Pre-trained model states for testing

### Error Testing
- Invalid input validation
- Missing file handling
- Database connection failures
- Model loading errors
- Out-of-bounds tensor operations

## Test Data

### Model Testing
- Small models (3 users, 3 movies) for fast testing
- Known input/output pairs for validation
- Edge cases (empty datasets, single samples)

### Database Testing
- Mock database responses
- Error condition simulation
- Schema validation testing

### Configuration Testing
- Environment variable parsing
- Default value validation
- Type conversion testing
- Invalid input handling

## Integration with CI/CD

These tests are designed to run in continuous integration environments:
- No external dependencies (mocked database, files)
- Deterministic results (fixed random seeds where needed)
- Fast execution (< 1 minute for full suite)
- Clear pass/fail indicators

## Test Maintenance

### Adding New Tests
1. Follow existing naming conventions (`test_*.py`)
2. Use appropriate fixtures for setup/teardown
3. Mock external dependencies
4. Include both positive and negative test cases
5. Add docstrings explaining test purpose

### Updating Tests
- Update tests when changing model architecture
- Maintain backward compatibility where possible
- Update mock data when changing data formats
- Keep test documentation current

## Expected Test Results

When all tests pass, you should see output similar to:
```
tests/test_config.py::TestConfig::test_load_config_with_defaults PASSED
tests/test_models.py::TestHybridRecommenderModel::test_model_initialization PASSED
tests/test_database.py::TestDatabaseManager::test_init PASSED
tests/test_inference.py::TestMovieRecommendationSystem::test_init_success PASSED
tests/test_main.py::TestMainUtilities::test_setup_logging_debug PASSED
...
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
2. **File Not Found**: Tests use temporary files - ensure write permissions
3. **GPU Tests Failing**: GPU tests are mocked - check torch installation
4. **Database Tests Failing**: All database operations are mocked

### Performance
- Tests should complete in under 60 seconds
- If tests are slow, check for missing mocks
- Large file operations should use temporary small files

## Coverage Goals

Target test coverage:
- Models: 95%+ (critical path for recommendations)
- Database: 90%+ (important for data integrity)
- Configuration: 90%+ (prevents runtime errors)
- Main/Training: 85%+ (complex integration scenarios)
- Inference: 95%+ (critical path for end users)

Run coverage analysis:
```bash
pytest tests/ --cov=. --cov-report=term-missing
```