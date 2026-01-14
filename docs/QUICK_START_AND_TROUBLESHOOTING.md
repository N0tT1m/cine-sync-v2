# CineSync v2 - Quick Start & Troubleshooting Guide

---

## Quick Start Guide

### Step 1: Clone and Setup Environment

```bash
# Clone repository
git clone <your-repo-url>
cd cine-sync-v2

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For advanced models (optional - larger install)
pip install -r src/models/advanced/requirements.txt
```

### Step 2: Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your values
nano .env  # or use your preferred editor
```

**Required Variables:**
```bash
# Discord Bot
DISCORD_TOKEN=your_discord_bot_token_here

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=cinesync
DB_USER=postgres
DB_PASSWORD=your_db_password

# Admin Interface (generate these!)
ADMIN_SECRET_KEY=<run: python -c "import secrets; print(secrets.token_hex(32))">
ADMIN_USERS=admin
ADMIN_PASSWORD_HASH=<run: python -c "from werkzeug.security import generate_password_hash; print(generate_password_hash('your_password'))">
```

### Step 3: Setup Database

```bash
# Create database
createdb cinesync

# Initialize schema
psql -U postgres -d cinesync -f configs/deployment/init-db.sql
```

Or with Docker:
```bash
docker-compose -f configs/deployment/docker-compose.yml up -d postgres
```

### Step 4: Verify Model Installation

```bash
# Quick verification of all 45 models
python tests/test_model_integration.py

# Or run full pytest
pytest tests/test_model_integration.py -v
```

### Step 5: Run the Bot

```bash
# Navigate to bot directory
cd services/lupe_python

# Run the bot
python main.py
```

### Step 6: Run Admin Interface (Optional)

```bash
# In a separate terminal
cd src/api
python admin_interface.py

# Access at http://localhost:5001
```

---

## Environment Variable Reference

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `DISCORD_TOKEN` | Yes | Discord bot token | `MTIzNDU2Nzg5...` |
| `DB_HOST` | Yes | Database host | `localhost` |
| `DB_PORT` | Yes | Database port | `5432` |
| `DB_NAME` | Yes | Database name | `cinesync` |
| `DB_USER` | Yes | Database user | `postgres` |
| `DB_PASSWORD` | Yes | Database password | `your_password` |
| `ADMIN_SECRET_KEY` | Yes* | Flask secret key | `a1b2c3d4...` (64 chars) |
| `ADMIN_USERS` | Yes* | Comma-separated admin usernames | `admin,moderator` |
| `ADMIN_PASSWORD_HASH` | Yes* | Hashed admin password | `pbkdf2:sha256:...` |
| `DEBUG` | No | Enable debug mode | `false` |
| `ENABLE_WANDB` | No | Enable WandB logging | `false` |
| `WANDB_PROJECT` | No | WandB project name | `cinesync-models` |

*Required for admin interface

---

## Generating Secure Credentials

### Generate Secret Key
```bash
python -c "import secrets; print(secrets.token_hex(32))"
# Output: a1b2c3d4e5f6...
```

### Generate Password Hash
```bash
python -c "from werkzeug.security import generate_password_hash; print(generate_password_hash('your_secure_password'))"
# Output: pbkdf2:sha256:260000$...
```

### Verify Password Hash
```bash
python -c "
from werkzeug.security import check_password_hash
hash = 'pbkdf2:sha256:260000\$...'  # Your hash
print(check_password_hash(hash, 'your_password'))
"
# Output: True or False
```

---

## Troubleshooting Guide

### Issue: "No module named 'src'"

**Symptom:**
```
ModuleNotFoundError: No module named 'src'
```

**Solution:**
Add project root to Python path. This is now automatically done in `train_all_models.py`, but for other scripts:

```python
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
```

Or run from project root:
```bash
cd /path/to/cine-sync-v2
PYTHONPATH=. python src/training/train_all_models.py --all
```

---

### Issue: "UnifiedTrainingPipeline got unexpected keyword argument 'epochs'"

**Symptom:**
```
TypeError: UnifiedTrainingPipeline.__init__() got an unexpected keyword argument 'epochs'
```

**Solution:**
This has been fixed. If you see this error, pull the latest code:
```bash
git pull origin main
```

The fix filters training kwargs from init kwargs in `train_category()`.

---

### Issue: Discord Button Callback Crash

**Symptom:**
```
NameError: name 'button' is not defined
```

**Solution:**
This has been fixed. The callback now uses `interaction.data['custom_id']` instead of `button.custom_id`.

---

### Issue: "ADMIN_PASSWORD_HASH not configured"

**Symptom:**
```
Admin authentication not properly configured
```

**Solution:**
Set the environment variables:
```bash
export ADMIN_USERS=admin
export ADMIN_PASSWORD_HASH=$(python -c "from werkzeug.security import generate_password_hash; print(generate_password_hash('your_password'))")
```

---

### Issue: "No module named 'torch'"

**Symptom:**
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**
Install PyTorch:
```bash
# CPU only
pip install torch torchvision

# With CUDA (check your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

### Issue: Database Connection Failed

**Symptom:**
```
psycopg2.OperationalError: could not connect to server
```

**Solutions:**

1. **Check PostgreSQL is running:**
```bash
# macOS
brew services list | grep postgresql

# Linux
sudo systemctl status postgresql
```

2. **Check credentials:**
```bash
psql -U postgres -h localhost -d cinesync
```

3. **Check .env file:**
```bash
cat .env | grep DB_
```

---

### Issue: Discord Bot Won't Connect

**Symptom:**
```
discord.errors.LoginFailure: Improper token has been passed
```

**Solutions:**

1. **Check token:**
```bash
echo $DISCORD_TOKEN | head -c 20
# Should show first 20 chars of your token
```

2. **Regenerate token:**
   - Go to Discord Developer Portal
   - Select your application
   - Go to Bot tab
   - Reset Token

3. **Check .env is loaded:**
```python
from dotenv import load_dotenv
load_dotenv()
import os
print(os.getenv('DISCORD_TOKEN'))  # Should print your token
```

---

### Issue: "Rate limited" on Admin Login

**Symptom:**
```
Too many failed attempts. Please try again later.
```

**Solution:**
Wait 15 minutes, or restart the admin interface to clear the rate limit cache.

---

### Issue: Model Loading Fails

**Symptom:**
```
Failed to load model: [model_name]
```

**Solutions:**

1. **Check model file exists:**
```bash
ls -la models/[category]/[model_name]/
```

2. **Check model can be imported:**
```bash
python -c "from src.models.movie.franchise_sequence import FranchiseSequenceModel; print('OK')"
```

3. **Run model verification:**
```bash
python src/api/extended_model_loader.py
```

---

### Issue: Commands Not Syncing

**Symptom:**
Commands don't appear in Discord after bot starts.

**Solutions:**

1. **Wait for sync:**
Discord command sync can take up to 1 hour globally.

2. **Check sync in logs:**
```
Synced 24 command(s)
```

3. **Force guild sync (faster for testing):**
```python
# In main.py on_ready
await self.tree.sync(guild=discord.Object(id=YOUR_GUILD_ID))
```

---

### Issue: Personalization Commands Missing

**Symptom:**
`/my_recommendations`, `/my_stats`, `/rate_movies` not available.

**Solutions:**

1. **Check personalization modules exist:**
```bash
ls services/lupe_python/preference_learner.py
ls services/lupe_python/personalized_trainer.py
ls services/lupe_python/personalized_commands.py
```

2. **Check import errors in logs:**
```
Personalization modules not available: [error]
```

3. **Verify PERSONALIZATION_AVAILABLE:**
```python
# In main.py
print(f"PERSONALIZATION_AVAILABLE: {PERSONALIZATION_AVAILABLE}")
```

---

### Issue: Out of Memory (OOM)

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size:**
```bash
python src/training/train_all_models.py --all --batch-size 64
```

2. **Use CPU:**
```bash
python src/training/train_all_models.py --all --device cpu
```

3. **Enable gradient checkpointing** (in model code)

---

### Issue: Slow Recommendations

**Symptom:**
Recommendations take >5 seconds.

**Solutions:**

1. **Use GPU:**
```python
api = UnifiedRecommendationAPI(device='cuda')
```

2. **Reduce candidate pool:**
```python
api.get_recommendations(user_id=123, top_k=10)  # Don't request too many
```

3. **Use single model instead of ensemble:**
```python
api.get_recommendations(user_id=123, model_type=ModelType.NCF)
```

---

## Diagnostic Commands

### Check System Status
```bash
# Python version
python --version

# PyTorch version and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Database connection
python -c "
import psycopg2
conn = psycopg2.connect(host='localhost', database='cinesync', user='postgres', password='password')
print('Database OK')
conn.close()
"

# Discord.py version
python -c "import discord; print(f'discord.py: {discord.__version__}')"
```

### Check Model Registry
```bash
python -c "
from src.training.train_all_models import ALL_MODELS
print(f'Total models registered: {len(ALL_MODELS)}')
for name in sorted(ALL_MODELS.keys()):
    print(f'  - {name}')
"
```

### Verify All Model Imports
```bash
python tests/test_model_integration.py
```

### Check Database Tables
```bash
psql -U postgres -d cinesync -c "\dt"
```

### View Recent Logs
```bash
# Bot logs
tail -f services/lupe_python/logs/bot.log

# Admin interface logs
tail -f src/api/logs/admin.log
```

---

## Getting Help

1. **Check this documentation first**
2. **Search existing issues:** [GitHub Issues]
3. **Check logs for errors**
4. **Create a new issue with:**
   - Error message
   - Steps to reproduce
   - Environment info (Python version, OS, etc.)

---

## Quick Command Reference

```bash
# Start bot
cd services/lupe_python && python main.py

# Start admin interface
cd src/api && python admin_interface.py

# Train all models
python src/training/train_all_models.py --all

# Run tests
pytest tests/ -v

# Verify models
python src/api/extended_model_loader.py

# Check database
psql -U postgres -d cinesync

# Generate credentials
python -c "import secrets; print(secrets.token_hex(32))"
python -c "from werkzeug.security import generate_password_hash; print(generate_password_hash('password'))"
```

---

**Document Version:** 1.0.0
**Last Updated:** 2026-01-14
