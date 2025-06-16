# 🔧 Lupe Discord Bot Troubleshooting Guide

## 🚨 Quick Fixes

**If you see these errors, here are the immediate solutions:**

```
NameError: name 'Tuple' is not defined
```
→ **Fixed:** Updated imports in main.py

```
ERROR: column "content_type" does not exist
```
→ **Run:** `python migrate_database.py`

```
ModuleNotFoundError: No module named 'utils.database'
```
→ **Run:** `pip install -r requirements.txt`

```
PyNaCl is not installed, voice will NOT be supported
```
→ **Run:** `pip install PyNaCl` (optional warning)

## Common Issues and Solutions

### 1. Import Errors

#### Issue: `ModuleNotFoundError: No module named 'utils.database'`
**Solution:**
```bash
# Make sure you're in the lupe(python) directory
cd lupe(python)

# Install dependencies
pip install -r requirements.txt

# If still having issues, use the setup script
python setup.py
```

#### Issue: `ModuleNotFoundError: No module named 'discord'`
**Solution:**
```bash
pip install discord.py>=2.3.0
```

### 2. Database Issues

#### Issue: `ERROR: column "content_type" does not exist`
**Solution:**
This happens when upgrading from an older version. Run the migration script:
```bash
python migrate_database.py
```

#### Issue: `psycopg2.OperationalError: could not connect to server`
**Solutions:**
1. **Check if PostgreSQL is running:**
   ```bash
   # Windows
   net start postgresql
   
   # Linux/macOS
   sudo systemctl start postgresql
   # or
   brew services start postgresql
   ```

2. **Verify database exists:**
   ```sql
   -- Connect to PostgreSQL and create database
   CREATE DATABASE cinesync;
   ```

3. **Check connection settings in .env:**
   ```env
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=cinesync
   DB_USER=postgres
   DB_PASSWORD=Babycakes15
   ```

### 3. Discord Bot Issues

#### Issue: `discord.errors.LoginFailure: Improper token has been passed`
**Solution:**
1. Get your bot token from [Discord Developer Portal](https://discord.com/developers/applications)
2. Add it to your `.env` file:
   ```env
   DISCORD_TOKEN=your_actual_bot_token_here
   ```

#### Issue: Bot doesn't respond to commands
**Solutions:**
1. **Check bot permissions:**
   - Send Messages
   - Embed Links
   - Use Slash Commands
   - Read Message History

2. **Verify bot is in your server:**
   - Use OAuth2 URL generator in Discord Developer Portal
   - Select "bot" and "applications.commands" scopes

3. **Check if commands are synced:**
   - Bot automatically syncs slash commands on startup
   - Look for "Synced X command(s)" in logs

### 4. Model Loading Issues

#### Issue: `FileNotFoundError: No such file or directory: '../models/best_model.pt'`
**Solution:**
This is expected if you haven't trained models yet. The bot will work with fallback recommendations.

#### Issue: `Lupe AI not initialized`
**Solutions:**
1. **Check models directory structure:**
   ```
   models/
   ├── best_model.pt (optional)
   ├── movie_lookup.pkl (required for movies)
   ├── tv_lookup.pkl (optional for TV shows)
   └── ...
   ```

2. **Create minimal movie lookup for testing:**
   ```python
   import pickle
   dummy_lookup = {
       1: {'title': 'The Matrix', 'genres': 'Action|Sci-Fi'},
       2: {'title': 'Titanic', 'genres': 'Drama|Romance'}
   }
   with open('../models/movie_lookup.pkl', 'wb') as f:
       pickle.dump(dummy_lookup, f)
   ```

### 5. Command Syntax Issues

#### Issue: Commands use old `!` prefix instead of `/`
**Solution:**
Lupe uses slash commands. Use `/recommend` not `!recommend`.

**Correct syntax:**
```
/recommend mixed 5
/recommend movie 3
/recommend tv 8
/cross_recommend movie tv
/similar "Breaking Bad"
/rate "The Office" 5
```

### 6. Environment Setup

#### Issue: Virtual environment activation
**Windows:**
```cmd
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

#### Issue: Python version compatibility
**Requirements:**
- Python 3.9 or higher
- pip (latest version)

### 7. Performance Issues

#### Issue: Slow recommendation responses
**Solutions:**
1. **Reduce recommendation count:**
   ```
   /recommend mixed 5  # instead of 10+
   ```

2. **Check database performance:**
   - Ensure indexes are created (automatic on first run)
   - Monitor PostgreSQL logs

3. **GPU acceleration (if available):**
   ```env
   DEVICE=cuda  # in .env file
   ```

### 8. Development and Testing

#### Issue: Testing bot functionality
**Use the test script:**
```bash
python test_tv_support.py
```

#### Issue: Debug logging
**Enable debug mode:**
```env
DEBUG=true
```

### 9. Quick Health Check

Run these commands to verify everything works:

```bash
# 1. Check Python version
python --version

# 2. Check dependencies
pip list | grep -E "(discord|torch|pandas|psycopg2)"

# 3. Test database connection
python -c "import psycopg2; print('DB OK')"

# 4. Check bot token (without revealing it)
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('Token length:', len(os.getenv('DISCORD_TOKEN', '')))"

# 5. Test bot startup (Ctrl+C to stop)
python main.py
```

### 10. Getting Help

If you're still having issues:

1. **Check logs:** Look for error messages in the console output
2. **Verify environment:** Use `/lupe_status` command in Discord
3. **Test with minimal setup:** Use dummy data for initial testing
4. **Check Discord API status:** Visit [Discord Status](https://discordstatus.com/)

## Quick Setup Checklist

- [ ] Python 3.9+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] PostgreSQL running and accessible
- [ ] `.env` file created with Discord token
- [ ] Bot invited to Discord server with proper permissions
- [ ] Bot started successfully (`python main.py`)
- [ ] Slash commands appear in Discord server

## Support

For additional help:
- Check the main README.md for setup instructions
- Review the test_tv_support.py script for working examples
- Ensure all file paths and permissions are correct