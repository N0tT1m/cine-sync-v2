# Security Setup Guide

This guide helps you properly configure CineSync v2 for public repositories and secure deployment.

## 🔒 Environment Variables Setup

### Required Environment Variables

Create a `.env` file in your project root and `lupe(python)` directory with the following variables:

```bash
# Database Configuration
DB_HOST=localhost
DB_NAME=cinesync
DB_USER=postgres
DB_PASSWORD=your_secure_database_password_here
DB_PORT=5432

# Discord Bot Configuration
DISCORD_TOKEN=your_discord_bot_token_here
DISCORD_PREFIX=!

# Model Configuration
MODEL_EMBEDDING_DIM=64
MODEL_HIDDEN_DIM=128
MODEL_EPOCHS=10
MODEL_BATCH_SIZE=1024
MODEL_LEARNING_RATE=0.001
MODELS_DIR=models

# Server Configuration
SERVER_HOST=localhost
SERVER_PORT=3000
API_BASE_URL=http://localhost:3000

# Development Configuration
DEBUG=false
DEVICE=auto
BATCH_SIZE=64
LEARNING_RATE=0.001
EPOCHS=20
```

### Setup Steps

1. **Copy the example file**:
   ```bash
   cp .env.example .env
   ```

2. **Generate secure passwords**:
   - Use a strong, unique password for your database
   - Create a Discord bot token from the Discord Developer Portal
   - Never commit these values to version control

3. **Configure your environment**:
   ```bash
   # Edit the .env file with your actual secure values
   nano .env
   ```

## 🛡️ Security Best Practices

### Database Security

- **Strong Passwords**: Use complex passwords with at least 16 characters
- **Network Security**: Restrict database access to localhost or specific IPs
- **Regular Updates**: Keep PostgreSQL updated to the latest version
- **Backup Encryption**: Encrypt database backups if storing remotely

### Discord Bot Security

- **Token Protection**: Never share or commit your Discord bot token
- **Permission Minimization**: Only grant necessary permissions to the bot
- **Regular Rotation**: Rotate bot tokens periodically
- **Monitoring**: Monitor bot usage for unusual activity

### Application Security

- **Input Validation**: All user inputs are validated and sanitized
- **SQL Injection Prevention**: Using parameterized queries throughout
- **Rate Limiting**: Implement rate limiting for API endpoints
- **Error Handling**: Secure error messages that don't leak sensitive information

## 🔐 File Permissions

Ensure proper file permissions for sensitive files:

```bash
# Secure environment files
chmod 600 .env
chmod 600 lupe(python)/.env

# Secure configuration files
chmod 644 config.py
chmod 644 lupe(python)/config.py
```

## 🚫 What NOT to Include in Version Control

The `.gitignore` file excludes these sensitive files:

- `.env` files (all variants)
- `*.key` and `*.pem` files
- `config.secret.py` files
- Database backups (`*.sql`, `*.db`)
- Log files containing sensitive data
- Personal API keys or tokens

## 🔍 Security Validation

### Check for Hardcoded Secrets

Before committing, run these commands to check for accidentally committed secrets:

```bash
# Search for potential API keys
grep -r "api_key\|API_KEY\|secret\|SECRET" --include="*.py" .

# Search for tokens
grep -r "token\|TOKEN" --include="*.py" .

# Search for database credentials
grep -r "password\|PASSWORD" --include="*.py" .
```

### Pre-commit Hooks

Consider using pre-commit hooks to automatically scan for secrets:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install
```

## 🚀 Deployment Security

### Production Environment

For production deployment:

1. **Use environment variables** instead of `.env` files
2. **Secure the server** with proper firewall rules
3. **Use HTTPS** for all external communications
4. **Monitor logs** for security incidents
5. **Regular updates** for all dependencies

### Docker Security

When using Docker:

```bash
# Use secrets for sensitive data
docker run -e DB_PASSWORD_FILE=/run/secrets/db_password

# Don't include .env in Docker images
# Ensure .dockerignore includes .env
```

## 📋 Security Checklist

Before making your repository public:

- [ ] All hardcoded passwords removed from code
- [ ] `.env.example` file created with placeholder values
- [ ] `.gitignore` properly configured to exclude sensitive files
- [ ] Configuration files use environment variables
- [ ] Documentation updated with security setup instructions
- [ ] No API keys or tokens in commit history
- [ ] Database passwords are strong and unique
- [ ] Discord bot permissions are minimal and appropriate

## 🆘 If You Accidentally Commit Secrets

If you accidentally commit sensitive information:

1. **Immediately rotate** all compromised credentials
2. **Remove from git history** using `git filter-branch` or BFG Repo-Cleaner
3. **Force push** the cleaned history
4. **Notify collaborators** to re-clone the repository
5. **Monitor** for any unauthorized access

## 📞 Security Contact

For security-related issues or questions:
- Open a GitHub issue with the "security" label
- Follow responsible disclosure practices
- Do not publicly disclose security vulnerabilities until they are fixed

---

**Remember**: Security is an ongoing process, not a one-time setup. Regularly review and update your security practices.