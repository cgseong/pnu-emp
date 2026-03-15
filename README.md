# 정보컴퓨터공학부 취업 현황 분석 (Employment Status Analysis)

**Latest Update**: Supabase Database Integration (v3.0)
**Status**: ✅ Production Ready

---

## 📊 Overview

A Streamlit-based dashboard for analyzing employment statistics of Computer Science graduates. The application now uses **Supabase PostgreSQL database** for data management instead of CSV files.

**Key Features**:
- 📈 Interactive employment status visualizations
- 📅 Year-based trend analysis (2020-2024)
- 🏢 Company and industry insights
- 💾 Real-time data from Supabase
- 🔒 Secure credential management

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Supabase account (free tier available)
- Git (for version control)

### Installation

1. **Clone or download the project**
   ```bash
   cd emp
   ```

2. **Create environment file**
   ```bash
   cp .env.example .env
   ```

3. **Add Supabase credentials**
   Edit `.env` and fill in your Supabase details:
   ```env
   SUPABASE_URL=https://your-project-id.supabase.co
   SUPABASE_KEY=your-anon-key-here
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

The app will open at `http://localhost:8501`

---

## 📁 Project Structure

```
emp/
├── streamlit_app.py              # Main application
├── config.py                     # Supabase configuration
├── requirements.txt              # Python dependencies
├── .env.example                  # Credentials template (commit to GitHub)
├── .env                          # Local credentials (NEVER commit)
├── .gitignore                    # Git ignore rules
│
├── README.md                     # This file
├── MIGRATION_COMPLETE.md         # Migration details
├── IMPLEMENTATION_CHECKLIST.md   # Verification checklist
│
├── ERROR_RESOLUTION.md           # Troubleshooting guide
├── SECRETS_SECURITY.md           # Security best practices
├── GITHUB_SETUP_GUIDE.md         # Deployment guide
├── SECURITY_CHECKLIST.md         # Pre-deployment checks
│
└── __pycache__/                  # Python cache (auto-generated)
```

---

## 🔐 Security

### Credential Management

The application uses a **3-tier credential loading system**:

1. **Environment Variables** (Highest Priority)
   ```bash
   export SUPABASE_URL="https://..."
   export SUPABASE_KEY="eyJ..."
   ```

2. **Streamlit Secrets (Top-level)**
   - In `~/.streamlit/secrets.toml`:
   ```toml
   SUPABASE_URL = "https://..."
   SUPABASE_KEY = "eyJ..."
   ```

3. **Streamlit Nested Secrets** (Lowest Priority)
   - In `~/.streamlit/secrets.toml`:
   ```toml
   [supabase]
   url = "https://..."
   key = "eyJ..."
   ```

### Protecting Secrets

⚠️ **IMPORTANT**: Never commit `.env` file to GitHub

**Protection Strategy**:
- `.env` is excluded in `.gitignore` ✅
- `.env.example` provides a safe template ✅
- `secrets.toml` is also excluded ✅

---

## 🗄️ Database Schema

### Supabase Table: `graduation_employment`

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| year | INT | Graduation year (2020-2024) |
| student_id | TEXT | Student identifier |
| employment_status | TEXT | Job status (취업, 공고대기, etc.) |
| company_name | TEXT | Employer name |
| position | TEXT | Job title |
| industry | TEXT | Industry classification |

**Total Records**: 670 graduation employment entries

---

## 🧪 Verification

Run this to verify the setup:

```bash
# Check configuration
python -c "from config import supabase_config; print(f'Configured: {supabase_config.is_configured()}')"

# Check if data loads
streamlit run streamlit_app.py
```

Expected output:
```
INFO:config:[SUCCESS] Supabase 연결 성공
INFO:__main__:[SUCCESS] 670건의 데이터 로드 완료
```

---

## 📚 Documentation

For more detailed information, see:

| Document | Purpose |
|----------|---------|
| [MIGRATION_COMPLETE.md](./MIGRATION_COMPLETE.md) | Complete migration details |
| [IMPLEMENTATION_CHECKLIST.md](./IMPLEMENTATION_CHECKLIST.md) | Implementation verification |
| [ERROR_RESOLUTION.md](./ERROR_RESOLUTION.md) | Troubleshooting guide |
| [SECRETS_SECURITY.md](./SECRETS_SECURITY.md) | Security best practices |
| [GITHUB_SETUP_GUIDE.md](./GITHUB_SETUP_GUIDE.md) | GitHub & Streamlit Cloud setup |
| [SECURITY_CHECKLIST.md](./SECURITY_CHECKLIST.md) | Pre-deployment checklist |

---

## 🌐 Deployment

### Local Development
```bash
streamlit run streamlit_app.py
```

### Streamlit Cloud
1. Push code to GitHub (`.env` is protected by `.gitignore`)
2. Deploy via Streamlit Cloud
3. Add secrets in app settings:
   - `SUPABASE_URL`
   - `SUPABASE_KEY`

See [GITHUB_SETUP_GUIDE.md](./GITHUB_SETUP_GUIDE.md) for detailed instructions.

### Docker
```bash
docker build -t emp-dashboard .
docker run -p 8501:8501 --env-file .env emp-dashboard
```

---

## 🐛 Troubleshooting

### Problem: "Supabase가 설정되지 않았습니다" (Supabase not configured)

**Solution**:
1. Check `.env` file exists
2. Verify `SUPABASE_URL` and `SUPABASE_KEY` are set
3. Verify credentials are valid in Supabase dashboard

### Problem: "테이블에서 데이터를 찾을 수 없습니다" (Table data not found)

**Solution**:
1. Check `graduation_employment` table exists in Supabase
2. Verify table has data (670 records)
3. Check RLS (Row Level Security) policies aren't blocking access

### Problem: "ModuleNotFoundError: No module named 'supabase'"

**Solution**:
```bash
pip install -r requirements.txt
# or
pip install supabase>=2.23.0
```

For more troubleshooting, see [ERROR_RESOLUTION.md](./ERROR_RESOLUTION.md)

---

## 📊 Data Features

The dashboard provides:

- **Year-based Analysis**: Filter and analyze by academic year
- **Status Distribution**: Visualize employment outcomes
- **Company Insights**: Top employers and positions
- **Industry Trends**: Industry-wise employment distribution
- **Export Options**: Download processed data

---

## ✨ What Changed (CSV → Supabase)

### Removed ❌
- CSV file dependency (`data/graduation_employment.xlsx`)
- File encoding detection
- Local file validation
- File read/write operations

### Added ✅
- Supabase PostgreSQL integration
- Real-time data synchronization
- Cloud-based data management
- Secure credential system
- 3-tier configuration loading

### Preserved ✅
- All dashboard features
- Data filtering and analysis
- Visualization charts
- Export functionality
- User interface

---

## 🔧 Technical Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| Streamlit | >=1.28.0 | Web framework |
| Pandas | >=2.0.0 | Data processing |
| Plotly | >=5.15.0 | Visualization |
| Supabase | >=2.23.0 | Database |
| Python | 3.8+ | Runtime |

---

## 📝 License

This project is part of the Computer Science Department employment analysis system.

---

## 👨‍💻 Development

### Running Tests
```bash
python -m pytest
```

### Code Style
```bash
black streamlit_app.py config.py
flake8 streamlit_app.py config.py
```

### Creating Backups
```bash
# Manual backup
pg_dump -h db.supabase.co -U postgres --password graduation_employment > backup.sql

# Or use Supabase dashboard for automated backups
```

---

## 📞 Support

For issues or questions:
1. Check [ERROR_RESOLUTION.md](./ERROR_RESOLUTION.md)
2. Review [SECURITY_CHECKLIST.md](./SECURITY_CHECKLIST.md)
3. See [GITHUB_SETUP_GUIDE.md](./GITHUB_SETUP_GUIDE.md)

---

**Status**: Ready for Production ✅
**Last Updated**: 2025-11-05
**Version**: 3.0 (Supabase Integration)
