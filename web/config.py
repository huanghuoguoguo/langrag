"""Web应用配置"""
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据存储根目录
DATA_DIR = PROJECT_ROOT / "web" / "data"

# 各个组件的数据目录
DB_DIR = DATA_DIR / "db"
CHROMA_DIR = DATA_DIR / "chroma"
DUCKDB_DIR = DATA_DIR / "duckdb"
SEEKDB_DIR = DATA_DIR / "seekdb"

# 确保目录存在
DATA_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)
DUCKDB_DIR.mkdir(exist_ok=True)
SEEKDB_DIR.mkdir(exist_ok=True)

# SQLite 数据库路径
DATABASE_URL = f"sqlite:///{DB_DIR / 'app.db'}"
