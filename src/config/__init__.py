"""配置模块：进程级配置项的唯一来源，其余模块一律通过 get_settings() 获取配置，
不直接读取环境变量，避免配置读取逻辑散落在各处。
"""
from src.config.settings import SystemConfiguration, get_settings

__all__ = ["SystemConfiguration", "get_settings"]
