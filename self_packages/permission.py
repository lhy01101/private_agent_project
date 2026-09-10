# 用户权限管理：user_id -> permission 映射表 + 构造 runtime context 的统一入口。
# 未来接真实权限服务时，只需把 USER_PERMISSIONS 换成数据库查询即可，调用方无需改动。

# --- 权限等级（由低到高） ---
BASIC = "basic"    # 免费用户：延迟行情
PRO = "pro"        # 付费用户：实时行情
QUANT = "quant"    # 量化用户：预留量化接口（未来）

# 等级数值，用于"是否达到某权限"的 >= 比较
LEVELS = {BASIC: 0, PRO: 1, QUANT: 2}

# --- 用户 -> 权限映射表（真实场景请改为数据库查询） ---
USER_PERMISSIONS = {
    "1": PRO,      # 付费用户
    "2": BASIC,    # 免费用户
    "3": QUANT,    # 量化用户
}


def build_context(user_id: str) -> dict:
    """根据 user_id 查权限，构造 runtime context。

    调用方只传 user_id，不传权限——权限由本表决定，
    避免客户端自报权限（否则人人都能说自己 pro）。
    """
    return {"user_id": user_id, "permission": USER_PERMISSIONS.get(user_id, BASIC)}


def has_permission(permission: str, required: str = PRO) -> bool:
    """工具侧校验：当前权限是否达到 required 等级（>=）。

    例：has_permission("quant", PRO) -> True（量化权限 >= 付费权限）
        has_permission("basic", PRO) -> False
    """
    return LEVELS.get(permission, 0) >= LEVELS.get(required, 0)
