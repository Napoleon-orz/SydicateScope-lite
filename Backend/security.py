"""
SyndicateScope Security Layer: RBAC, Centralized Authorization, and Audit Logging.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import Depends, Header, HTTPException, status, Request
import uuid

# ============================================================
# 1. ROLES & GRANULAR PERMISSIONS CONFIGURATION
# ============================================================

ROLES_PERMISSIONS = {
    "Admin": {
        "users.read",
        "users.manage",
        "roles.manage",
        "datasets.read",
        "datasets.write",
        "graphs.read",
        "graphs.write",
        "prediction.run",
        "audit.read"
    },
    "Analyst": {
        "datasets.read",
        "graphs.read",
        "prediction.run"
    },
    "Viewer": {
        "datasets.read",
        "graphs.read"
    }
}

# Mock user directory for demo testing (maps user_id -> role & profile)
MOCK_USERS = {
    "u_admin": {"user_id": "u_admin", "name": "Commander Apex", "role": "Admin"},
    "u_analyst": {"user_id": "u_analyst", "name": "Agent Vance", "role": "Analyst"},
    "u_viewer": {"user_id": "u_viewer", "name": "Observer Sam", "role": "Viewer"}
}

# ============================================================
# 2. AUDIT LOGGING SYSTEM (Append-Oriented & Searchable)
# ============================================================

AUDIT_LOG_FILE = Path(__file__).parent / "audit_logs.jsonl"


def record_audit_event(
        user_id: str,
        user_role: str,
        action: str,
        resource_type: str,
        resource_id: str,
        result: str,
        request_id: str
):
    """Appends a structured audit event to the JSONLines log file."""
    event = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": user_id,
        "user_role": user_role,
        "action": action,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "result": result,
        "request_id": request_id
    }
    with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def get_audit_logs(
        user_id: Optional[str] = None,
        role: Optional[str] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        result: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Reads and filters audit logs for administrative review."""
    if not AUDIT_LOG_FILE.exists():
        return []

    logs = []
    with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                logs.append(json.loads(line))

    # Apply filters if provided
    filtered = []
    for log in logs:
        if user_id and log["user_id"] != user_id:
            continue
        if role and log["user_role"].lower() != role.lower():
            continue
        if action and action.lower() not in log["action"].lower():
            continue
        if resource_type and log["resource_type"].lower() != resource_type.lower():
            continue
        if result and log["result"].lower() != result.lower():
            continue
        filtered.append(log)
    return filtered


# ============================================================
# 3. RESOURCE-LEVEL AUTHORIZATION LOGIC
# ============================================================

def verify_resource_access(user_role: str, resource_type: str, resource_id: str) -> bool:
    """
    Checks if a user role is permitted to access a specific resource instance.
    For example, certain sensitive investigation nodes or restricted logs can be guarded here.
    """
    if not resource_id:
        return True

    # Admins can access everything
    if user_role == "Admin":
        return True

    # Example resource-level restriction:
    # Viewers or Analysts might be restricted from viewing specific top-secret nodes if prefixed with SEC_
    if resource_id.upper().startswith("SEC_") and user_role not in ["Admin"]:
        return False

    return True


# ============================================================
# 4. CENTRAL AUTHORIZATION MIDDLEWARE & DEPENDENCY
# ============================================================

def require_permission(permission: str, resource_type: str = "general"):
    """
    Central authorization dependency factory.
    Performs: Identity check -> Role resolution -> Permission validation -> Resource check -> Audit Logging.
    """

    def permission_dependency(
            request: Request,
            x_user_id: Optional[str] = Header(default="u_viewer", description="Simulated User ID header"),
            x_resource_id: Optional[str] = Header(default="global", description="Target Resource ID header")
    ):
        request_id = str(uuid.uuid4())[:8]
        user_info = MOCK_USERS.get(x_user_id)

        # Fallback to default viewer if header unknown
        if not user_info:
            user_info = {"user_id": x_user_id or "anonymous", "role": "Viewer"}

        role = user_info["role"]
        user_permissions = ROLES_PERMISSIONS.get(role, set())

        # 1. Check Global Permission
        has_perm = permission in user_permissions

        # 2. Check Resource-Level Access
        resource_allowed = verify_resource_access(role, resource_type, x_resource_id) if has_perm else False

        final_allowed = has_perm and resource_allowed
        result_str = "allowed" if final_allowed else "denied"

        # Record audit event for all protected operations
        record_audit_event(
            user_id=user_info["user_id"],
            user_role=role,
            action=permission,
            resource_type=resource_type,
            resource_id=x_resource_id,
            result=result_str,
            request_id=request_id
        )

        if not final_allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied for role '{role}' on action '{permission}' (Resource: {x_resource_id})."
            )

        return user_info

    return permission_dependency