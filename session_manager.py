"""
AI Studio Account Pool & Session Manager
----------------------------------------
Manages multi-account authentication, load balancing, health checks, and cooldowns.
Zero external dependencies - 100% Python standard library compatible.
"""

import time
import json
import threading
from typing import Dict, Any, List, Optional, Tuple

class AccountSession:
    """Represents a single AI Studio credential identity."""
    def __init__(
        self,
        identifier: str,
        auth_type: str = "api_key",  # "api_key", "relay_token", or "cookie"
        credential: str = "",
        endpoint: str = "https://generativelanguage.googleapis.com",
        tier: str = "Pro",
        models: Optional[List[str]] = None
    ):
        self.identifier = identifier
        self.auth_type = auth_type
        self.credential = credential
        self.endpoint = endpoint
        self.tier = tier
        self.models = models or []
        self.is_active = True
        self.cooldown_until = 0.0
        self.fail_count = 0
        self.success_count = 0
        self.last_error = ""

    def is_available(self) -> bool:
        if not self.is_active:
            return False
        if time.time() < self.cooldown_until:
            return False
        return True

    def set_cooldown(self, seconds: float = 60.0, reason: str = "Rate limited"):
        self.cooldown_until = time.time() + seconds
        self.last_error = reason

    def mark_success(self):
        self.fail_count = 0
        self.success_count += 1
        self.last_error = ""

    def mark_failure(self, error_msg: str, status_code: int = 500):
        self.fail_count += 1
        self.last_error = error_msg
        if status_code in (401, 403):
            self.is_active = False  # Requires re-authentication
        elif status_code == 429:
            self.set_cooldown(60.0, "Quota / Rate Limit exceeded")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier,
            "auth_type": self.auth_type,
            "tier": self.tier,
            "is_active": self.is_active,
            "cooldown_remaining": max(0, int(self.cooldown_until - time.time())),
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "last_error": self.last_error
        }


class AIStudioSessionPool:
    """Thread-safe round-robin pool of AI Studio accounts."""
    def __init__(self):
        self._accounts: List[AccountSession] = []
        self._lock = threading.Lock()
        self._rr_index = 0

    def add_account(self, account: AccountSession):
        with self._lock:
            # Avoid duplicate identifiers
            self._accounts = [a for a in self._accounts if a.identifier != account.identifier]
            self._accounts.append(account)

    def get_next_account(self, requested_model: Optional[str] = None) -> Optional[AccountSession]:
        """Round-robin selection of next healthy, non-cooldown account."""
        with self._lock:
            if not self._accounts:
                return None

            total = len(self._accounts)
            for _ in range(total):
                acc = self._accounts[self._rr_index % total]
                self._rr_index = (self._rr_index + 1) % total

                if acc.is_available():
                    # If specific models are configured on account, ensure compatibility
                    if requested_model and acc.models:
                        if requested_model in acc.models or any(m in requested_model for m in acc.models):
                            return acc
                    else:
                        return acc

            return None

    def get_status_summary(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._accounts)
            available = sum(1 for a in self._accounts if a.is_available())
            cooldown = sum(1 for a in self._accounts if time.time() < a.cooldown_until)
            inactive = sum(1 for a in self._accounts if not a.is_active)

            return {
                "total": total,
                "ready": available,
                "cooldown": cooldown,
                "auth_required": inactive,
                "accounts": [a.to_dict() for a in self._accounts]
            }

    def clear_all_cooldowns(self):
        with self._lock:
            for a in self._accounts:
                a.cooldown_until = 0.0
