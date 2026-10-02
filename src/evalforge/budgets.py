"""Thread-safe, cooperative experiment budgets shared by targets, judges, and retries."""

import threading
import time
from contextvars import ContextVar

from evalforge.models import ExecutionLimits


class BudgetExceeded(RuntimeError):
    pass


CURRENT_BUDGET: ContextVar["RunBudget | None"] = ContextVar("evalforge_budget", default=None)


class RunBudget:
    def __init__(self, limits: ExecutionLimits):
        self.limits = limits.model_copy(deep=True)
        self.started = time.monotonic()
        self.requests = 0
        self.retries = 0
        self.observed_cost = 0.0
        self.unknown_cost = False
        self.lock = threading.Lock()

    def remaining_seconds(self):
        if self.limits.max_seconds is None:
            return None
        return max(0.0, self.limits.max_seconds - (time.monotonic() - self.started))

    def check_deadline(self):
        remaining = self.remaining_seconds()
        if remaining is not None and remaining <= 0:
            raise BudgetExceeded("experiment time budget exhausted")

    def check_start(self):
        self.check_deadline()
        with self.lock:
            if self.limits.max_observed_cost is not None:
                if self.unknown_cost:
                    raise BudgetExceeded("cost budget cannot continue with unknown provider usage")
                if self.observed_cost >= self.limits.max_observed_cost:
                    raise BudgetExceeded("observed provider cost budget exhausted")

    def check_completion(self):
        self.check_deadline()
        with self.lock:
            cap = self.limits.max_observed_cost
            if cap is not None:
                if self.unknown_cost:
                    raise BudgetExceeded("cost budget cannot validate unknown provider usage")
                if self.observed_cost > cap:
                    raise BudgetExceeded("observed provider cost budget exceeded")

    def reserve_request(self, *, retry=False):
        self.check_start()
        with self.lock:
            if (
                self.limits.max_provider_requests is not None
                and self.requests >= self.limits.max_provider_requests
            ):
                raise BudgetExceeded("provider request budget exhausted")
            self.requests += 1
            self.retries += int(retry)

    def observe_cost(self, cost):
        with self.lock:
            if cost is None:
                self.unknown_cost = True
            else:
                self.observed_cost += cost

    def summary(self):
        with self.lock:
            return {
                "limits": self.limits.model_dump(mode="json"),
                "provider_requests": self.requests,
                "provider_retries": self.retries,
                "observed_provider_cost": self.observed_cost,
                "unknown_provider_cost": self.unknown_cost,
                "elapsed_seconds": time.monotonic() - self.started,
            }
