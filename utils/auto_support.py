import logging

logger = logging.getLogger(__name__)


class AutoSupportController:
    """
    Adjust min_support based on observed rule count per turn.
    """

    def __init__(
        self,
        base_support,
        target_min,
        target_max,
        step,
        min_support,
        max_support,
    ):
        self.current_support = float(base_support)
        self.target_min = int(target_min)
        self.target_max = int(target_max)
        self.step = float(step)
        self.min_support = float(min_support)
        self.max_support = float(max_support)

    def get_support(self):
        return self.current_support

    def update(self, rule_count, turn_counter=None):
        if rule_count is None:
            return self.current_support

        action = "hold"
        new_support = self.current_support
        if rule_count > self.target_max:
            new_support = min(self.current_support * self.step, self.max_support)
            action = "increase"
        elif rule_count < self.target_min:
            new_support = max(self.current_support / self.step, self.min_support)
            action = "decrease"

        if new_support != self.current_support:
            logger.info(
                f"[AutoSupport] Turn {turn_counter}: {action} min_support "
                f"{self.current_support:.6f} -> {new_support:.6f} "
                f"(rules={rule_count}, target={self.target_min}-{self.target_max})"
            )
            self.current_support = new_support
        else:
            logger.info(
                f"[AutoSupport] Turn {turn_counter}: hold min_support={self.current_support:.6f} "
                f"(rules={rule_count}, target={self.target_min}-{self.target_max})"
            )
        return self.current_support
