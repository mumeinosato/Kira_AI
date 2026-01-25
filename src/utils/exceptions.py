class SafetyViolationError(Exception):
    """
    Raised when generated text violates safety filters (NSFW or Meta/OOC).
    """
    def __init__(self, reason: str, partial_text: str = ""):
        self.reason = reason
        self.partial_text = partial_text
        super().__init__(f"Safety violation: {reason}")
