"""
AetherMind DCLA - Universal Body Component
Path: body/adapter_base.py
"""

class BodyAdapter:
    def execute(self, intent):
        raise NotImplementedError('Subclasses must implement execution logic')