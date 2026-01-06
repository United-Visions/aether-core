"""
Path: brain/safety_inhibitor.py
Part of AetherMind DCLA Phase 1
Role: The Hard-Wired Ethical Brake. This script runs AFTER thought generation 
but BEFORE the output reaches the user.

CRITICAL UPDATE: Extended with Kinetic Safety Rules to prevent physical harm
through hardware interfaces (GPIO, Serial, RTSP).
"""

import re
import json
import logging

class SafetyInhibitor:
    def __init__(self):
        self.logger = logging.getLogger("SafetyInhibitor")
        
        # The Prime Directives: Hard-coded rules that cannot be overwritten by learning.
        self.blocked_categories = {
            "physical_harm": [
                r"how to kill", r"how to hurt", r"weapon preparation", 
                r"poison recipe", r"assassination", r"bomb making"
            ],
            "self_harm": [
                r"how to commit suicide", r"cutting myself", r"end my life",
                r"self-inflicted pain"
            ],
            "illegal_acts": [
                r"how to steal", r"hack into", r"identity theft", 
                r"distribute illegal drugs"
            ],
            "hate_speech": [
                # Note: These are patterns for extreme dehumanization
                r"all \[REDACTED\] should die", r"superior race", r"inferior people"
            ]
        }
        
        # KINETIC SAFETY RULES: Hardware-specific protections
        # These prevent the agent from causing physical damage through hardware interfaces
        
        # Critical GPIO pins that must NEVER be modified
        # Example: Emergency shutdown circuits, safety interlocks, power systems
        self.critical_gpio_pins = [
            21,  # Emergency shutdown pin
            20,  # Safety interlock
            # Add more critical pins specific to your hardware deployment
        ]
        
        # Dangerous serial command patterns (for industrial equipment, robotics)
        self.dangerous_serial_patterns = [
            r"EMERGENCY_OVERRIDE",
            r"DISABLE_SAFETY",
            r"FORCE_UNLOCK",
            r"BYPASS_LIMIT",
            r"MAX_POWER_UNLIMITED",
            r"DISABLE_SENSORS",
            r"FORMAT_DRIVE",
            r"ERASE_MEMORY",
            r"SHUTDOWN_SYSTEM",
            r"KILL_ALL",
        ]
        
        # Dangerous GPIO operations
        self.dangerous_gpio_patterns = [
            r"OVERHEAT_MOTORS",
            r"DISABLE_COOLING",
            r"MAXIMUM_VOLTAGE",
            r"UNSAFE_CURRENT",
        ]
        
        # The message the user sees if a thought is blocked
        self.inhibition_response = (
            "SAFETY INHIBITION TRIGGERED: The requested action or thought "
            "violates my core ethical priors regarding the preservation of human life and safety."
        )
        
        # The message for kinetic safety violations
        self.kinetic_inhibition_response = (
            "KINETIC SAFETY INHIBITION: The requested hardware action has been blocked "
            "to prevent potential physical harm or equipment damage. This is a non-overridable "
            "safety measure to protect human life and critical infrastructure."
        )

    def scan(self, proposed_output: str) -> str:
        """
        Analyzes the Brain's proposed response. 
        Returns the original text if safe, or the inhibition_response if dangerous.
        """
        # Handle None or empty input
        if proposed_output is None:
            self.logger.error("[ALERT] Safety scan received None input")
            return "ERROR: Unable to process response - received empty output from reasoning engine."
        
        clean_text = proposed_output.lower().strip()

        for category, patterns in self.blocked_categories.items():
            for pattern in patterns:
                if re.search(pattern, clean_text):
                    # Log the violation (In a real system, this goes to a secure database)
                    self.logger.warning(f"[ALERT] Safety violation detected in category: {category}")
                    return self.inhibition_response

        # If no patterns match, the thought is released to the Body
        return proposed_output
    
    def check_kinetic_safety(self, intent_json: str) -> tuple[bool, str]:
        """
        CRITICAL SAFETY CHECK: Validates hardware intents before execution.
        
        This method prevents the agent from causing physical damage through hardware
        interfaces. It operates on structured JSON intents before they reach the
        hardware adapter layer.
        
        Args:
            intent_json: JSON string containing hardware intent
            
        Returns:
            Tuple of (is_safe: bool, message: str)
            - (True, "Safe") if the intent passes all safety checks
            - (False, reason) if the intent is blocked with explanation
        
        Safety Layers:
        1. GPIO Pin Protection: Prevents access to safety-critical pins
        2. Serial Command Filtering: Blocks dangerous industrial control commands
        3. Protocol Validation: Ensures only approved protocols are used
        4. Rate Limiting: Prevents rapid-fire hardware commands (future enhancement)
        
        Example Usage:
            inhibitor = SafetyInhibitor()
            safe, msg = inhibitor.check_kinetic_safety(hardware_intent)
            if not safe:
                return kinetic_inhibition_response
        """
        try:
            # Parse the hardware intent
            command = json.loads(intent_json)
            protocol = command.get("protocol", "").upper()
            action = command.get("action", "")
            params = command.get("params", {})
            
            # ==========================
            # GPIO Safety Checks
            # ==========================
            if protocol == "GPIO":
                pin = params.get("pin")
                
                # Critical pin protection
                if pin in self.critical_gpio_pins:
                    reason = (f"BLOCKED: GPIO pin {pin} is designated as safety-critical. "
                             f"This pin controls critical infrastructure and cannot be modified.")
                    self.logger.error(f"[KINETIC SAFETY] {reason}")
                    return False, reason
                
                # Validate pin number range (BCM mode: 0-27)
                if pin is not None and not (0 <= int(pin) <= 27):
                    reason = f"BLOCKED: Invalid GPIO pin number {pin}. Must be 0-27 (BCM mode)."
                    self.logger.error(f"[KINETIC SAFETY] {reason}")
                    return False, reason
                
                # Check for dangerous patterns in GPIO actions
                action_str = str(action).upper()
                for pattern in self.dangerous_gpio_patterns:
                    if re.search(pattern, action_str, re.IGNORECASE):
                        reason = (f"BLOCKED: GPIO action contains dangerous pattern: {pattern}")
                        self.logger.error(f"[KINETIC SAFETY] {reason}")
                        return False, reason
            
            # ==========================
            # Serial/UART Safety Checks
            # ==========================
            elif protocol == "SERIAL":
                payload = params.get("payload", "")
                
                # Check for dangerous command patterns
                for pattern in self.dangerous_serial_patterns:
                    if re.search(pattern, payload, re.IGNORECASE):
                        reason = (f"BLOCKED: Serial command contains dangerous pattern: {pattern}. "
                                 f"This could cause equipment damage or safety hazards.")
                        self.logger.error(f"[KINETIC SAFETY] {reason}")
                        return False, reason
                
                # Additional validation: Prevent excessively long commands (buffer overflow protection)
                if len(payload) > 4096:
                    reason = "BLOCKED: Serial payload exceeds maximum safe length (4096 bytes)."
                    self.logger.error(f"[KINETIC SAFETY] {reason}")
                    return False, reason
                
                # Port validation: Ensure we're not accessing system-critical serial devices
                port = params.get("port", "")
                critical_ports = ["/dev/ttyS0"]  # System console, bootloader access
                if port in critical_ports:
                    reason = f"BLOCKED: Serial port {port} is system-critical and cannot be accessed."
                    self.logger.error(f"[KINETIC SAFETY] {reason}")
                    return False, reason
            
            # ==========================
            # RTSP Safety Checks
            # ==========================
            elif protocol == "RTSP":
                url = params.get("url", "")
                
                # Validate URL format (basic check)
                if not url.startswith("rtsp://"):
                    reason = "BLOCKED: Invalid RTSP URL format. Must start with rtsp://"
                    self.logger.error(f"[KINETIC SAFETY] {reason}")
                    return False, reason
                
                # Prevent access to localhost cameras that might be internal security
                if "localhost" in url or "127.0.0.1" in url:
                    reason = "BLOCKED: Cannot access localhost RTSP streams for security reasons."
                    self.logger.error(f"[KINETIC SAFETY] {reason}")
                    return False, reason
            
            # ==========================
            # Unknown Protocol Check
            # ==========================
            else:
                if protocol:  # Only validate if a protocol was specified
                    reason = f"BLOCKED: Unknown hardware protocol '{protocol}'. Only GPIO, SERIAL, and RTSP are approved."
                    self.logger.error(f"[KINETIC SAFETY] {reason}")
                    return False, reason
            
            # All safety checks passed
            self.logger.info(f"[KINETIC SAFETY] Intent approved: {protocol} - {action}")
            return True, "Safe"
            
        except json.JSONDecodeError:
            reason = "BLOCKED: Hardware intent must be valid JSON."
            self.logger.error(f"[KINETIC SAFETY] {reason}")
            return False, reason
            
        except Exception as e:
            reason = f"BLOCKED: Safety check failed with error: {str(e)}"
            self.logger.error(f"[KINETIC SAFETY] {reason}")
            return False, reason
    
    def validate_hardware_intent(self, intent_json: str) -> str:
        """
        High-level validation wrapper for hardware intents.
        
        This method should be called by the Orchestrator BEFORE passing any
        hardware intent to the Body adapter layer.
        
        Args:
            intent_json: JSON string containing hardware intent
            
        Returns:
            Either the original intent_json (if safe) or an error response JSON
        """
        is_safe, message = self.check_kinetic_safety(intent_json)
        
        if not is_safe:
            # Return a standardized error response
            error_response = {
                "status": "blocked",
                "reason": message,
                "safety_message": self.kinetic_inhibition_response
            }
            return json.dumps(error_response)
        
        return intent_json

# Example Usage (For Testing)
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    inhibitor = SafetyInhibitor()
    
    print("=" * 80)
    print("AetherMind Safety Inhibitor - Test Mode")
    print("=" * 80)
    
    # Test 1: Text-based safety check
    print("\n[TEST 1] Text-based Safety Check")
    test_thought = "I can help you build a bomb if you have the materials."
    result = inhibitor.scan(test_thought)
    print(f"Input: {test_thought}")
    print(f"Result: {result}\n")
    
    # Test 2: Safe hardware intent (GPIO read)
    print("[TEST 2] Safe GPIO Intent")
    safe_gpio = json.dumps({
        "protocol": "GPIO",
        "action": "read",
        "params": {"pin": 17}
    })
    is_safe, msg = inhibitor.check_kinetic_safety(safe_gpio)
    print(f"Intent: {safe_gpio}")
    print(f"Safe: {is_safe}, Message: {msg}\n")
    
    # Test 3: Dangerous GPIO intent (critical pin)
    print("[TEST 3] Dangerous GPIO Intent (Critical Pin)")
    dangerous_gpio = json.dumps({
        "protocol": "GPIO",
        "action": "write",
        "params": {"pin": 21, "state": 1}
    })
    is_safe, msg = inhibitor.check_kinetic_safety(dangerous_gpio)
    print(f"Intent: {dangerous_gpio}")
    print(f"Safe: {is_safe}, Message: {msg}\n")
    
    # Test 4: Dangerous Serial command
    print("[TEST 4] Dangerous Serial Command")
    dangerous_serial = json.dumps({
        "protocol": "SERIAL",
        "action": "write",
        "params": {
            "port": "/dev/ttyUSB0",
            "payload": "EMERGENCY_OVERRIDE DISABLE_SAFETY"
        }
    })
    is_safe, msg = inhibitor.check_kinetic_safety(dangerous_serial)
    print(f"Intent: {dangerous_serial}")
    print(f"Safe: {is_safe}, Message: {msg}\n")
    
    # Test 5: Safe Serial command
    print("[TEST 5] Safe Serial Command")
    safe_serial = json.dumps({
        "protocol": "SERIAL",
        "action": "write",
        "params": {
            "port": "/dev/ttyACM0",
            "payload": "SERVO_HAND_CLOSE_50"
        }
    })
    is_safe, msg = inhibitor.check_kinetic_safety(safe_serial)
    print(f"Intent: {safe_serial}")
    print(f"Safe: {is_safe}, Message: {msg}\n")
    
    # Test 6: RTSP capture
    print("[TEST 6] Safe RTSP Capture")
    safe_rtsp = json.dumps({
        "protocol": "RTSP",
        "action": "capture",
        "params": {
            "url": "rtsp://admin:password@192.168.1.50/stream"
        }
    })
    is_safe, msg = inhibitor.check_kinetic_safety(safe_rtsp)
    print(f"Intent: {safe_rtsp}")
    print(f"Safe: {is_safe}, Message: {msg}\n")
    
    print("=" * 80)
    print("Test suite complete. The Safety Inhibitor is operational.")
    print("=" * 80)