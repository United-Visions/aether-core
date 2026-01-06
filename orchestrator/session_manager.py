"""
Path: orchestrator/session_manager.py
Role: Managing User Personas, Domain Preferences, and Session State.
"""

from typing import Dict, Optional
from loguru import logger
import sys
import os

# Add config directory to path for domain profiles
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.domain_profiles import get_domain_profile, DomainProfile, MULTI_DOMAIN_MASTER_PROFILE

class SessionManager:
    def __init__(self):
        # In a real app, this would connect to Supabase
        self.user_sessions = {}
        self.user_domains = {}  # user_id -> domain selection
        self.user_learning_context = {}  # user_id -> learned preferences
        logger.info("SessionManager initialized with domain-aware capabilities")

    def get_session(self, user_id: str) -> Dict:
        """
        Get or create a session for a user.
        Session stores temporary state like execution results for feedback loops.
        
        Args:
            user_id: User identifier
            
        Returns:
            Session dictionary with user state
        """
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                "last_execution_results": [],
                "conversation_turns": 0,
                "session_start": None
            }
        return self.user_sessions[user_id]

    def get_user_persona(self, user_id: str) -> Dict:
        """
        Returns the personality settings for a user.
        DEPRECATED: Use get_user_profile() instead for domain-aware settings.
        """
        # Default persona (backward compatibility)
        return self.user_sessions.get(user_id, {
            "name": "User",
            "tone": "helpful and logical",
            "expertise": "generalist"
        })

    def set_user_persona(self, user_id: str, settings: dict):
        """
        Set user persona settings.
        DEPRECATED: Use set_user_domain() for domain-specific configuration.
        """
        self.user_sessions[user_id] = settings
    
    def set_user_domain(self, user_id: str, domain: str):
        """
        Set the user's selected domain during onboarding.
        This fundamentally changes how AetherMind interacts with the user.
        
        Args:
            user_id: User identifier
            domain: Domain ID (code, research, business, legal, finance, general)
        """
        profile = get_domain_profile(domain)
        self.user_domains[user_id] = domain
        
        # Initialize learning context for this user
        if user_id not in self.user_learning_context:
            self.user_learning_context[user_id] = {
                "domain": domain,
                "interaction_count": 0,
                "learned_preferences": {},
                "common_topics": [],
                "communication_adaptations": {}
            }
        
        logger.info(f"User {user_id} domain set to: {profile.display_name}")
    
    def get_user_domain(self, user_id: str) -> str:
        """
        Get the user's selected domain.
        Defaults to 'general' (Multi-Domain Master) if not set.
        """
        return self.user_domains.get(user_id, "general")
    
    def get_domain_profile(self, user_id: str) -> DomainProfile:
        """
        Get the complete domain profile for a user.
        This includes communication style, knowledge priorities, and behavior patterns.
        """
        domain = self.get_user_domain(user_id)
        return get_domain_profile(domain)
    
    def get_user_profile(self, user_id: str) -> Dict:
        """
        Get comprehensive user profile including domain specialization.
        This replaces the deprecated get_user_persona().
        
        Returns:
            {
                "user_id": str,
                "domain": str,
                "domain_profile": DomainProfile,
                "learning_context": dict,
                "interaction_count": int
            }
        """
        domain = self.get_user_domain(user_id)
        profile = get_domain_profile(domain)
        learning_context = self.user_learning_context.get(user_id, {})
        
        return {
            "user_id": user_id,
            "domain": domain,
            "domain_display_name": profile.display_name,
            "domain_profile": profile,
            "learning_context": learning_context,
            "interaction_count": learning_context.get("interaction_count", 0)
        }
    
    def update_learning_context(self, user_id: str, interaction_data: Dict):
        """
        Update the user's learning context based on their interactions.
        This helps AetherMind adapt to the specific user over time.
        
        Args:
            interaction_data: {
                "topic": str,
                "tools_used": List[str],
                "response_quality": float,  # User feedback (optional)
                "domain_relevant": bool,
                "cross_domain": bool
            }
        """
        if user_id not in self.user_learning_context:
            domain = self.get_user_domain(user_id)
            self.user_learning_context[user_id] = {
                "domain": domain,
                "interaction_count": 0,
                "learned_preferences": {},
                "common_topics": [],
                "communication_adaptations": {}
            }
        
        context = self.user_learning_context[user_id]
        context["interaction_count"] += 1
        
        # Track common topics
        if "topic" in interaction_data:
            topic = interaction_data["topic"]
            if topic not in context["common_topics"]:
                context["common_topics"].append(topic)
            
            # Keep only top 20 most recent topics
            if len(context["common_topics"]) > 20:
                context["common_topics"] = context["common_topics"][-20:]
        
        # Track tool preferences
        if "tools_used" in interaction_data:
            if "preferred_tools" not in context["learned_preferences"]:
                context["learned_preferences"]["preferred_tools"] = {}
            
            for tool in interaction_data["tools_used"]:
                context["learned_preferences"]["preferred_tools"][tool] = \
                    context["learned_preferences"]["preferred_tools"].get(tool, 0) + 1
        
        # Track cross-domain thinking
        if interaction_data.get("cross_domain", False):
            context["learned_preferences"]["enjoys_cross_domain"] = \
                context["learned_preferences"].get("enjoys_cross_domain", 0) + 1
        
        logger.debug(f"Updated learning context for {user_id}: {context['interaction_count']} interactions")
    
    def get_namespace_weights(self, user_id: str) -> Dict[str, float]:
        """
        Get the namespace retrieval weights based on user's domain.
        This determines which knowledge sources are prioritized.
        """
        profile = self.get_domain_profile(user_id)
        return profile.namespace_weights
    
    def should_use_tool(self, user_id: str, tool_name: str) -> bool:
        """
        Check if a tool is appropriate for the user's domain.
        Returns True if tool should be prioritized for this domain.
        """
        profile = self.get_domain_profile(user_id)
        return profile.should_use_tool(tool_name)
    
    def get_mega_prompt_prefix(self, user_id: str, additional_context: str = "") -> str:
        """
        Generate the domain-specific mega-prompt prefix for the Brain.
        This shapes how the Brain thinks and communicates.
        Includes structured action-tag system.
        """
        from brain.system_prompts import get_aether_system_prompt
        
        profile = self.get_domain_profile(user_id)
        learning = self.user_learning_context.get(user_id, {})
        
        # Get structured action-tag prompt for this domain
        domain = self.user_domains.get(user_id, "general")
        structured_prompt = get_aether_system_prompt(domain, include_thinking=True)
        
        # Add learned context to the prompt
        learned_context = ""
        if learning.get("interaction_count", 0) > 0:
            learned_context = f"\n\n## Your Learning Context\nYou've had {learning['interaction_count']} interactions with this user."
            
            if learning.get("common_topics"):
                topics = ", ".join(learning["common_topics"][-5:])
                learned_context += f"\nRecent topics: {topics}"
            
            if learning.get("tools_used"):
                tools = ", ".join(set(learning.get("tools_used", [])))
                learned_context += f"\nTools you've used: {tools}"
        
        # Combine: Structured prompt + Domain-specific + Learned context
        full_prompt = f"{structured_prompt}\n\n{profile.get_mega_prompt_prefix(additional_context)}{learned_context}"
        return full_prompt
    
    def get_response_format_preferences(self, user_id: str) -> Dict:
        """Get domain-specific response formatting preferences"""
        profile = self.get_domain_profile(user_id)
        return profile.response_format