"""
Path: orchestrator/active_inference.py
Role: The Production Active Inference Loop with Domain-Aware Reasoning.
"""

import numpy as np
import datetime
from brain.logic_engine import LogicEngine
from mind.episodic_memory import EpisodicMemory
from mind.vector_store import AetherVectorStore
from .router import Router
from .session_manager import SessionManager
from heart.heart_orchestrator import Heart
from loguru import logger
from mind.promoter import Promoter
from heart.uncertainty_gate import UncertaintyGate
from config.settings import settings
from orchestrator.action_parser import ActionParser, ActionExecutor
import asyncio
import re

class ActiveInferenceLoop:
    def __init__(self, brain: LogicEngine, memory: EpisodicMemory, store: AetherVectorStore, router: Router, heart: Heart, surprise_detector=None, session_manager=None):
        self.brain = brain
        self.memory = memory
        self.store = store
        self.router = router
        self.heart = heart
        self.surprise_detector = surprise_detector
        self.session_manager = session_manager or SessionManager()
        self.last_trace_data = {} # Simple cache for the trace
        self.promoter = Promoter(store, UncertaintyGate(self.heart.reward_model))
        self.activity_events = []  # Track activities for frontend display
        self.action_parser = ActionParser()  # Parse action tags from Brain
        self.action_executor = ActionExecutor(router, store, memory)  # Execute actions

    async def run_cycle(self, user_id: str, user_input: str, namespace: str = "universal"):
        """
        Production Loop with Full Heart Integration + Domain-Aware Reasoning:
        Sense -> Reason (Domain-Focused) -> Embellish -> Act -> Learn
        """
        # Get user's domain profile
        user_profile = self.session_manager.get_user_profile(user_id)
        domain_profile = user_profile["domain_profile"]
        domain = user_profile["domain"]
        
        logger.info(f"User {user_id} active inference started ({domain_profile.display_name})")

        # Get session data to retrieve previous execution results (feedback loop)
        session_data = self.session_manager.get_session(user_id)
        last_execution_results = session_data.get("last_execution_results", [])
        
        # 1. SENSE: Retrieve context with domain-weighted namespaces
        namespace_weights = domain_profile.namespace_weights
        k12_context, state_vec = await self._domain_aware_context_retrieval(
            user_input, 
            user_id,
            namespace_weights
        )
        logger.debug(f"State vector shape: {len(state_vec)}, Domain: {domain}")

        # Use EpisodicMemory wrapper to get timestamped context
        episodic_context = self.memory.get_recent_context(user_id, user_input)
        
        # 2. FEEL: Compute emotional and moral context from the Heart
        emotion_vector = self.heart.compute_emotion(user_input, user_id)
        predicted_flourishing = self.heart.predict_flourishing(state_vec)

        # Calculate surprise (Agent State)
        surprise_score = 0.0
        is_researching = False
        if self.surprise_detector:
             # Use the state vector for surprise calculation
             try:
                surprise_score = await self.surprise_detector.score(np.array(state_vec))
                is_researching = surprise_score > self.surprise_detector.novelty_threshold
             except Exception as e:
                 logger.warning(f"Surprise detection failed: {e}")

        agent_state = {
            "surprise_score": surprise_score,
            "is_researching": is_researching,
            "reason": "High novelty detected in context." if is_researching else "Routine interaction.",
            "domain": domain,
            "domain_display_name": domain_profile.display_name
        }

        current_time_str = f"Current System Time: {datetime.datetime.now()}"
        combined_context = f"{current_time_str}\n" + "\n".join(k12_context + episodic_context)
        
        # Add execution feedback from previous turn (if any)
        if last_execution_results:
            feedback_str = "\n\n## EXECUTION RESULTS FROM PREVIOUS TURN:\n"
            for i, result in enumerate(last_execution_results, 1):
                status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
                feedback_str += f"\n{i}. {result['action_type']}: {status}\n"
                if result.get("output"):
                    feedback_str += f"   Output: {result['output'][:200]}\n"
                if result.get("error"):
                    feedback_str += f"   Error: {result['error']}\n"
            combined_context += feedback_str
            logger.info(f"Including {len(last_execution_results)} execution results in context for self-healing")

        # 3. BUILD DOMAIN-SPECIFIC MEGA-PROMPT
        domain_mega_prompt = self.session_manager.get_mega_prompt_prefix(user_id)
        
        # 4. REASON: Brain processes input with domain-aware context and personality
        brain_response = await self.brain.generate_thought(
            user_input, 
            combined_context, 
            state_vec, 
            emotion_vector, # Pass the full vector
            predicted_flourishing,
            domain_prompt=domain_mega_prompt  # NEW: Domain-specific instruction
        )

        # Check for actual error responses (not just the text "500" anywhere)
        if brain_response.startswith("ERROR:") or brain_response.startswith("500"):
            return "The Brain is still waking up. Please wait 30 seconds and try again.", None, {}, {}

        # 5. EMBELLISH: Heart adapts the response based on emotion and morals
        embellished_response = self.heart.embellish_response(brain_response, emotion_vector, predicted_flourishing)
        logger.debug(f"Embellished response (first 300 chars): {embellished_response[:300]}")
        
        # 6. EXTRACT THINKING: Parse <think> tags to show agent's reasoning process
        thinking_steps, response_without_thinking = self.action_parser.parse_thinking(embellished_response)
        logger.debug(f"Found {len(thinking_steps)} thinking steps")
        
        # 7. PARSE ACTION TAGS: Extract structured actions from Brain response
        action_tags, cleaned_response = self.action_parser.parse(response_without_thinking)
        logger.info(f"Parsed {len(action_tags)} action tags from response")
        
        # 8. EXECUTE ACTIONS: Run parsed action tags and collect results for feedback
        execution_results = []
        for action_tag in action_tags:
            # Convert to activity event
            activity_event = action_tag.to_activity_event(user_id)
            self.activity_events.append(activity_event)
            
            # Execute the action
            result = await self.action_executor.execute(action_tag, user_id)
            execution_results.append({
                "action_type": action_tag.tag_type,
                "action_description": action_tag.attributes.get("description", ""),
                "success": result["success"],
                "output": result.get("output", ""),
                "error": result.get("error"),
                "metadata": result.get("metadata", {})
            })
            
            # Update activity status
            for event in self.activity_events:
                if event["id"] == activity_event["id"]:
                    event["status"] = "completed" if result["success"] else "error"
                    event["details"] = result["result"] if result["success"] else result["error"]
                    # Add execution details for UI
                    event["data"]["execution_output"] = result.get("output")
                    event["data"]["execution_metadata"] = result.get("metadata", {})
                    break
        
        # 8.5. STORE EXECUTION RESULTS FOR NEXT CYCLE (Feedback Loop)
        # These results will be included in context for the next user message
        # This enables the Brain to see what actually happened and adjust accordingly
        if execution_results:
            session_data["last_execution_results"] = execution_results
            logger.debug(f"Stored {len(execution_results)} execution results for feedback")
        
        # 9. EXTRACT CODE BLOCKS: Find markdown code blocks for display
        code_blocks = self.action_parser.extract_code_blocks(embellished_response)
        if code_blocks and not action_tags:
            # If code blocks exist but no action tags, create file_change activity
            for block in code_blocks:
                self.activity_events.append({
                    "id": f"code_{datetime.datetime.now().timestamp()}",
                    "type": "file_change",
                    "status": "completed",
                    "title": f"Generated {block['language']} code",
                    "details": "Code snippet provided in response",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "data": {
                        "code": block["code"],
                        "language": block["language"],
                        "files": []
                    }
                })
        
        # Track traditional activity patterns (fallback for non-tagged responses)
        self._track_agent_activities(brain_response, user_id)

        # 10. ACT: Route the final, cleaned response to the body
        final_output = self.router.forward_intent(cleaned_response)

        # 11. PREPARE FOR LEARNING: Cache the data needed for the feedback loop
        self.last_trace_data[emotion_vector["message_id"]] = {
            "state_vector": state_vec,
            "action_text": final_output,
            "predicted_flourishing": predicted_flourishing,
            "user_id": user_id,  # Store for promoter
            "thinking_steps": thinking_steps  # Store reasoning process
        }

        
        # 11. LEARN (Episodic): Save the interaction to memory
        self.memory.record_interaction(user_id, "user", user_input)
        self.memory.record_interaction(user_id, "assistant", final_output)
        logger.info(f"Successful interaction saved to user_{user_id}_episodic")
        
        # 12. UPDATE USER LEARNING CONTEXT
        self.session_manager.update_learning_context(user_id, {
            "topic": user_input[:100],  # First 100 chars as topic
            "domain_relevant": True,
            "tools_used": [tag.tag_type for tag in action_tags],  # Track tools used
            "cross_domain": False  # TODO: Detect cross-domain queries
        })

        logger.info(f"Cycle complete for User {user_id} ({domain_profile.display_name})")
        
        # 13. UPDATE AGENT STATE WITH THINKING
        agent_state["thinking_steps"] = thinking_steps
        agent_state["action_count"] = len(action_tags)
        agent_state["activity_events"] = self.activity_events
        
        return final_output, emotion_vector["message_id"], emotion_vector, agent_state
    
    async def _domain_aware_context_retrieval(self, user_input: str, user_id: str, namespace_weights: dict):
        """
        Retrieves context from multiple namespaces with domain-specific weighting.
        
        Args:
            user_input: The user's query
            user_id: User identifier for personalized namespaces
            namespace_weights: Dict of namespace -> weight (e.g., {"core_universal": 0.2, "domain_code": 0.6})
        
        Returns:
            (combined_contexts: List[str], state_vector: List[float])
        """
        all_contexts = []
        weighted_vectors = []
        
        for namespace, weight in namespace_weights.items():
            if weight == 0:
                continue
                
            try:
                # Handle user-specific namespaces
                if namespace.startswith("user_"):
                    actual_namespace = f"user_{user_id}_{namespace.split('_')[-1]}"
                else:
                    actual_namespace = namespace
                
                # Query this namespace
                contexts, state_vec = self.store.query_context(
                    user_input, 
                    namespace=actual_namespace,
                    top_k=max(1, int(5 * weight))  # More results for higher-weighted namespaces
                )
                
                # Weight the contexts
                all_contexts.extend([f"[{namespace}] {ctx}" for ctx in contexts])
                
                # Weight the state vector
                if state_vec and len(state_vec) > 0:
                    weighted_vectors.append((np.array(state_vec), weight))
                    
            except Exception as e:
                logger.warning(f"Failed to retrieve from namespace {namespace}: {e}")
        
        # Combine weighted vectors
        if weighted_vectors:
            final_vec = sum(vec * w for vec, w in weighted_vectors) / sum(w for _, w in weighted_vectors)
            final_vec = final_vec.tolist()
        else:
            # Fallback to core_universal if all else fails
            logger.warning("No weighted vectors found, falling back to core_universal")
            all_contexts, final_vec = self.store.query_context(user_input, namespace="core_universal")
        
        logger.debug(f"Retrieved {len(all_contexts)} contexts from {len(namespace_weights)} namespaces")
        return all_contexts, final_vec

    def close_feedback_loop(self, message_id: str, user_reaction_score: float):
        """
        Called by the API to close the learning loop with user feedback.
        """
        trace_data = self.last_trace_data.pop(message_id, None)
        if trace_data:
            self.heart.close_loop(trace_data, user_reaction_score)
            logger.success(f"Feedback loop closed for message {message_id}. Reward model updated.")

            if settings.promoter_gate:
                cleaned = re.sub(r"\S+@\S+", "", trace_data["action_text"])  # fast PII strip
                asyncio.create_task(
                    self.promoter.nugget_maybe_promote(
                        trace_data.get("user_id", "unknown"),
                        cleaned,
                        surprise=abs(user_reaction_score - trace_data["predicted_flourishing"]),
                        flourish=user_reaction_score
                    )
                )
        else:
            logger.warning(f"Could not find trace data for message {message_id} to close feedback loop.")

    def _track_agent_activities(self, brain_response: str, user_id: str):
        """
        Parse brain response for activities like tool creation, file changes, 
        code execution, sandbox spinup, etc. and track them for frontend display.
        """
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        
        # Clear previous activities
        self.activity_events = []
        
        # Detect tool creation patterns
        if "create tool" in brain_response.lower() or "forge" in brain_response.lower():
            self.activity_events.append({
                "id": f"tool_{timestamp}",
                "type": "tool_creation",
                "status": "in_progress",
                "title": "Creating custom tool",
                "details": "ToolForge generating new capability",
                "timestamp": timestamp,
                "data": {
                    "tool_name": self._extract_tool_name(brain_response),
                    "code": self._extract_code_block(brain_response)
                }
            })
        
        # Detect file creation/modification
        if "create file" in brain_response.lower() or "write to" in brain_response.lower():
            self.activity_events.append({
                "id": f"file_{timestamp}",
                "type": "file_change",
                "status": "in_progress",
                "title": "Creating/modifying files",
                "details": "Writing code to filesystem",
                "timestamp": timestamp,
                "data": {
                    "files": self._extract_file_paths(brain_response),
                    "code": self._extract_code_block(brain_response),
                    "language": self._detect_language(brain_response)
                }
            })
        
        # Detect code execution
        if "execute" in brain_response.lower() or "run" in brain_response.lower() or "sandbox" in brain_response.lower():
            self.activity_events.append({
                "id": f"exec_{timestamp}",
                "type": "code_execution",
                "status": "in_progress",
                "title": "Executing code in sandbox",
                "details": "Running generated code safely",
                "timestamp": timestamp,
                "data": {
                    "code": self._extract_code_block(brain_response),
                    "language": self._detect_language(brain_response),
                    "sandbox_id": f"sandbox_{user_id}_{timestamp}"
                }
            })
        
        logger.debug(f"Tracked {len(self.activity_events)} agent activities")
    
    def get_activity_events(self):
        """Return current activity events for API response"""
        return self.activity_events
    
    def _extract_tool_name(self, text: str) -> str:
        """Extract tool name from response"""
        import re
        match = re.search(r'tool[:\s]+([a-zA-Z_][a-zA-Z0-9_]*)', text, re.IGNORECASE)
        return match.group(1) if match else "custom_tool"
    
    def _extract_file_paths(self, text: str) -> list:
        """Extract file paths from response"""
        import re
        # Match common file path patterns
        paths = re.findall(r'(?:^|\s)([a-zA-Z0-9_/\\.-]+\.py|[a-zA-Z0-9_/\\.-]+\.js|[a-zA-Z0-9_/\\.-]+\.json)', text)
        return paths if paths else ["generated_file.py"]
    
    def _extract_code_block(self, text: str) -> str:
        """Extract code block from markdown-style code fences"""
        import re
        # Match ```language ... ``` blocks
        match = re.search(r'```(?:\w+)?\n(.*?)```', text, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    def _detect_language(self, text: str) -> str:
        """Detect programming language from context"""
        text_lower = text.lower()
        if 'python' in text_lower or 'def ' in text or 'import ' in text:
            return 'python'
        elif 'javascript' in text_lower or 'const ' in text or 'function ' in text:
            return 'javascript'
        elif 'typescript' in text_lower:
            return 'typescript'
        elif 'java' in text_lower:
            return 'java'
        else:
            return 'python'  # Default