"""
LLM Streaming Module
Streaming text generation using OpenAI GPT-4
"""
import asyncio
import json
from openai import AsyncOpenAI
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pathlib import Path
import config

# Try importing httpx for direct HTTP requests (for Azure OpenAI)
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

class LLMStream:
    """Streaming language model processor"""
    
    def __init__(self):
        # Initialize OpenAI client (Azure or standard)
        if config.USE_AZURE:
            # Validate required Azure OpenAI configuration
            if not all([config.AZURE_OPENAI_ENDPOINT, config.AZURE_OPENAI_API_KEY, config.AZURE_OPENAI_DEPLOYMENT]):
                raise ValueError(
                    "Azure OpenAI requires AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, "
                    "and AZURE_OPENAI_DEPLOYMENT environment variables"
                )
            
            # Clean up endpoint - remove trailing slashes (like STS implementation)
            endpoint = config.AZURE_OPENAI_ENDPOINT.rstrip('/')
            
            # For Azure OpenAI Chat Completions API, deployment MUST be in base_url path
            # The SDK does NOT automatically add /deployments/{model} when using Azure
            # Format: {endpoint}/openai/deployments/{deployment}
            # The SDK will then construct: {base_url}/chat/completions
            # Final URL: {endpoint}/openai/deployments/{deployment}/chat/completions
            base_url = f"{endpoint}/openai/deployments/{config.AZURE_OPENAI_DEPLOYMENT}"
            
            # Initialize Azure OpenAI client
            # Try with default_query first (preferred method)
            try:
                self.client = AsyncOpenAI(
                    api_key=config.AZURE_OPENAI_API_KEY,
                    base_url=base_url,
                    default_query={"api-version": config.AZURE_API_VERSION}
                )
            except TypeError:
                # Fallback: include api-version in base_url for older OpenAI library versions
                base_url = f"{base_url}?api-version={config.AZURE_API_VERSION}"
                self.client = AsyncOpenAI(
                    api_key=config.AZURE_OPENAI_API_KEY,
                    base_url=base_url
                )
            
            # Store original endpoint for debugging
            self._azure_endpoint = endpoint
            self._azure_base_url = base_url
            
            # When deployment is in base_url path, we can use a dummy model name or the deployment name
            # The SDK will use the deployment from the base_url path
            self.model = config.AZURE_OPENAI_DEPLOYMENT  # Still use deployment name for compatibility
            
            if config.DEBUG:
                print(f"✓ Using Azure OpenAI Chat Completions API")
                print(f"  Endpoint: {config.AZURE_OPENAI_ENDPOINT}")
                print(f"  Deployment: {config.AZURE_OPENAI_DEPLOYMENT}")
                print(f"  API Version: {config.AZURE_API_VERSION}")
                print(f"  Base URL: {base_url}")
                
                # Validate API version format
                valid_versions = ["2025-01-01-preview", "2024-10-01-preview", "2024-02-15-preview", "2023-12-01-preview", "2023-05-15"]
                if config.AZURE_API_VERSION not in valid_versions:
                    print(f"\n  ℹ️  Using API version: {config.AZURE_API_VERSION}")
                    print(f"     (Common versions: {', '.join(valid_versions[:3])})")
                
                print(f"\n  ⚠️  IMPORTANT: The deployment name '{config.AZURE_OPENAI_DEPLOYMENT}'")
                print(f"     must be a Chat Completions deployment, NOT a Realtime API deployment.")
                print(f"     If you see 404 errors, check:")
                print(f"     1. Deployment exists in Azure Portal")
                print(f"     2. Deployment is for 'Chat Completions' (not Realtime API)")
                print(f"     3. Deployment name matches exactly (case-sensitive)")
                print(f"     4. API version is valid (try: 2024-10-01-preview)")
                print(f"     5. API key has access to this deployment")
        else:
            # Standard OpenAI
            if not config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required when not using Azure")
            
            self.client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
            self.model = config.OPENAI_MODEL
            if config.DEBUG:
                print(f"✓ Using OpenAI: {config.OPENAI_MODEL}")
        
        self.conversation_history = []
        self.system_prompt = config.SYSTEM_PROMPT
        
        # MCP setup
        self.mcp_session = None
        self.exit_stack = AsyncExitStack()
        
    async def initialize(self):
        """Initialize LLM and optional MCP tools"""
        if config.ENABLE_TOOLS:
            await self._connect_mcp()
        
        if config.DEBUG:
            print("✓ LLM initialized")
    
    async def _connect_mcp(self):
        """Connect to MCP server for tool access"""
        try:
            server_path = Path(__file__).parent / config.MCP_SERVER_PATH
            
            if not server_path.exists():
                if config.VERBOSE:
                    print(f"⚠️ MCP server not found at {server_path}")
                return
            
            server_params = StdioServerParameters(
                command="python",
                args=[str(server_path)]
            )
            
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            stdio, write = stdio_transport
            
            self.mcp_session = await self.exit_stack.enter_async_context(
                ClientSession(stdio, write)
            )
            
            await self.mcp_session.initialize()
            
            tools_result = await self.mcp_session.list_tools()
            
            if config.DEBUG:
                print(f"✓ MCP connected ({len(tools_result.tools)} tools)")
                
        except Exception as e:
            if config.VERBOSE:
                print(f"⚠️ MCP connection failed: {e}")
    
    async def _get_mcp_tools(self):
        """Get available MCP tools in OpenAI format"""
        if not self.mcp_session:
            return []
        
        tools_result = await self.mcp_session.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in tools_result.tools
        ]
    
    async def _azure_direct_http(self, messages, tools, stream_callback):
        """Use direct HTTP requests for Azure OpenAI (SDK doesn't handle deployment in base_url correctly)"""
        if not HAS_HTTPX:
            raise RuntimeError("httpx is required for Azure OpenAI direct HTTP mode. Install with: pip install httpx")
        
        endpoint = config.AZURE_OPENAI_ENDPOINT.rstrip('/')
        url = f"{endpoint}/openai/deployments/{config.AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={config.AZURE_API_VERSION}"
        
        headers = {
            "api-key": config.AZURE_OPENAI_API_KEY,
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": messages,
            "temperature": config.OPENAI_TEMPERATURE,
            "max_tokens": config.OPENAI_MAX_TOKENS
        }
        
        # Add tools if available (convert to Azure format if needed)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        if config.DEBUG:
            print(f"  📡 Request URL: {url}")
            print(f"  📦 Model/Deployment: {config.AZURE_OPENAI_DEPLOYMENT}")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            if stream_callback:
                # Streaming request
                payload["stream"] = True
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    full_response = ""
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]  # Remove "data: " prefix
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk_data = json.loads(data_str)
                                if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                                    delta = chunk_data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        content = delta["content"]
                                        full_response += content
                                        await stream_callback(content)
                            except json.JSONDecodeError:
                                continue
                    
                    # Add to history
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": full_response
                    })
                    
                    return full_response
            else:
                # Non-streaming request
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
                
                assistant_message = result["choices"][0]["message"]
                response_text = assistant_message.get("content", "")
                
                # Handle tool calls if present
                if assistant_message.get("tool_calls") and self.mcp_session:
                    # Execute tools (similar to SDK path)
                    self.conversation_history.append(assistant_message)
                    
                    for tool_call in assistant_message["tool_calls"]:
                        result = await self.mcp_session.call_tool(
                            tool_call["function"]["name"],
                            arguments=json.loads(tool_call["function"]["arguments"])
                        )
                        
                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": result.content[0].text
                        })
                    
                    # Get final response
                    final_payload = {
                        "messages": [
                            {"role": "system", "content": self.system_prompt},
                            *self.conversation_history
                        ],
                        "temperature": config.OPENAI_TEMPERATURE,
                        "max_tokens": config.OPENAI_MAX_TOKENS
                    }
                    
                    final_response = await client.post(url, headers=headers, json=final_payload)
                    final_response.raise_for_status()
                    final_result = final_response.json()
                    response_text = final_result["choices"][0]["message"]["content"]
                
                # Add to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response_text
                })
                
                return response_text
    
    async def generate_response(self, user_message: str, stream_callback=None):
        """Generate response with optional streaming"""
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Keep conversation history manageable
        if len(self.conversation_history) > config.MAX_CONVERSATION_HISTORY:
            self.conversation_history = self.conversation_history[-config.MAX_CONVERSATION_HISTORY:]
        
        # Prepare messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.conversation_history
        ]
        
        # Get tools
        tools = await self._get_mcp_tools()
        
        try:
            # For Azure OpenAI, use direct HTTP since SDK doesn't handle deployment in base_url correctly
            if config.USE_AZURE and HAS_HTTPX:
                return await self._azure_direct_http(messages, tools, stream_callback)
            
            # For standard OpenAI or if httpx not available, use SDK
            # Prepare request parameters
            request_params = {
                "messages": messages,
                "temperature": config.OPENAI_TEMPERATURE,
                "max_tokens": config.OPENAI_MAX_TOKENS,
                "stream": stream_callback is not None
            }
            
            # For standard OpenAI, always pass model parameter
            if not config.USE_AZURE:
                request_params["model"] = self.model
            else:
                # For Azure with SDK, we need to pass model even though it's in base_url
                # (SDK strips it from base_url and expects it as parameter)
                request_params["model"] = self.model
            
            # Add tools if available
            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"
            
            # Debug: Show request details
            if config.DEBUG:
                if config.USE_AZURE:
                    expected_url = f"{self._azure_endpoint}/openai/deployments/{self.model}/chat/completions?api-version={config.AZURE_API_VERSION}"
                    print(f"  📡 Request URL: {expected_url}")
                print(f"  📦 Model/Deployment: {self.model}")
            
            # Create completion
            response = await self.client.chat.completions.create(**request_params)
            
            # Handle streaming
            if stream_callback:
                full_response = ""
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        await stream_callback(content)
                
                # Add to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": full_response
                })
                
                return full_response
            else:
                # Non-streaming
                assistant_message = response.choices[0].message
                
                # Handle tool calls
                if assistant_message.tool_calls and self.mcp_session:
                    # Execute tools
                    self.conversation_history.append(assistant_message)
                    
                    for tool_call in assistant_message.tool_calls:
                        result = await self.mcp_session.call_tool(
                            tool_call.function.name,
                            arguments=json.loads(tool_call.function.arguments)
                        )
                        
                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result.content[0].text
                        })
                    
                    # Get final response
                    # For Azure OpenAI, deployment is in base_url, so don't pass model parameter
                    final_request_params = {
                        "messages": [
                            {"role": "system", "content": self.system_prompt},
                            *self.conversation_history
                        ],
                        "temperature": config.OPENAI_TEMPERATURE,
                        "max_tokens": config.OPENAI_MAX_TOKENS
                    }
                    
                    # Only add model parameter for standard OpenAI
                    if not config.USE_AZURE:
                        final_request_params["model"] = self.model
                    
                    final_response = await self.client.chat.completions.create(**final_request_params)
                    
                    response_text = final_response.choices[0].message.content
                else:
                    response_text = assistant_message.content
                
                # Add to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response_text
                })
                
                return response_text
                
        except Exception as e:
            error_msg = str(e)
            error_details = ""
            status_code = None
            
            # Extract more details from OpenAI library errors
            if hasattr(e, 'status_code'):
                status_code = e.status_code
            elif hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                status_code = e.response.status_code
            
            # Try to extract error body
            if hasattr(e, 'body'):
                try:
                    if isinstance(e.body, dict):
                        error_details = str(e.body)
                    else:
                        import json
                        error_details = json.loads(e.body) if isinstance(e.body, str) else str(e.body)
                except:
                    error_details = str(e.body)
            elif hasattr(e, 'response') and hasattr(e.response, 'json'):
                try:
                    error_details = e.response.json()
                except:
                    pass
            
            # Format error details
            if error_details:
                if isinstance(error_details, dict):
                    error_details = str(error_details)
                error_details = f"Error code: {status_code} - {error_details}" if status_code else f"Error details: {error_details}"
            elif status_code:
                error_details = f"Error code: {status_code}"
            
            # Print detailed error information
            print(f"❌ LLM error: {error_msg}")
            if error_details:
                print(f"   {error_details}")
            
            # Show full request URL for debugging
            if config.USE_AZURE and config.DEBUG:
                endpoint = config.AZURE_OPENAI_ENDPOINT.rstrip('/')
                base_url = f"{endpoint}/openai"
                full_url = f"{base_url}/chat/completions?api-version={config.AZURE_API_VERSION}"
                print(f"   🔗 Request URL: {full_url}")
            
            # Provide helpful troubleshooting info for 404 errors
            if status_code == 404 or "404" in error_msg or "not found" in error_msg.lower():
                print(f"\n   🔍 Troubleshooting 404 error:")
                print(f"   1. Deployment name: '{self.model}' (verify in Azure Portal)")
                print(f"   2. Deployment type: Must be 'Chat Completions' (NOT Realtime API)")
                print(f"   3. API version: '{config.AZURE_API_VERSION}'")
                print(f"      → Try changing to: 2024-10-01-preview")
                print(f"      → Or try: 2024-02-15-preview")
                print(f"   4. Endpoint: {config.AZURE_OPENAI_ENDPOINT}")
                print(f"   5. API key: Verify it has access to this deployment")
                print(f"\n   💡 Quick fix: Update your .env file:")
                print(f"      AZURE_API_VERSION=2024-10-01-preview")
                
                # If using an invalid API version, suggest trying a known-good one
                if config.USE_AZURE and config.AZURE_API_VERSION not in ["2024-10-01-preview", "2024-02-15-preview", "2023-12-01-preview", "2023-05-15"]:
                    print(f"\n   ⚠️  Your API version '{config.AZURE_API_VERSION}' may be invalid!")
                    print(f"      This is likely the cause of the 404 error.")
            
            return "I apologize, I encountered an error processing your request."
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()
        
        if config.DEBUG:
            print("🧹 LLM cleaned up")


async def test_llm():
    """Test LLM generation"""
    print("Testing LLM stream...")
    
    llm = LLMStream()
    await llm.initialize()
    
    response = await llm.generate_response("Hello! How are you?")
    print(f"✓ Response: {response}")
    
    await llm.cleanup()
    print("✓ LLM test complete")


if __name__ == "__main__":
    asyncio.run(test_llm())

