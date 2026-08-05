//! Core agent: the event loop, builder, and result type.
//!
//! Ports the essential agent loop from `agent/agent.ts` (`_stream`, `_invokeModel`,
//! `executeTools`) and `AgentResult` from `types/agent.ts`. The hooks, middleware,
//! interrupt, checkpoint, telemetry, session, and structured-output surfaces of
//! the TypeScript loop are out of scope for the vertical slice; the loop control
//! flow they wrap is preserved.

mod builder;
mod result;

pub use builder::AgentBuilder;
pub use result::AgentResult;

use std::sync::Arc;

use crate::errors::StrandsError;
use crate::models::{Model, StreamOptions};
use crate::tools::{execute_tools, ToolRegistry};
use crate::types::messages::{ContentBlock, Message, Role, StopReason, SystemPrompt};

/// Upper bound on agent-loop cycles per invocation.
///
/// The TypeScript loop is bounded by hook-driven limits (`InvokeOptions.limits`);
/// the vertical slice omits that surface, so this guard prevents an unbounded
/// tool-call loop. Reaching it stops the turn with [`StopReason::EndTurn`].
const MAX_LOOP_ITERATIONS: usize = 100;

/// A model-driven agent.
///
/// Drives the loop: send the conversation to the model, and while the model
/// requests tool use, run the tools and feed their results back until the model
/// stops requesting tools.
pub struct Agent {
    /// The conversation history.
    pub messages: Vec<Message>,
    /// The system prompt, if any.
    pub system_prompt: Option<SystemPrompt>,
    model: Box<dyn Model>,
    tool_registry: ToolRegistry,
}

impl Agent {
    /// Returns a builder for constructing an [`Agent`].
    pub fn builder() -> AgentBuilder {
        AgentBuilder::new()
    }

    pub(crate) fn new(
        model: Box<dyn Model>,
        system_prompt: Option<SystemPrompt>,
        messages: Vec<Message>,
        tool_registry: ToolRegistry,
    ) -> Self {
        Agent {
            messages,
            system_prompt,
            model,
            tool_registry,
        }
    }

    /// The tools registered on this agent, in registration order.
    pub fn tools(&self) -> Vec<Arc<dyn crate::tools::Tool>> {
        self.tool_registry.list()
    }

    /// Runs the agent loop with a text prompt, returning the final result.
    ///
    /// Appends the prompt as a user message, then drives the loop to completion.
    pub async fn invoke(&mut self, prompt: impl Into<String>) -> Result<AgentResult, StrandsError> {
        self.invoke_message(Message::user(prompt.into())).await
    }

    /// Runs the agent loop starting from a caller-constructed user message.
    pub async fn invoke_message(&mut self, message: Message) -> Result<AgentResult, StrandsError> {
        self.messages.push(message);
        self.run_loop().await
    }

    async fn run_loop(&mut self) -> Result<AgentResult, StrandsError> {
        let mut iterations = 0;

        loop {
            if iterations >= MAX_LOOP_ITERATIONS {
                let last_message = self.last_message();
                return Ok(AgentResult::new(StopReason::EndTurn, last_message));
            }
            iterations += 1;

            let stream_options = self.build_stream_options();
            let result = self
                .model
                .stream_aggregated(&self.messages, &stream_options)
                .await?;

            if result.stop_reason != StopReason::ToolUse {
                // Normal end of turn: record the assistant message and return.
                self.messages.push(result.message.clone());
                return Ok(AgentResult::new(result.stop_reason, result.message));
            }

            let assistant_message = result.message;
            let tool_result_message = execute_tools(&self.tool_registry, &assistant_message).await;

            // Deferred append: both messages are pushed together after tools run,
            // so history never holds a tool-use without its matching results.
            self.messages.push(assistant_message);
            self.messages.push(tool_result_message);
        }
    }

    fn build_stream_options(&self) -> StreamOptions {
        StreamOptions {
            system_prompt: self.system_prompt.clone(),
            tool_specs: self.tool_registry.tool_specs(),
            tool_choice: None,
        }
    }

    fn last_message(&self) -> Message {
        self.messages
            .last()
            .cloned()
            .unwrap_or_else(|| Message::new(Role::Assistant, vec![ContentBlock::text("")]))
    }
}
