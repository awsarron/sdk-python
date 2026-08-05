//! Builder for [`Agent`]. Ports the constructor-config surface of `agent/agent.ts`
//! into the idiomatic Rust builder pattern.

use std::sync::Arc;

use crate::agent::Agent;
use crate::errors::StrandsError;
use crate::models::Model;
use crate::tools::{Tool, ToolRegistry};
use crate::types::messages::{Message, SystemPrompt};

/// Builder for constructing an [`Agent`].
#[derive(Default)]
pub struct AgentBuilder {
    model: Option<Box<dyn Model>>,
    system_prompt: Option<SystemPrompt>,
    messages: Vec<Message>,
    tool_registry: ToolRegistry,
}

impl AgentBuilder {
    /// Creates an empty builder.
    pub fn new() -> Self {
        AgentBuilder::default()
    }

    /// Sets the model provider that drives the agent loop.
    pub fn model(mut self, model: impl Model + 'static) -> Self {
        self.model = Some(Box::new(model));
        self
    }

    /// Sets the model provider from an already-boxed trait object. Useful when
    /// the concrete model type is chosen at runtime.
    pub fn model_boxed(mut self, model: Box<dyn Model>) -> Self {
        self.model = Some(model);
        self
    }

    /// Sets the system prompt.
    pub fn system_prompt(mut self, system_prompt: impl Into<SystemPrompt>) -> Self {
        self.system_prompt = Some(system_prompt.into());
        self
    }

    /// Seeds the conversation with initial messages.
    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = messages;
        self
    }

    /// Registers a tool. Ignores a tool whose name fails registry validation;
    /// use [`AgentBuilder::try_tool`] to surface the error.
    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        let _ = self.tool_registry.add(Arc::new(tool));
        self
    }

    /// Registers a tool, returning a validation error if the name is invalid or
    /// conflicts with an existing tool.
    pub fn try_tool(mut self, tool: impl Tool + 'static) -> Result<Self, StrandsError> {
        self.tool_registry.add(Arc::new(tool))?;
        Ok(self)
    }

    /// Builds the agent.
    ///
    /// # Panics
    /// Panics if no model was set. Use [`AgentBuilder::try_build`] for a
    /// non-panicking variant.
    pub fn build(self) -> Agent {
        self.try_build()
            .expect("Agent requires a model; call .model(...) before .build()")
    }

    /// Builds the agent, returning an error if no model was set.
    pub fn try_build(self) -> Result<Agent, StrandsError> {
        let model = self.model.ok_or_else(|| {
            StrandsError::model("Agent requires a model; call .model(...) before building")
        })?;
        Ok(Agent::new(
            model,
            self.system_prompt,
            self.messages,
            self.tool_registry,
        ))
    }
}
