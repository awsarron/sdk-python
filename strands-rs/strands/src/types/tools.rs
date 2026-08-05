//! Tool specification types. Ports `tools/types.ts`.

use serde::{Deserialize, Serialize};

/// Specification for a tool the model can use.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolSpec {
    /// The unique name of the tool.
    pub name: String,
    /// A description of what the tool does, to help the model decide when to use it.
    pub description: String,
    /// JSON Schema describing the tool's input. Defaults to an empty object schema when absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<serde_json::Value>,
    /// JSON Schema describing the tool's output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<serde_json::Value>,
}

/// Specifies how the model should choose which tool to use.
///
/// Mirrors the TypeScript union `{ auto: {} } | { any: {} } | { tool: { name } }`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ToolChoice {
    /// Let the model decide whether to use a tool.
    Auto,
    /// Force the model to use one of the available tools.
    Any,
    /// Force the model to use a specific tool by name.
    Tool {
        /// The name of the tool the model must use.
        name: String,
    },
}
