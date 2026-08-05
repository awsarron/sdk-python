//! Streaming event types for model interactions. Ports `models/streaming.ts`.
//!
//! The TypeScript module models each event as a `Data` interface plus a class
//! carrying a `type` discriminator. Rust collapses these into a single
//! discriminated `enum ModelStreamEvent` whose variants carry the payload
//! structs directly.

use serde::{Deserialize, Serialize};

use crate::types::messages::{Role, StopReason};

/// Token usage statistics for a model invocation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    /// Number of tokens in the input (prompt).
    pub input_tokens: u64,
    /// Number of tokens in the output (completion).
    pub output_tokens: u64,
    /// Total number of tokens (input + output).
    pub total_tokens: u64,
    /// Number of input tokens read from cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u64>,
    /// Number of input tokens written to cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write_input_tokens: Option<u64>,
}

impl Usage {
    /// Accumulates another `Usage` into this one, mirroring `accumulateUsage`.
    pub fn accumulate(&mut self, source: &Usage) {
        self.input_tokens += source.input_tokens;
        self.output_tokens += source.output_tokens;
        self.total_tokens += source.total_tokens;
        if let Some(cache_read) = source.cache_read_input_tokens {
            self.cache_read_input_tokens =
                Some(self.cache_read_input_tokens.unwrap_or(0) + cache_read);
        }
        if let Some(cache_write) = source.cache_write_input_tokens {
            self.cache_write_input_tokens =
                Some(self.cache_write_input_tokens.unwrap_or(0) + cache_write);
        }
    }
}

/// Performance metrics for a model invocation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Metrics {
    /// End-to-end latency in milliseconds.
    pub latency_ms: u64,
    /// Latency from request to first content chunk, in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_to_first_byte_ms: Option<u64>,
}

/// Information about a tool-use content block that is starting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolUseStart {
    /// The name of the tool being used.
    pub name: String,
    /// Unique identifier for this tool use.
    pub tool_use_id: String,
    /// Reasoning signature from thinking models, preserved for multi-turn tool use.
    pub reasoning_signature: Option<String>,
}

/// An incremental chunk of content within a content block.
#[derive(Debug, Clone, PartialEq)]
pub enum ContentBlockDelta {
    /// Incremental text content.
    Text(String),
    /// A partial JSON string of the tool input.
    ToolUseInput(String),
    /// Incremental reasoning content.
    Reasoning {
        /// Incremental reasoning text.
        text: Option<String>,
        /// Incremental signature data.
        signature: Option<String>,
        /// Incremental redacted content data.
        redacted_content: Option<Vec<u8>>,
    },
}

/// A streaming event emitted by a model provider.
///
/// Discriminated union mirroring the TypeScript `ModelStreamEvent`. Providers
/// emit these raw; [`crate::models::Model::stream_aggregated`] accumulates them
/// into complete content blocks and a final message.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelStreamEvent {
    /// A new message is starting.
    MessageStart {
        /// The role of the message being started.
        role: Role,
    },
    /// A new content block is starting.
    ContentBlockStart {
        /// Present only for tool-use blocks.
        start: Option<ToolUseStart>,
    },
    /// New incremental content within the current block.
    ContentBlockDelta {
        /// The incremental content update.
        delta: ContentBlockDelta,
    },
    /// The current content block has completed.
    ContentBlockStop,
    /// The message has completed.
    MessageStop {
        /// Why generation stopped.
        stop_reason: StopReason,
    },
    /// Metadata about the stream (usage, metrics).
    Metadata {
        /// Token usage information.
        usage: Option<Usage>,
        /// Performance metrics.
        metrics: Option<Metrics>,
    },
}
