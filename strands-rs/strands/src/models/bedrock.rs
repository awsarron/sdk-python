//! AWS Bedrock model provider. Ports `models/bedrock.ts`.
//!
//! Uses the Bedrock Converse Stream API via the AWS Rust SDK. This slice ports
//! the request formatting (messages, system prompt, tools, inference config) and
//! the streamed-event mapping; guardrails, prompt caching, citations, and native
//! token counting from the TypeScript provider are out of scope.

use async_stream::stream;
use aws_sdk_bedrockruntime::types as brt;
use aws_sdk_bedrockruntime::Client;

use crate::errors::StrandsError;
use crate::models::{Model, ModelEventStream, StreamOptions};
use crate::types::messages::{ContentBlock, Message, StopReason, ToolResultContent};
use crate::types::streaming::{ContentBlockDelta, Metrics, ModelStreamEvent, ToolUseStart, Usage};
use crate::types::tools::{ToolChoice, ToolSpec};

/// Default model ID used when none is configured. Mirrors the TypeScript SDK's
/// Bedrock default.
const DEFAULT_MODEL_ID: &str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0";

/// Substrings that identify a Bedrock context-window-overflow error, mapped to
/// [`StrandsError::ContextWindowOverflow`]. Mirrors `BEDROCK_CONTEXT_WINDOW_OVERFLOW_MESSAGES`.
const CONTEXT_WINDOW_OVERFLOW_MESSAGES: &[&str] = &[
    "Input is too long for requested model",
    "input length and `max_tokens` exceed context limit",
    "too many total text bytes",
];

/// AWS Bedrock implementation of [`Model`], using the Converse Stream API.
pub struct BedrockModel {
    client: Client,
    model_id: String,
    max_tokens: Option<i32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
}

impl BedrockModel {
    /// Creates a Bedrock model with the given model ID, loading AWS config from
    /// the environment (the standard credential/region chain).
    pub async fn new(model_id: impl Into<String>) -> Self {
        let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
        BedrockModel {
            client: Client::new(&config),
            model_id: model_id.into(),
            max_tokens: None,
            temperature: None,
            top_p: None,
        }
    }

    /// Creates a Bedrock model with the default model ID.
    pub async fn default_model() -> Self {
        BedrockModel::new(DEFAULT_MODEL_ID).await
    }

    /// Creates a Bedrock model from an existing client, without touching the
    /// environment. Useful for tests and custom credential setups.
    pub fn from_client(client: Client, model_id: impl Into<String>) -> Self {
        BedrockModel {
            client,
            model_id: model_id.into(),
            max_tokens: None,
            temperature: None,
            top_p: None,
        }
    }

    /// Sets the maximum number of tokens to generate.
    pub fn with_max_tokens(mut self, max_tokens: i32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sets the sampling temperature.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the nucleus-sampling `top_p`.
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    fn build_tool_config(
        &self,
        tool_specs: &[ToolSpec],
        tool_choice: Option<&ToolChoice>,
    ) -> Option<brt::ToolConfiguration> {
        if tool_specs.is_empty() {
            return None;
        }
        let mut tools = Vec::new();
        for spec in tool_specs {
            let schema = spec
                .input_schema
                .clone()
                .unwrap_or_else(|| serde_json::json!({ "type": "object" }));
            let tool_spec = brt::ToolSpecification::builder()
                .name(&spec.name)
                .description(&spec.description)
                .input_schema(brt::ToolInputSchema::Json(json_to_document(&schema)))
                .build()
                .expect("tool name and input schema are always set");
            tools.push(brt::Tool::ToolSpec(tool_spec));
        }

        let mut builder = brt::ToolConfiguration::builder().set_tools(Some(tools));
        if let Some(choice) = tool_choice {
            builder = builder.tool_choice(match choice {
                ToolChoice::Auto => brt::ToolChoice::Auto(brt::AutoToolChoice::builder().build()),
                ToolChoice::Any => brt::ToolChoice::Any(brt::AnyToolChoice::builder().build()),
                ToolChoice::Tool { name } => brt::ToolChoice::Tool(
                    brt::SpecificToolChoice::builder()
                        .name(name)
                        .build()
                        .expect("tool name is set"),
                ),
            });
        }
        builder.build().ok()
    }
}

#[async_trait::async_trait]
impl Model for BedrockModel {
    fn model_id(&self) -> Option<&str> {
        Some(&self.model_id)
    }

    fn stream<'a>(
        &'a self,
        messages: &'a [Message],
        options: &'a StreamOptions,
    ) -> ModelEventStream<'a> {
        Box::pin(stream! {
            let bedrock_messages = match format_messages(messages) {
                Ok(messages) => messages,
                Err(error) => {
                    yield Err(error);
                    return;
                }
            };

            let mut request = self.client
                .converse_stream()
                .model_id(&self.model_id)
                .set_messages(Some(bedrock_messages));

            if let Some(prompt) = &options.system_prompt {
                request = request.system(brt::SystemContentBlock::Text(prompt.clone()));
            }

            if let Some(tool_config) = self.build_tool_config(&options.tool_specs, options.tool_choice.as_ref()) {
                request = request.tool_config(tool_config);
            }

            let mut inference = brt::InferenceConfiguration::builder();
            if let Some(max_tokens) = self.max_tokens {
                inference = inference.max_tokens(max_tokens);
            }
            if let Some(temperature) = self.temperature {
                inference = inference.temperature(temperature);
            }
            if let Some(top_p) = self.top_p {
                inference = inference.top_p(top_p);
            }
            request = request.inference_config(inference.build());

            let mut response = match request.send().await {
                Ok(response) => response,
                Err(error) => {
                    let is_throttling = error.as_service_error().is_some_and(|service| service.is_throttling_exception());
                    yield Err(map_bedrock_error(&error, is_throttling, "bedrock converse stream request failed"));
                    return;
                }
            };

            loop {
                match response.stream.recv().await {
                    Ok(Some(output)) => {
                        for event in map_stream_output(output) {
                            yield Ok(event);
                        }
                    }
                    Ok(None) => break,
                    Err(error) => {
                        let is_throttling = error.as_service_error().is_some_and(|service| service.is_throttling_exception());
                        yield Err(map_bedrock_error(&error, is_throttling, "error receiving bedrock stream event"));
                        return;
                    }
                }
            }
        })
    }
}

/// Maps a Bedrock error into a typed [`StrandsError`].
///
/// Ports the vendor-error translation from `bedrock.ts`: throttling →
/// [`StrandsError::ModelThrottled`], context-window-overflow messages →
/// [`StrandsError::ContextWindowOverflow`], everything else → a generic model
/// error wrapping the cause. Applied on both the request and stream-receive
/// paths, matching the TypeScript provider's single outer catch that covers
/// both request-time and mid-stream failures.
fn map_bedrock_error<E>(error: &E, is_throttling: bool, context: &str) -> StrandsError
where
    E: std::error::Error,
{
    let message = format!(
        "{}",
        aws_smithy_types::error::display::DisplayErrorContext(error)
    );
    if is_throttling {
        return StrandsError::ModelThrottled {
            message,
            source: None,
        };
    }
    if CONTEXT_WINDOW_OVERFLOW_MESSAGES
        .iter()
        .any(|needle| message.contains(needle))
    {
        return StrandsError::ContextWindowOverflow(message);
    }
    StrandsError::model(format!("{context}: {message}"))
}

/// Maps one Bedrock `ConverseStreamOutput` chunk to zero or more SDK events.
///
/// Ports `_mapStreamedBedrockEventToSDKEvent`. Unknown or unhandled event types
/// yield nothing, matching the TypeScript default branch's warn-and-skip.
fn map_stream_output(output: brt::ConverseStreamOutput) -> Vec<ModelStreamEvent> {
    match output {
        brt::ConverseStreamOutput::MessageStart(event) => {
            let role = match event.role {
                brt::ConversationRole::Assistant => crate::types::messages::Role::Assistant,
                _ => crate::types::messages::Role::User,
            };
            vec![ModelStreamEvent::MessageStart { role }]
        }
        brt::ConverseStreamOutput::ContentBlockStart(event) => {
            let start = match event.start {
                Some(brt::ContentBlockStart::ToolUse(tool_use)) => Some(ToolUseStart {
                    name: tool_use.name,
                    tool_use_id: tool_use.tool_use_id,
                    reasoning_signature: None,
                }),
                _ => None,
            };
            vec![ModelStreamEvent::ContentBlockStart { start }]
        }
        brt::ConverseStreamOutput::ContentBlockDelta(event) => {
            let Some(delta) = event.delta else {
                return Vec::new();
            };
            match delta {
                brt::ContentBlockDelta::Text(text) => {
                    vec![ModelStreamEvent::ContentBlockDelta {
                        delta: ContentBlockDelta::Text(text),
                    }]
                }
                brt::ContentBlockDelta::ToolUse(tool_use) => {
                    vec![ModelStreamEvent::ContentBlockDelta {
                        delta: ContentBlockDelta::ToolUseInput(tool_use.input),
                    }]
                }
                brt::ContentBlockDelta::ReasoningContent(reasoning) => match reasoning {
                    brt::ReasoningContentBlockDelta::Text(text) => {
                        vec![ModelStreamEvent::ContentBlockDelta {
                            delta: ContentBlockDelta::Reasoning {
                                text: Some(text),
                                signature: None,
                                redacted_content: None,
                            },
                        }]
                    }
                    brt::ReasoningContentBlockDelta::Signature(signature) => {
                        vec![ModelStreamEvent::ContentBlockDelta {
                            delta: ContentBlockDelta::Reasoning {
                                text: None,
                                signature: Some(signature),
                                redacted_content: None,
                            },
                        }]
                    }
                    _ => Vec::new(),
                },
                _ => Vec::new(),
            }
        }
        brt::ConverseStreamOutput::ContentBlockStop(_) => vec![ModelStreamEvent::ContentBlockStop],
        brt::ConverseStreamOutput::MessageStop(event) => {
            vec![ModelStreamEvent::MessageStop {
                stop_reason: map_stop_reason(&event.stop_reason),
            }]
        }
        brt::ConverseStreamOutput::Metadata(event) => {
            let usage = event.usage.map(|usage| Usage {
                input_tokens: usage.input_tokens.max(0) as u64,
                output_tokens: usage.output_tokens.max(0) as u64,
                total_tokens: usage.total_tokens.max(0) as u64,
                cache_read_input_tokens: usage
                    .cache_read_input_tokens
                    .map(|value| value.max(0) as u64),
                cache_write_input_tokens: usage
                    .cache_write_input_tokens
                    .map(|value| value.max(0) as u64),
            });
            let metrics = event.metrics.map(|metrics| Metrics {
                latency_ms: metrics.latency_ms.max(0) as u64,
                time_to_first_byte_ms: None,
            });
            vec![ModelStreamEvent::Metadata { usage, metrics }]
        }
        _ => Vec::new(),
    }
}

/// Maps a Bedrock stop reason to the SDK's [`StopReason`]. Ports `STOP_REASON_MAP`
/// and the `_transformStopReason` fallback (unknown values pass through as-is).
fn map_stop_reason(reason: &brt::StopReason) -> StopReason {
    match reason {
        brt::StopReason::EndTurn => StopReason::EndTurn,
        brt::StopReason::ToolUse => StopReason::ToolUse,
        brt::StopReason::MaxTokens => StopReason::MaxTokens,
        brt::StopReason::StopSequence => StopReason::StopSequence,
        brt::StopReason::ContentFiltered => StopReason::ContentFiltered,
        brt::StopReason::GuardrailIntervened => StopReason::GuardrailIntervened,
        other => StopReason::from_wire(other.as_str()),
    }
}

/// Formats SDK messages into Bedrock `Message`s. Ports `_formatMessages` /
/// `_formatContentBlock`. Empty messages are dropped, matching the TypeScript
/// `content.length > 0` guard.
fn format_messages(messages: &[Message]) -> Result<Vec<brt::Message>, StrandsError> {
    let mut formatted = Vec::new();
    for message in messages {
        let mut content = Vec::new();
        for block in &message.content {
            if let Some(bedrock_block) = format_content_block(block)? {
                content.push(bedrock_block);
            }
        }
        if content.is_empty() {
            continue;
        }
        let role = match message.role {
            crate::types::messages::Role::User => brt::ConversationRole::User,
            crate::types::messages::Role::Assistant => brt::ConversationRole::Assistant,
        };
        let bedrock_message = brt::Message::builder()
            .role(role)
            .set_content(Some(content))
            .build()
            .map_err(|error| {
                StrandsError::model_with_source("failed to build bedrock message", error)
            })?;
        formatted.push(bedrock_message);
    }
    Ok(formatted)
}

fn format_content_block(block: &ContentBlock) -> Result<Option<brt::ContentBlock>, StrandsError> {
    match block {
        ContentBlock::Text(text) => Ok(Some(brt::ContentBlock::Text(text.clone()))),
        ContentBlock::ToolUse(tool_use) => {
            let bedrock_tool_use = brt::ToolUseBlock::builder()
                .tool_use_id(&tool_use.tool_use_id)
                .name(&tool_use.name)
                .input(json_to_document(&tool_use.input))
                .build()
                .map_err(|error| {
                    StrandsError::model_with_source("failed to build tool use block", error)
                })?;
            Ok(Some(brt::ContentBlock::ToolUse(bedrock_tool_use)))
        }
        ContentBlock::ToolResult(tool_result) => {
            let mut result_content = Vec::new();
            for item in &tool_result.content {
                match item {
                    ToolResultContent::Text(text) => {
                        result_content.push(brt::ToolResultContentBlock::Text(text.clone()));
                    }
                    ToolResultContent::Json(value) => {
                        result_content
                            .push(brt::ToolResultContentBlock::Json(json_to_document(value)));
                    }
                    // Media tool-result content is out of scope for the slice.
                    _ => {}
                }
            }
            let status = match tool_result.status {
                crate::types::messages::ToolResultStatus::Success => brt::ToolResultStatus::Success,
                crate::types::messages::ToolResultStatus::Error => brt::ToolResultStatus::Error,
            };
            let bedrock_result = brt::ToolResultBlock::builder()
                .tool_use_id(&tool_result.tool_use_id)
                .set_content(Some(result_content))
                .status(status)
                .build()
                .map_err(|error| {
                    StrandsError::model_with_source("failed to build tool result block", error)
                })?;
            Ok(Some(brt::ContentBlock::ToolResult(bedrock_result)))
        }
        // Reasoning, cache points, and media blocks are not sent back in the slice.
        _ => Ok(None),
    }
}

/// Converts a `serde_json::Value` into an AWS Smithy `Document`.
fn json_to_document(value: &serde_json::Value) -> aws_smithy_types::Document {
    use aws_smithy_types::{Document, Number};
    match value {
        serde_json::Value::Null => Document::Null,
        serde_json::Value::Bool(boolean) => Document::Bool(*boolean),
        serde_json::Value::Number(number) => {
            if let Some(unsigned) = number.as_u64() {
                Document::Number(Number::PosInt(unsigned))
            } else if let Some(signed) = number.as_i64() {
                Document::Number(Number::NegInt(signed))
            } else {
                Document::Number(Number::Float(number.as_f64().unwrap_or(0.0)))
            }
        }
        serde_json::Value::String(text) => Document::String(text.clone()),
        serde_json::Value::Array(items) => {
            Document::Array(items.iter().map(json_to_document).collect())
        }
        serde_json::Value::Object(map) => Document::Object(
            map.iter()
                .map(|(key, value)| (key.clone(), json_to_document(value)))
                .collect(),
        ),
    }
}
