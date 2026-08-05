//! Name-keyed registry of tools. Ports `registry/tool-registry.ts`.

use std::sync::Arc;

use crate::errors::StrandsError;
use crate::tools::Tool;
use crate::types::tools::ToolSpec;

const TOOL_NAME_MAX_LENGTH: usize = 64;

/// Registry for managing [`Tool`] instances with name-based operations.
///
/// Insertion order is preserved (a `Vec` of `(name, tool)`), matching the
/// TypeScript SDK's `Map`, so `tool_specs()` reports tools in registration order.
#[derive(Default, Clone)]
pub struct ToolRegistry {
    tools: Vec<(String, Arc<dyn Tool>)>,
}

impl ToolRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        ToolRegistry { tools: Vec::new() }
    }

    /// Registers a tool.
    ///
    /// # Errors
    /// Returns [`StrandsError::ToolValidation`] if the name is invalid, already
    /// registered, or conflicts with an existing name that differs only by
    /// `-`/`_`, mirroring the TypeScript `add` validation.
    pub fn add(&mut self, tool: Arc<dyn Tool>) -> Result<(), StrandsError> {
        let name = tool.name().to_string();
        validate_name(&name)?;
        if tool.description().is_empty() {
            return Err(StrandsError::ToolValidation(
                "Tool description must be a non-empty string".to_string(),
            ));
        }
        if self.get(&name).is_some() {
            return Err(StrandsError::ToolValidation(format!(
                "Tool with name '{name}' already registered"
            )));
        }
        self.check_normalized_conflict(&name)?;
        self.tools.push((name, tool));
        Ok(())
    }

    /// Retrieves a tool by exact name.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, tool)| tool)
    }

    /// Resolves a tool name using the TypeScript resolution order: exact match,
    /// then underscore-to-hyphen substitution, then case-insensitive match.
    ///
    /// # Errors
    /// Returns [`StrandsError::ToolNotFound`] when no tool matches.
    pub fn resolve(&self, name: &str) -> Result<&Arc<dyn Tool>, StrandsError> {
        if let Some(tool) = self.get(name) {
            return Ok(tool);
        }
        if name.contains('_') {
            if let Some((_, tool)) = self
                .tools
                .iter()
                .find(|(key, _)| key.replace('-', "_") == name)
            {
                return Ok(tool);
            }
        }
        let lower = name.to_lowercase();
        if let Some((_, tool)) = self
            .tools
            .iter()
            .find(|(key, _)| key.to_lowercase() == lower)
        {
            return Ok(tool);
        }
        Err(StrandsError::ToolNotFound(name.to_string()))
    }

    /// Removes a tool by name. No-op if the tool does not exist.
    pub fn remove(&mut self, name: &str) {
        self.tools.retain(|(key, _)| key != name);
    }

    /// Returns all registered tools in registration order.
    pub fn list(&self) -> Vec<Arc<dyn Tool>> {
        self.tools
            .iter()
            .map(|(_, tool)| Arc::clone(tool))
            .collect()
    }

    /// Returns the specs of all registered tools, in registration order.
    pub fn tool_specs(&self) -> Vec<ToolSpec> {
        self.tools
            .iter()
            .map(|(_, tool)| tool.tool_spec())
            .collect()
    }

    fn check_normalized_conflict(&self, name: &str) -> Result<(), StrandsError> {
        let normalized = name.replace('-', "_");
        for (existing, _) in &self.tools {
            if existing != name && existing.replace('-', "_") == normalized {
                return Err(StrandsError::ToolValidation(format!(
                    "Tool name '{name}' already exists as '{existing}'. \
                     Cannot add a duplicate tool which differs by a '-' or '_'"
                )));
            }
        }
        Ok(())
    }
}

fn validate_name(name: &str) -> Result<(), StrandsError> {
    if name.is_empty() || name.len() > TOOL_NAME_MAX_LENGTH {
        return Err(StrandsError::ToolValidation(
            "Tool name must be between 1 and 64 characters".to_string(),
        ));
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err(StrandsError::ToolValidation(
            "Tool name must contain only alphanumeric characters, hyphens, and underscores"
                .to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Ports `registry/__tests__/tool-registry.test.ts`. Rust has no `-`/`_`- vs
    //! case-only tension in the mock, so the same named tools are used.

    use super::*;
    use crate::tools::{Tool, ToolContext};
    use crate::types::tools::ToolSpec;
    use async_trait::async_trait;

    struct MockTool {
        name: String,
        description: String,
    }

    impl MockTool {
        fn arc(name: &str) -> Arc<dyn Tool> {
            Arc::new(MockTool {
                name: name.to_string(),
                description: "A valid tool description.".to_string(),
            })
        }

        fn arc_with_description(name: &str, description: &str) -> Arc<dyn Tool> {
            Arc::new(MockTool {
                name: name.to_string(),
                description: description.to_string(),
            })
        }
    }

    #[async_trait]
    impl Tool for MockTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            &self.description
        }
        fn tool_spec(&self) -> ToolSpec {
            ToolSpec {
                name: self.name.clone(),
                description: self.description.clone(),
                input_schema: None,
                output_schema: None,
            }
        }
        async fn invoke(&self, _context: ToolContext) -> Result<serde_json::Value, StrandsError> {
            Ok(serde_json::Value::Null)
        }
    }

    // add: registers a single tool
    #[test]
    fn registers_a_single_tool() {
        let mut registry = ToolRegistry::new();
        registry.add(MockTool::arc("valid-tool")).unwrap();
        assert_eq!(registry.list().len(), 1);
        assert!(registry.get("valid-tool").is_some());
    }

    // add: registers multiple tools in order
    #[test]
    fn registers_tools_in_registration_order() {
        let mut registry = ToolRegistry::new();
        registry.add(MockTool::arc("tool-1")).unwrap();
        registry.add(MockTool::arc("tool-2")).unwrap();
        let names: Vec<_> = registry
            .list()
            .iter()
            .map(|tool| tool.name().to_string())
            .collect();
        assert_eq!(names, vec!["tool-1", "tool-2"]);
    }

    // add: throws ToolValidationError for a duplicate tool name
    #[test]
    fn rejects_duplicate_tool_name() {
        let mut registry = ToolRegistry::new();
        registry.add(MockTool::arc("duplicate")).unwrap();
        let error = registry.add(MockTool::arc("duplicate")).unwrap_err();
        assert!(matches!(error, StrandsError::ToolValidation(_)));
        assert_eq!(
            error.to_string(),
            "Tool with name 'duplicate' already registered"
        );
    }

    // add: throws when a name differs only by '-' vs '_'
    #[test]
    fn rejects_name_differing_only_by_hyphen_underscore() {
        let mut registry = ToolRegistry::new();
        registry.add(MockTool::arc("foo-bar")).unwrap();
        let error = registry.add(MockTool::arc("foo_bar")).unwrap_err();
        assert_eq!(
            error.to_string(),
            "Tool name 'foo_bar' already exists as 'foo-bar'. \
             Cannot add a duplicate tool which differs by a '-' or '_'"
        );
    }

    // add: throws ToolValidationError for an invalid tool name pattern
    #[test]
    fn rejects_invalid_name_pattern() {
        let mut registry = ToolRegistry::new();
        let error = registry.add(MockTool::arc("invalid name!")).unwrap_err();
        assert_eq!(
            error.to_string(),
            "Tool name must contain only alphanumeric characters, hyphens, and underscores"
        );
    }

    // add: throws for a name that is too long / too short
    #[test]
    fn rejects_name_length_out_of_bounds() {
        let mut registry = ToolRegistry::new();
        let long_name = "a".repeat(65);
        let long_error = registry.add(MockTool::arc(&long_name)).unwrap_err();
        assert_eq!(
            long_error.to_string(),
            "Tool name must be between 1 and 64 characters"
        );
        let short_error = registry.add(MockTool::arc("")).unwrap_err();
        assert_eq!(
            short_error.to_string(),
            "Tool name must be between 1 and 64 characters"
        );
    }

    // add: throws ToolValidationError for an empty string description
    #[test]
    fn rejects_empty_description() {
        let mut registry = ToolRegistry::new();
        let error = registry
            .add(MockTool::arc_with_description("tool-1", ""))
            .unwrap_err();
        assert!(matches!(error, StrandsError::ToolValidation(_)));
        assert_eq!(
            error.to_string(),
            "Tool description must be a non-empty string"
        );
    }

    // add: registers a tool with a name at the maximum length
    #[test]
    fn accepts_name_at_maximum_length() {
        let mut registry = ToolRegistry::new();
        let name = "a".repeat(64);
        assert!(registry.add(MockTool::arc(&name)).is_ok());
    }

    // get: returns None for a non-existent tool
    #[test]
    fn get_returns_none_for_missing_tool() {
        let registry = ToolRegistry::new();
        assert!(registry.get("non-existent").is_none());
    }

    // resolve: exact name match
    #[test]
    fn resolve_exact_match() {
        let mut registry = ToolRegistry::new();
        registry.add(MockTool::arc("my-tool")).unwrap();
        assert_eq!(registry.resolve("my-tool").unwrap().name(), "my-tool");
    }

    // resolve: underscore-to-hyphen substitution
    #[test]
    fn resolve_underscore_to_hyphen() {
        let mut registry = ToolRegistry::new();
        registry.add(MockTool::arc("my-tool")).unwrap();
        assert_eq!(registry.resolve("my_tool").unwrap().name(), "my-tool");
    }

    // resolve: case-insensitive match
    #[test]
    fn resolve_case_insensitive() {
        let mut registry = ToolRegistry::new();
        registry.add(MockTool::arc("MyTool")).unwrap();
        assert_eq!(registry.resolve("mytool").unwrap().name(), "MyTool");
    }

    // resolve: prefers exact match over case-insensitive match
    #[test]
    fn resolve_prefers_exact_over_case_insensitive() {
        let mut registry = ToolRegistry::new();
        registry.add(MockTool::arc("mytool")).unwrap();
        registry.add(MockTool::arc("MYTOOL")).unwrap();
        assert_eq!(registry.resolve("mytool").unwrap().name(), "mytool");
    }

    // resolve: throws ToolNotFoundError when no tool matches (and message form)
    #[test]
    fn resolve_missing_tool_errors_with_name() {
        let registry = ToolRegistry::new();
        let error = match registry.resolve("missing") {
            Ok(_) => panic!("expected resolve() to error"),
            Err(error) => error,
        };
        assert!(matches!(error, StrandsError::ToolNotFound(ref name) if name == "missing"));
        assert_eq!(error.to_string(), "Tool 'missing' not found");
    }

    // remove: removes a tool; no-op for a non-existent tool
    #[test]
    fn remove_tool() {
        let mut registry = ToolRegistry::new();
        registry.add(MockTool::arc("remove-me")).unwrap();
        registry.remove("remove-me");
        assert!(registry.get("remove-me").is_none());
        registry.remove("non-existent"); // no panic
    }

    // list: empty by default
    #[test]
    fn list_empty_by_default() {
        let registry = ToolRegistry::new();
        assert!(registry.list().is_empty());
    }
}
