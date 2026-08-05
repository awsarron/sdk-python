//! A minimal agent with one tool, backed by Bedrock.
//!
//! Run with AWS credentials configured in the environment:
//! `cargo run --example weather_agent`

use strands_agents::models::BedrockModel;
use strands_agents::tool;
use strands_agents::Agent;

/// Get the current weather for a location.
#[tool]
async fn get_weather(location: String) -> String {
    format!("Weather in {location}: 72F, Sunny")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = BedrockModel::default_model().await;

    let mut agent = Agent::builder()
        .model(model)
        .system_prompt("You are a helpful assistant.")
        .tool(GetWeatherTool::new())
        .build();

    let result = agent.invoke("What's the weather in Seattle?").await?;
    println!("{result}");

    Ok(())
}
