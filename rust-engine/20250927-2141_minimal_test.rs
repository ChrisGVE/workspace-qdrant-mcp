use serde_yaml;
use std::path::Path;

fn main() {
    // Try to parse the configuration and show exactly what's missing
    let config_path = Path::new("20250927-1946_corrected_config.yaml");

    match std::fs::read_to_string(config_path) {
        Ok(content) => {
            println!("File content loaded successfully");

            // Try parsing as raw YAML first
            match serde_yaml::from_str::<serde_yaml::Value>(&content) {
                Ok(value) => {
                    println!("YAML parsed successfully");
                    println!("Root keys: {:?}", value.as_mapping().map(|m| m.keys().collect::<Vec<_>>()));

                    // Check if there's a server field at root level
                    if let Some(mapping) = value.as_mapping() {
                        if mapping.contains_key(&serde_yaml::Value::String("server".to_string())) {
                            println!("Found 'server' field at root level");
                        } else {
                            println!("No 'server' field at root level");
                        }

                        // Check grpc.server
                        if let Some(grpc) = mapping.get(&serde_yaml::Value::String("grpc".to_string())) {
                            if let Some(grpc_map) = grpc.as_mapping() {
                                if grpc_map.contains_key(&serde_yaml::Value::String("server".to_string())) {
                                    println!("Found 'grpc.server' field");
                                } else {
                                    println!("No 'grpc.server' field found");
                                }
                            }
                        }
                    }

                    // Try parsing as DaemonConfig
                    match serde_yaml::from_str::<crate::config::DaemonConfig>(&content) {
                        Ok(_config) => {
                            println!("DaemonConfig parsed successfully!");
                        },
                        Err(e) => {
                            println!("DaemonConfig parsing failed: {}", e);
                            println!("Error location: {:?}", e.location());
                        }
                    }
                },
                Err(e) => {
                    println!("YAML parsing failed: {}", e);
                }
            }
        },
        Err(e) => {
            println!("Failed to read file: {}", e);
        }
    }
}