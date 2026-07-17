use std::path::PathBuf;
use std::process::Command;

fn example(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples")
        .join(name)
}

#[test]
fn help_is_available() {
    let output = Command::new(env!("CARGO_BIN_EXE_rangeforge"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("analyze"));
}

#[test]
fn model_and_scenario_validation_work() {
    let model = example("toy_postflop_v1.json");
    let scenario = example("flop_large_bet.json");
    let model_output = Command::new(env!("CARGO_BIN_EXE_rangeforge"))
        .args(["validate-model", model.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(model_output.status.success());

    let scenario_output = Command::new(env!("CARGO_BIN_EXE_rangeforge"))
        .args([
            "validate",
            scenario.to_str().unwrap(),
            "--model",
            model.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(scenario_output.status.success());
}

#[test]
fn json_analysis_is_machine_readable() {
    let model = example("toy_postflop_v1.json");
    let scenario = example("turn_exact.json");
    let output = Command::new(env!("CARGO_BIN_EXE_rangeforge"))
        .args([
            "analyze",
            scenario.to_str().unwrap(),
            "--model",
            model.to_str().unwrap(),
            "--format",
            "json",
            "--top",
            "3",
        ])
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["schema_version"], "0.1");
    assert_eq!(value["equity"]["method"], "exact");
    assert_eq!(value["top_combos"].as_array().unwrap().len(), 3);
}
